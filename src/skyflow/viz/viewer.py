"""
The live pygame host (DESIGN.md §13).

`Viewer` owns the window, pacing, keys and pane layout — and nothing else: pixels come
from the shared builders (scenepane/hud/fpv), data arrives as ViewFrames. It never steps
the env and never owns the loop; `frame(state, ...)` is non-blocking and drops to the
display rate, so a sim running faster than the screen costs one small host pull per
DISPLAYED frame, not per step.

By default a background RENDER THREAD owns the window and does all drawing: the feeding
thread only snapshots watch rows and posts them to a latest-wins mailbox, so the sim loop
never pays draw time. Replay and export run synchronous (`threaded=False`) — scrubbing and
video need frame-exact draws in the caller's thread.

Keys: Space pause · ←/→ step/scrub (replay) · [ ] speed · Tab focus world · V pilot-cam
res · T iso/top/profile · G fleet scatter · S screenshot · R reset request · Esc quit.
"""

import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import pygame
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "skyflow.viz needs pygame — install the viz extra: "
        "`uv sync --extra viz` or `pip install 'skyflow[viz]'`"
    ) from e

from skyflow.viz import palette
from skyflow.viz.fpv import PilotCam, compose, obs_mask, upscale
from skyflow.viz.frame import ViewFrame, snapshot
from skyflow.viz.hud import draw_hud
from skyflow.viz.primitives import Grid, Scene
from skyflow.viz.projection import KINDS, Projection
from skyflow.viz.scenepane import draw_scene

__all__ = ["Viewer"]

_PILOT_RES = ((144, 192), (192, 256), (288, 384))  # (H, W) cycles on V
_TOP_H, _HUD_H, _FPV_W = 30, 150, 336
_TRAIL_LEN = 300


class _Mailbox:
    """
    Latest-wins single-slot handoff to the render thread; old frames drop by design.
    Carries either a prebuilt ViewFrame (push) or a raw (state, ...) tuple whose
    snapshot the render thread takes itself (frame).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._item: Any = None
        self._seq = 0

    def put(self, item: Any) -> int:
        with self._lock:
            self._item = item
            self._seq += 1
            self._new.set()
            return self._seq

    def take(self, timeout: float) -> tuple[Any, int]:
        if not self._new.wait(timeout):
            return None, 0
        with self._lock:
            self._new.clear()
            return self._item, self._seq


class Viewer:
    """One window: scene pane, FPV column (pilot + policy), instrument strip."""

    def __init__(
        self,
        scene: Scene,
        *,
        camera: Any = None,
        gates: Any = None,
        watch: tuple[int, ...] = (0,),
        image_shape: tuple[int, int, int] | None = None,
        obs_layout: dict[str, slice] | None = None,
        image_term: str = "mask",
        omega_max: float | None = None,
        control: str = "motors",
        dt: float = 0.01,
        task_state_of: Any = None,
        title: str = "SkyFlow Viz",
        size: tuple[int, int] = (1280, 800),
        display_hz: float = 60.0,
        headless: bool = False,
        frames: int | None = None,
        shot: str | None = None,
        threaded: bool = True,
    ) -> None:
        """
        Args:
          scene: the display world (primitives; binds resolve against each ViewFrame).
          camera: lens/mount for the FPV panes (any CameraModel); None uses the default.
          gates: GateSet for the pilot cam's gate render; None = floor/horizon only.
          watch: fleet rows to snapshot and draw.
          image_shape/obs_layout: enable the policy pane (vision tasks) — the obs image
            block is shown verbatim (DESIGN.md §13 honesty rule).
          image_term: name of the image block in the obs layout (default "mask").
          omega_max: rotor-speed normaliser for glyph arcs and motor bars.
          control: "motors" | "sticks" — picks the stick-cross vs action-bar HUD.
          dt: control period, seconds (turns steps into the clock readout).
          task_state_of: callable state → the task's own pytree (for_env wires
            `env.task_state`, which unwraps the sticks-mode firmware carry).
          display_hz: draw-rate cap; frame() calls beyond it return without drawing.
          headless: force the SDL dummy video driver (CI, screenshots, export).
          frames: auto-close after this many drawn frames (with `shot`, save it first).
          shot: screenshot path written when `frames` runs out.
          threaded: True (default) runs a render thread that owns the window — feeding
            calls never pay draw time. False draws in the caller's thread (replay,
            export, and anything that needs frame-exact draws).
        """
        self.scene = scene
        self.watch = tuple(int(w) for w in watch)
        self.title = title
        self.dt = float(dt)
        self.control = control
        self.omega_max = omega_max
        self.image_shape = image_shape
        self.obs_layout = obs_layout
        self.image_term = image_term
        self._task_state_of = task_state_of
        self._camera = camera
        self._gates = gates
        self._pilot_i = 1
        self._pilot = PilotCam(camera, gates, height=_PILOT_RES[1][0], width=_PILOT_RES[1][1])
        self._policy_floor = (
            PilotCam(camera, None, height=image_shape[0], width=image_shape[1])
            if image_shape is not None
            else None
        )

        self._size = size
        self._headless = headless
        self._display_dt = 1.0 / float(display_hz)
        self._frames_left = frames
        self._shot = shot
        self._open = True
        self.paused = False
        self.speed = 1.0
        self._seek = 0
        self._reset_requested = False
        self.show_fleet = False
        self._proj_kind = "iso"
        self._projs: dict[str, Projection] = {}
        self._focus = 0
        self._last_draw = 0.0
        self._last_submit = 0.0
        self._last_fleet: tuple[float, np.ndarray | None] = (0.0, None)
        self._last_vf: ViewFrame | None = None
        self._trails: dict[int, deque] = {i: deque(maxlen=_TRAIL_LEN) for i in range(len(self.watch))}
        self._hists: dict[str, deque] = {}  # one trace per channel, focused world

        # render thread: owns the window and every pygame call after this point
        self._running = True
        self._thread: threading.Thread | None = None
        self._init_error: Exception | None = None
        self._shot_req: tuple[str, threading.Event] | None = None
        if threaded:
            self._mail = _Mailbox()
            self._cond = threading.Condition()
            self._drawn = 0
            self._ready = threading.Event()
            self._thread = threading.Thread(target=self._run, name="skyflow-viz", daemon=True)
            self._thread.start()
            self._ready.wait(timeout=10.0)
            if self._init_error is not None:
                raise self._init_error
        else:
            self._init_display()

    def _init_display(self) -> None:
        """Window + fonts — called exactly once, in whichever thread owns pygame."""
        if self._headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        pygame.font.init()
        try:
            self._screen = pygame.display.set_mode(self._size)
        except pygame.error:  # no video device: fall back to the dummy driver
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.quit()
            pygame.display.init()
            self._screen = pygame.display.set_mode(self._size)
        pygame.display.set_caption(self.title)
        self._font = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 14)
        self._small = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 11)

    def _run(self) -> None:
        """Render-thread main: draw fresh frames; redraw the last one for liveness."""
        try:
            self._init_display()
        except Exception as e:  # propagate to the constructor
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()
        while self._running:
            item, seq = self._mail.take(0.05)
            if not self._running:
                break
            if item is not None:
                vf = item if isinstance(item, ViewFrame) else self._snap(*item)
                if vf is not None:
                    self._process(vf, force=True)
                with self._cond:
                    self._drawn = seq
                    self._cond.notify_all()
            elif self._last_vf is not None:
                # no fresh frame: keep keys, the PAUSED overlay and the frames budget
                # alive by redrawing the last one (self-throttled to ~20 fps)
                self._process(self._last_vf)
        pygame.display.quit()

    # -- construction --------------------------------------------------------------

    @classmethod
    def for_env(cls, env: Any, watch: tuple[int, ...] = (0,), **kw: Any) -> "Viewer":
        """
        Viewer wired to a SkyFlowEnv: Grid + glyphs + instruments always; the task's
        optional duck-typed `viz_scene()` hook contributes its primitives; `task.gates`
        and `task.camera` (when present) feed the pilot cam and course dots.
        """
        watch = tuple(int(w) for w in watch)
        if any(w < 0 or w >= env.fleet for w in watch):
            raise ValueError(f"watch {watch} out of range for fleet {env.fleet}")
        task = env.task
        scene = kw.pop("scene", None)
        if scene is None:
            hook: Callable[[], list[dict]] | None = getattr(task, "viz_scene", None)
            scene = Scene.from_dicts(hook()) if callable(hook) else Scene()
        if not any(isinstance(p, Grid) for p in scene):
            scene.add(Grid())
        kw.setdefault("camera", getattr(task, "camera", None))
        kw.setdefault("gates", getattr(task, "gates", None))
        kw.setdefault("image_shape", env.image_shape)
        kw.setdefault("obs_layout", env.obs_spec.layout if env.image_shape else None)
        kw.setdefault("omega_max", float(env.airframe.rotor_speed_max))
        kw.setdefault("control", env.cfg.control)
        kw.setdefault("dt", env.dt_control)
        kw.setdefault("task_state_of", getattr(env, "task_state", None))
        kw.setdefault("title", f"SkyFlow Viz — {env.cfg.task} · {env.cfg.control}")
        return cls(scene, watch=watch, **kw)

    # -- host state ------------------------------------------------------------------

    @property
    def open(self) -> bool:
        return self._open

    def take_reset(self) -> bool:
        """True once per R press — the hosting loop decides what a reset means."""
        r, self._reset_requested = self._reset_requested, False
        return r

    def take_seek(self) -> int:
        """Accumulated ←/→ steps since last read (replay hosts consume this)."""
        s, self._seek = self._seek, 0
        return s

    def close(self) -> None:
        self._open = False
        self._running = False
        if self._thread is not None:
            with self._cond:
                self._cond.notify_all()  # unblock forced pushes
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=2.0)
            # the render thread quits the display itself on exit
        else:
            pygame.display.quit()

    # -- feeding ----------------------------------------------------------------------

    def frame(
        self,
        state: Any,
        *,
        obs: Any = None,
        action: Any = None,
        reward: Any = None,
        channels: dict[str, Any] | None = None,
        done: Any = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        """
        Show a live SimState. Non-blocking: between display frames it returns
        immediately; on a threaded viewer even the host transfer happens on the render
        thread (jax arrays are immutable, so the snapshot is safe there — but a step
        jitted with donated buffers may free them, in which case that frame drops).
        """
        if not self._open:
            return
        now = time.perf_counter()
        if now - self._last_submit < self._display_dt:
            return
        self._last_submit = now
        if self._thread is not None:
            self._mail.put((state, obs, action, reward, channels, done, info))
            return
        vf = self._snap(state, obs, action, reward, channels, done, info)
        if vf is not None:
            self._process(vf)

    def _snap(self, state, obs, action, reward, channels, done, info) -> ViewFrame | None:
        """Snapshot + fleet-pull bookkeeping — runs in whichever thread draws."""
        now = time.perf_counter()
        want_fleet = self.show_fleet and now - self._last_fleet[0] > 0.5
        try:
            vf = snapshot(
                state,
                self.watch,
                dt=self.dt,
                obs=obs,
                action=action,
                reward=reward,
                channels=channels,
                done=done,
                info=info,
                task_state=(
                    None if self._task_state_of is None else self._task_state_of(state)
                ),
                focus=self._focus,
                fleet_positions=want_fleet,
            )
        except Exception:  # donated/freed buffers: the frame is gone; drop it
            return None
        if want_fleet:
            self._last_fleet = (now, vf.positions)
        elif self.show_fleet:
            vf.positions = self._last_fleet[1]
        return vf

    def idle(self) -> None:
        """Keep the window live while the hosting loop is paused."""
        if self._thread is None and self._last_vf is not None:
            self._process(self._last_vf)  # the render thread does this by itself
        time.sleep(0.01)

    def wait_closed(self, timeout: float = 10.0) -> bool:
        """Block until the viewer closes (frames budget, Esc); True if it did."""
        deadline = time.perf_counter() + timeout
        while self._open and time.perf_counter() < deadline:
            time.sleep(0.01)
        return not self._open

    def screenshot(self, path: str, timeout: float = 5.0) -> bool:
        """Save the next drawn frame to `path` (thread-safe); True when written."""
        done = threading.Event()
        self._shot_req = (str(path), done)
        if self._thread is None and self._last_vf is not None:
            self._process(self._last_vf, force=True)
        return done.wait(timeout)

    def push(self, vf: ViewFrame, force: bool = False) -> None:
        """
        Show a prebuilt ViewFrame (replay/export enter here). Threaded viewers post it to
        the render thread (latest wins; `force` waits until it is drawn); synchronous
        viewers draw it in place.
        """
        if not self._open:
            return
        if self._thread is not None:
            seq = self._mail.put(vf)
            if force:
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._drawn >= seq or not self._open, timeout=5.0
                    )
            return
        self._process(vf, force)

    def _process(self, vf: ViewFrame, force: bool = False) -> None:
        """Draw one frame — runs only in the thread that owns pygame."""
        if not self._open:
            return
        now = time.perf_counter()
        if not force and vf is self._last_vf and now - self._last_draw < 0.05:
            return
        self._last_draw = now
        vf.focus = self._focus = min(self._focus, vf.plant.shape[0] - 1)
        self._events()
        if not self._open:
            return
        if vf is not self._last_vf:
            self._track(vf)
        self._last_vf = vf
        self._draw(vf)
        pygame.display.flip()
        req, self._shot_req = self._shot_req, None
        if req is not None:
            pygame.image.save(self._screen, req[0])
            req[1].set()
        if self._frames_left is not None:
            self._frames_left -= 1
            if self._frames_left <= 0:
                if self._shot:
                    pygame.image.save(self._screen, self._shot)
                self.close()

    def grab(self) -> np.ndarray:
        """The current window as [H,W,3] uint8 — synchronous viewers only (export)."""
        if self._thread is not None:
            raise RuntimeError("grab() needs a synchronous viewer: pass threaded=False")
        return np.transpose(pygame.surfarray.array3d(self._screen), (1, 0, 2))

    # -- internals ---------------------------------------------------------------------

    def _track(self, vf: ViewFrame) -> None:
        for w in range(vf.plant.shape[0]):
            trail = self._trails.setdefault(w, deque(maxlen=_TRAIL_LEN))
            if vf.done is not None and bool(vf.done[w]):
                trail.clear()  # respawned world: a trail across the jump would lie
            else:
                trail.append(np.asarray(vf.pos[w], np.float64))
        for name, values in vf.channels.items():
            self._hists.setdefault(name, deque(maxlen=240)).append(float(values[vf.focus]))

    def _events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
            elif ev.type == pygame.KEYDOWN:
                self._key(ev.key)

    def _key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.close()
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_TAB:
            self._focus = (self._focus + 1) % len(self.watch)
            self._hists.clear()
        elif key == pygame.K_t:
            self._proj_kind = KINDS[(KINDS.index(self._proj_kind) + 1) % len(KINDS)]
        elif key == pygame.K_g:
            self.show_fleet = not self.show_fleet
        elif key == pygame.K_v:
            self._pilot_i = (self._pilot_i + 1) % len(_PILOT_RES)
            h, w = _PILOT_RES[self._pilot_i]
            self._pilot = PilotCam(self._camera, self._gates, height=h, width=w)
        elif key == pygame.K_s:
            pygame.image.save(self._screen, f"skyflow_viz_{int(time.time())}.png")
        elif key == pygame.K_r:
            self._reset_requested = True
        elif key == pygame.K_LEFT:
            self._seek -= 1
        elif key == pygame.K_RIGHT:
            self._seek += 1
        elif key == pygame.K_LEFTBRACKET:
            self.speed = max(0.125, self.speed / 2.0)
        elif key == pygame.K_RIGHTBRACKET:
            self.speed = min(8.0, self.speed * 2.0)

    def _proj(self, rect: tuple[int, int, int, int]) -> Projection:
        key = self._proj_kind
        if key not in self._projs:
            lo, hi = self.scene.aabb()
            self._projs[key] = Projection.fit(key, lo, hi, rect)
        return self._projs[key]

    def _blit_image(self, img: np.ndarray, x: int, y: int, width: int) -> int:
        """Blit an [H,W,3] image scaled (nearest) to `width` px; returns drawn height."""
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        h = round(width * img.shape[0] / img.shape[1])
        self._screen.blit(pygame.transform.scale(surf, (width, h)), (x, y))
        pygame.draw.rect(self._screen, palette.DIM, pygame.Rect(x, y, width, h), 1)
        return h

    def _draw(self, vf: ViewFrame) -> None:
        screen = self._screen
        wpx, hpx = self._size
        screen.fill(palette.BG)

        # top bar
        left = f"{self.title} · world {self.watch[vf.focus]} ({vf.focus + 1}/{len(self.watch)})"
        right = (
            f"{'PAUSED' if self.paused else f'{self.speed:g}x'} · t {vf.t:7.2f} s"
            f" · step {vf.step}"
        )
        screen.blit(self._font.render(left, True, palette.MUTED), (12, 8))
        r_img = self._font.render(right, True, palette.MUTED)
        screen.blit(r_img, (wpx - r_img.get_width() - 12, 8))
        pygame.draw.aaline(screen, palette.DIM, (0, _TOP_H), (wpx, _TOP_H))

        # panes
        scene_rect = (8, _TOP_H + 8, wpx - _FPV_W - 24, hpx - _TOP_H - _HUD_H - 16)
        draw_scene(
            screen,
            scene_rect,
            self._proj(scene_rect),
            self.scene,
            vf,
            trails={k: list(v) for k, v in self._trails.items()},
            omega_max=self.omega_max,
            label=f"SCENE · {self._proj_kind.upper()}",
            font=self._small,
        )

        fx = wpx - _FPV_W - 8
        y = _TOP_H + 8
        cam = self._pilot.camera
        screen.blit(
            self._small.render(
                f"FPV · PILOT CAM {cam.width}x{cam.height}", True, palette.MUTED
            ),
            (fx, y),
        )
        y += 16
        pilot_img = self._pilot.render(vf.pos[vf.focus], vf.quat[vf.focus])
        y += self._blit_image(pilot_img, fx, y, _FPV_W - 16) + 10

        if self.image_shape is not None and self.obs_layout is not None and vf.obs is not None:
            screen.blit(
                self._small.render(
                    f"FPV · POLICY OBS {self.image_shape[1]}x{self.image_shape[0]} (verbatim)",
                    True,
                    palette.MUTED,
                ),
                (fx, y),
            )
            y += 16
            mask = obs_mask(vf.obs[vf.focus], self.image_shape, self.obs_layout, self.image_term)
            floor = None
            if self._policy_floor is not None:  # display-only backdrop, never an obs
                _, floor = self._policy_floor.channels(vf.pos[vf.focus], vf.quat[vf.focus])
            img = compose(mask, floor)
            k = max(1, (_FPV_W - 16) // img.shape[1])
            self._blit_image(upscale(img, k), fx, y, _FPV_W - 16)

        hud_rect = (0, hpx - _HUD_H, wpx, _HUD_H)
        armed = None
        if vf.info and "armed" in vf.info:
            armed = bool(np.asarray(vf.info["armed"]).reshape(-1)[vf.focus])
        draw_hud(
            screen,
            hud_rect,
            vf,
            control=self.control,
            omega_max=self.omega_max,
            histories={k: list(v) for k, v in self._hists.items()},
            armed=armed,
            font=self._font,
            small=self._small,
        )
