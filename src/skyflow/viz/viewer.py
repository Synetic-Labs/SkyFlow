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

Keys (always listed in a bar along the window's bottom edge): Space pause · ←/→
step/scrub (replay) · [ ] speed · Tab focus world · V iso/top/profile (a switch lands on
a fresh fit) · C follow the focused world · X full glyphs for all watched worlds ·
G fleet scatter · left-drag orbit (the scene's near side follows the cursor) ·
right-drag pan · wheel zoom · P print a screenshot · R reset request (only hosts that
consume take_reset) · Esc quit.
"""

import os
import threading
import time
import traceback
import warnings
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

_PILOT_RES = (192, 256)  # (H, W) of the default pilot cam pane
_TOP_H, _HUD_H, _FPV_W, _KEYBAR_H = 30, 150, 336, 20
_TRAIL_LEN = 300


class _EpTrace:
    """A growing trace that always fits its panel: values append in bins of `bin`
    (mean); when the stored row hits `cap` it halves pairwise and `bin` doubles, so
    the WHOLE history stays visible at bounded cost. Used for the episode-length
    bars and the per-episode channel graphs."""

    def __init__(self, cap: int = 512) -> None:
        self.vals: list[float] = []
        self.bin = 1
        self.cap = int(cap)
        self._part: list[float] = []

    def add(self, v: float) -> None:
        self._part.append(float(v))
        if len(self._part) >= self.bin:
            self.vals.append(sum(self._part) / len(self._part))
            self._part.clear()
            if len(self.vals) >= self.cap:
                pairs = zip(self.vals[::2], self.vals[1::2], strict=True)
                self.vals = [(a + b) / 2.0 for a, b in pairs]
                self.bin *= 2

    def clear(self) -> None:
        self.vals.clear()
        self._part.clear()
        self.bin = 1


class _Mailbox:
    """
    Latest-wins single-slot handoff to the render thread; old frames drop by design.
    Carries either a prebuilt ViewFrame (push) or a raw (state, ...) tuple whose
    snapshot the render thread takes itself (frame). `on_drop` fires (under the lock,
    keep it cheap) for every never-taken item a put replaces — the viewer uses it to
    keep the dropped frame's done flags, so a respawn is never lost.
    """

    def __init__(self, on_drop: Callable[[Any], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._item: Any = None
        self._seq = 0
        self._on_drop = on_drop

    def put(self, item: Any) -> int:
        with self._lock:
            if self._new.is_set() and self._item is not None and self._on_drop is not None:
                self._on_drop(self._item)  # replaced before ever being taken
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
        pilot: Any = None,
        policy_floor: Any = None,
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
          pilot / policy_floor: replacements for the FPV renderers (anything with
            `.camera` and `.render(pos, quat) -> [H,W,3] u8`). The defaults render
            through jax.jit on the DEFAULT DEVICE — inside a live training process
            that call stalls behind the fused chunk, so a trainer must inject
            CPU-pinned (or static) implementations here.
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
        self._pilot = pilot if pilot is not None else PilotCam(
            camera, gates, height=_PILOT_RES[0], width=_PILOT_RES[1]
        )
        if policy_floor is not None:
            self._policy_floor = policy_floor
        else:
            self._policy_floor = (
                PilotCam(camera, None, height=image_shape[0], width=image_shape[1])
                if image_shape is not None
                else None
            )
        self._run_error: Exception | None = None  # a dead render thread's cause
        self.snap_drops = 0  # frames dropped in _snap (first one warns with traceback)

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
        self._drag: str | None = None  # scene-pane mouse mode: "orbit" | "pan"
        self.follow = False  # C: keep the focused world centred in the scene pane
        self.show_glyphs = False  # X: full glyphs for ALL watched worlds, not just focus
        self._gauges: dict[str, float] = {}  # HUD dial full-scales (grow-only)
        self._fed_t: deque[float] = deque(maxlen=120)  # feed timestamps → health fps
        self._drawn_t: deque[float] = deque(maxlen=120)  # fresh-draw timestamps
        # done flags of DROPPED frames (mailbox latest-wins, display throttle): folded
        # into the next drawn frame, so a crash→respawn jump never draws as a trail
        self._lost_dones: deque[tuple[bool, Any]] = deque(maxlen=32)
        self._eps = _EpTrace(cap=2048)  # steps-per-episode bars, focused world
        self._ep_base: int | None = None  # step at the current episode's first sighting
        self._ep_last_step: int | None = None
        self._scene_rect = (
            8, _TOP_H + 8, size[0] - _FPV_W - 24, size[1] - _TOP_H - _HUD_H - _KEYBAR_H - 16
        )
        self._last_draw = 0.0
        self._last_submit = 0.0
        self._last_fleet: tuple[float, np.ndarray | None] = (0.0, None)
        self._last_vf: ViewFrame | None = None
        self._trails: dict[int, deque] = {i: deque(maxlen=_TRAIL_LEN) for i in range(len(self.watch))}
        self._hists: dict[str, _EpTrace] = {}  # one per-episode trace per channel, focused world

        # render thread: owns the window and every pygame call after this point
        self._running = True
        self._thread: threading.Thread | None = None
        self._init_error: Exception | None = None
        self._shot_req: tuple[str, threading.Event] | None = None
        if threaded:
            self._mail = _Mailbox(on_drop=self._stash_done)
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
        try:
            while self._running:
                item, seq = self._mail.take(0.05)
                if not self._running:
                    break
                if item is not None:
                    vf = item if isinstance(item, ViewFrame) else self._snap(*item)
                    if vf is not None:
                        self._merge_lost_dones(vf)
                        self._process(vf, force=True)
                    with self._cond:
                        self._drawn = seq
                        self._cond.notify_all()
                elif self._last_vf is not None:
                    # no fresh frame: keep keys, the PAUSED overlay and the frames budget
                    # alive by redrawing the last one (self-throttled to ~20 fps)
                    self._process(self._last_vf)
                else:
                    # nothing ever arrived: splash instead of a black window, and keep
                    # the event pump alive so Esc works during a long compile
                    self._draw_waiting()
        except Exception as e:
            # A draw failure must not present as a frozen-forever window: record it,
            # mark the viewer closed, release forced pushers, and re-raise loudly so
            # the traceback lands in stderr instead of nowhere.
            self._run_error = e
            self._open = False
            self._running = False
            with self._cond:
                self._cond.notify_all()
            raise
        finally:
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
        self._fed_t.append(now)  # health: the true feed rate, counted before the throttle
        if now - self._last_submit < self._display_dt:
            self._stash_done((state, obs, action, reward, channels, done, info))
            return
        self._last_submit = now
        if self._thread is not None:
            self._mail.put((state, obs, action, reward, channels, done, info))
            return
        vf = self._snap(state, obs, action, reward, channels, done, info)
        if vf is not None:
            self._merge_lost_dones(vf)
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
                # scatter dots saturate visually near ~2k; thin ON DEVICE so a huge
                # fleet never crosses the host bus whole (TECH_DEBT V5)
                fleet_stride=max(1, state.plant.shape[0] // 2048),
            )
        except Exception:
            # Common benign cause: donated/freed device buffers — that frame is gone.
            # But a snapshot bug (bad reward shape, raising task_state_of) lands here
            # too, and silence turned that into "the viewer stopped updating". Count
            # every drop and warn ONCE with the real traceback.
            self.snap_drops += 1
            if self.snap_drops == 1:
                warnings.warn(
                    "viewer dropped a frame in snapshot(); further drops are counted "
                    "silently (viewer.snap_drops). First cause:\n"
                    f"{traceback.format_exc()}",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
        self._fed_t.append(time.perf_counter())
        if self._thread is not None:
            seq = self._mail.put(vf)
            if force:
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._drawn >= seq or not self._open, timeout=5.0
                    )
            return
        self._merge_lost_dones(vf)
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
            self._drawn_t.append(now)
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

    def _stash_done(self, item: Any) -> None:
        """A frame is being dropped whole (mailbox latest-wins or the display
        throttle) — keep its done flags. Without this a respawn between drawn frames
        is invisible: the trail draws a crash→respawn streak and the episode stats
        miss the boundary."""
        is_vf = isinstance(item, ViewFrame)
        done = item.done if is_vf else item[5]
        if done is not None:
            self._lost_dones.append((is_vf, done))

    def _merge_lost_dones(self, vf: ViewFrame) -> None:
        """Fold dropped frames' done flags into the frame that IS drawn."""
        while self._lost_dones:
            is_vf, d = self._lost_dones.popleft()
            try:
                rows = np.asarray(d)  # raw path: pulls the device array
                if not is_vf:
                    rows = rows[list(self.watch)]  # [F] fleet rows → watch rows
                rows = rows.reshape(-1).astype(bool)
            except Exception:
                continue  # donated/freed device buffer: that signal is gone
            if vf.done is not None and rows.shape != vf.done.shape:
                continue
            vf.done = rows if vf.done is None else np.logical_or(vf.done, rows)

    @staticmethod
    def _fps(ts: "deque[float]") -> float:
        """Rate over a timestamp window; 0 until two samples exist."""
        return 0.0 if len(ts) < 2 else (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)

    def _track(self, vf: ViewFrame) -> None:
        # STEPS-PER-EPISODE, focused world. Lengths are step DIFFS against the
        # episode's first seen step, so a global counter (training viz) and a
        # per-episode counter (eval) both read correctly. A step DROP between drawn
        # frames reveals a missed done (frames drop by design).
        step = int(vf.step)
        boundary = False
        if self._ep_last_step is not None and step < self._ep_last_step:
            length = self._ep_last_step - (self._ep_base or 0)
            if length > 0:
                self._eps.add(length)
            self._ep_base = 0  # a resetting counter is already inside the next episode
            boundary = True
        if self._ep_base is None:
            self._ep_base = step  # first sighting: the episode is already in progress
        if vf.done is not None and bool(vf.done[vf.focus]):
            length = step - self._ep_base
            if length > 0:
                self._eps.add(length)
            self._ep_base = None  # re-base on the next frame, whatever the counter does
            self._ep_last_step = None
            boundary = True
        else:
            self._ep_last_step = step
        if boundary:
            for trace in self._hists.values():
                trace.clear()  # channel graphs span ONE episode, compressed to fit
        for w in range(vf.plant.shape[0]):
            trail = self._trails.setdefault(w, deque(maxlen=_TRAIL_LEN))
            if vf.done is not None and bool(vf.done[w]):
                trail.clear()  # respawned world: a trail across the jump would lie
            else:
                trail.append(np.asarray(vf.pos[w], np.float64))
        for name, values in vf.channels.items():
            self._hists.setdefault(name, _EpTrace()).add(float(values[vf.focus]))

    def _events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
            elif ev.type == pygame.KEYDOWN:
                self._key(ev.key)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 2, 3) and self._in_scene(ev.pos):
                self._drag = "orbit" if ev.button == 1 else "pan"
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (1, 2, 3):
                self._drag = None
            elif ev.type == pygame.MOUSEMOTION and self._drag == "orbit":
                # -dx: the scene's NEAR side follows the cursor (the three.js turntable
                # feel); +dx would make the far side follow and the near side fight you
                self._proj(self._scene_rect).orbit(-ev.rel[0] * 0.4, ev.rel[1] * 0.4)
            elif ev.type == pygame.MOUSEMOTION and self._drag == "pan":
                self._proj(self._scene_rect).pan(*ev.rel)
            elif ev.type == pygame.MOUSEWHEEL:
                pos = pygame.mouse.get_pos()
                if self._in_scene(pos):  # zoom about the cursor, scene pane only
                    self._proj(self._scene_rect).zoom_at(*pos, 1.15 ** float(ev.y))

    def _in_scene(self, pos: tuple[int, int]) -> bool:
        return pygame.Rect(self._scene_rect).collidepoint(pos)

    def _key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.close()
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_TAB:
            self._focus = (self._focus + 1) % len(self.watch)
            self._hists.clear()
            self._ep_last_step = None  # the bars persist; the step baseline must not
            self._ep_base = None
        elif key == pygame.K_v:
            self._proj_kind = KINDS[(KINDS.index(self._proj_kind) + 1) % len(KINDS)]
            self._projs.pop(self._proj_kind, None)  # a view switch lands on a fresh fit
        elif key == pygame.K_c:
            self.follow = not self.follow
        elif key == pygame.K_x:
            self.show_glyphs = not self.show_glyphs
        elif key == pygame.K_g:
            self.show_fleet = not self.show_fleet
        elif key == pygame.K_p:
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

    #: (key, action) rows for the always-on key bar — keep in step with _key/_events.
    #: Ordered by importance: a narrow window drops the tail, never the head.
    _KEYS = (
        ("Space", "pause"),
        ("Tab", "focus"),
        ("V", "view"),
        ("C", "follow"),
        ("X", "glyphs"),
        ("G", "fleet"),
        ("drag", "orbit/pan"),
        ("wheel", "zoom"),
        ("P", "print"),
        ("R", "reset"),
        ("←→", "scrub"),
        ("[ ]", "speed"),
        ("Esc", "quit"),
    )

    def _draw_keybar(self) -> None:
        """Always-on key list: one horizontal row along the very bottom edge."""
        wpx, hpx = self._size
        y = hpx - _KEYBAR_H
        pygame.draw.aaline(self._screen, palette.DIM, (0, y), (wpx, y))
        sep = self._small.render(" · ", True, palette.dim(palette.DIM, 0.9))
        x = 12
        for i, (k, what) in enumerate(self._KEYS):
            key_img = self._small.render(k, True, palette.MUTED)
            what_img = self._small.render(" " + what, True, palette.DIM)
            need = (sep.get_width() if i else 0) + key_img.get_width() + what_img.get_width()
            if x + need > wpx - 8:
                break
            if i:
                self._screen.blit(sep, (x, y + 4))
                x += sep.get_width()
            self._screen.blit(key_img, (x, y + 4))
            x += key_img.get_width()
            self._screen.blit(what_img, (x, y + 4))
            x += what_img.get_width()

    def _draw_waiting(self) -> None:
        """Pre-first-frame splash: the window is alive, the sim has not stepped yet
        (a training run sits in its first XLA compile for minutes). Keeps the event
        pump running so Esc works before any frame arrives."""
        self._events()
        if not self._open:
            return
        self._screen.fill(palette.BG)
        title = self._font.render(self.title, True, palette.MUTED)
        self._screen.blit(title, (12, 8))
        pygame.draw.aaline(self._screen, palette.DIM, (0, _TOP_H), (self._size[0], _TOP_H))
        mid_y = self._size[1] // 2
        for dy, (txt, col) in enumerate((
            ("WAITING FOR FIRST FRAME…", palette.BRIGHT),
            ("no sim steps yet — a training run compiles first, this can take minutes",
             palette.MUTED),
        )):
            img = (self._font if dy == 0 else self._small).render(txt, True, col)
            self._screen.blit(img, ((self._size[0] - img.get_width()) // 2, mid_y + dy * 26))
        self._draw_keybar()
        pygame.display.flip()
        req, self._shot_req = self._shot_req, None
        if req is not None:  # screenshot() works on the splash too
            pygame.image.save(self._screen, req[0])
            req[1].set()

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
        health = f" · {self._fps(self._drawn_t):.0f}/{self._fps(self._fed_t):.0f} fps"
        if self.snap_drops:
            health += f" · drops {self.snap_drops}"
        right = (
            f"{'PAUSED' if self.paused else f'{self.speed:g}x'} · t {vf.t:7.2f} s"
            f" · step {vf.step}{health}"
        )
        screen.blit(self._font.render(left, True, palette.MUTED), (12, 8))
        r_img = self._font.render(right, True, palette.MUTED)
        screen.blit(r_img, (wpx - r_img.get_width() - 12, 8))
        pygame.draw.aaline(screen, palette.DIM, (0, _TOP_H), (wpx, _TOP_H))

        # panes
        scene_rect = self._scene_rect
        proj = self._proj(scene_rect)
        if self.follow:  # keep the focused world centred; orbit/zoom still compose
            px, py = proj.point(vf.pos[vf.focus])
            proj.pan(scene_rect[0] + scene_rect[2] / 2.0 - px,
                     scene_rect[1] + scene_rect[3] / 2.0 - py)
        label = f"SCENE · {self._proj_kind.upper()}"
        if proj.orbited:
            label += f" · ORBIT {proj.azim:+.0f}°/{proj.elev:+.0f}°"
        draw_scene(
            screen,
            scene_rect,
            proj,
            self.scene,
            vf,
            trails={k: list(v) for k, v in self._trails.items()},
            omega_max=self.omega_max,
            label=label,
            font=self._small,
            show_glyphs=self.show_glyphs,
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

        self._draw_keybar()

        hud_rect = (0, hpx - _HUD_H - _KEYBAR_H, wpx, _HUD_H)
        armed = None
        if vf.info and "armed" in vf.info:
            armed = bool(np.asarray(vf.info["armed"]).reshape(-1)[vf.focus])
        draw_hud(
            screen,
            hud_rect,
            vf,
            control=self.control,
            omega_max=self.omega_max,
            histories={k: list(v.vals) for k, v in self._hists.items()},
            armed=armed,
            ranges=self._gauges,
            episodes=self._eps.vals or None,
            font=self._font,
            small=self._small,
        )
