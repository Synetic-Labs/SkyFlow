"""
The live pygame host (DESIGN.md §13).

`Viewer` owns the window, pacing, keys and pane layout — and nothing else: pixels come
from the shared builders (scenepane/hud/fpv), data arrives as ViewFrames. It never steps
the env and never owns the loop; `frame(state, ...)` is non-blocking and drops to the
display rate, so a sim running faster than the screen costs one small host pull per
DISPLAYED frame, not per step. The same push path serves live flight, replay and export.

Keys: Space pause · ←/→ step/scrub (replay) · [ ] speed · Tab focus world · V pilot-cam
res · T iso/top/profile · G fleet scatter · S screenshot · R reset request · Esc quit.
"""

import os
import time
from collections import deque
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
        """
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        pygame.font.init()
        try:
            self._screen = pygame.display.set_mode(size)
        except pygame.error:  # no video device: fall back to the dummy driver
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.quit()
            pygame.display.init()
            self._screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self._font = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 14)
        self._small = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 11)

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
        self._last_fleet: tuple[float, np.ndarray | None] = (0.0, None)
        self._last_vf: ViewFrame | None = None
        self._trails: dict[int, deque] = {i: deque(maxlen=_TRAIL_LEN) for i in range(len(self.watch))}
        self._hists: dict[str, deque] = {}  # one trace per channel, focused world

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
            hook = getattr(task, "viz_scene", None)
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
        """Show a live SimState. Non-blocking: skips work entirely between display frames."""
        if not self._open:
            return
        now = time.perf_counter()
        if now - self._last_draw < self._display_dt:
            return
        want_fleet = False
        if self.show_fleet:
            t_fleet, _ = self._last_fleet
            want_fleet = now - t_fleet > 0.5
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
            task_state=None if self._task_state_of is None else self._task_state_of(state),
            focus=self._focus,
            fleet_positions=want_fleet,
        )
        if want_fleet:
            self._last_fleet = (now, vf.positions)
        elif self.show_fleet:
            vf.positions = self._last_fleet[1]
        self.push(vf)

    def idle(self) -> None:
        """Keep the window live while the hosting loop is paused."""
        if self._last_vf is not None:
            self.push(self._last_vf, force=False)
        time.sleep(0.01)

    def push(self, vf: ViewFrame, force: bool = False) -> None:
        """Draw a prebuilt ViewFrame (replay/export enter here). Throttled unless forced."""
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
        if self._frames_left is not None:
            self._frames_left -= 1
            if self._frames_left <= 0:
                if self._shot:
                    pygame.image.save(self._screen, self._shot)
                self.close()

    def grab(self) -> np.ndarray:
        """The current window as [H,W,3] uint8 (export hosts read this after push)."""
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
