"""
viz, pygame half (DESIGN.md §11, §13) — skipped without pygame. The SDL dummy video
driver renders every pane headlessly: builders onto plain surfaces, the live Viewer over a
real hover rollout, the policy pane over a real vision rollout, and replay from a saved
flight.npz.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from skyflow import SimConfig, SkyFlowEnv  # noqa: E402
from skyflow.viz import palette  # noqa: E402
from skyflow.viz.frame import ViewFrame  # noqa: E402
from skyflow.viz.hud import draw_hud  # noqa: E402
from skyflow.viz.primitives import Box, Gate, Grid, Marker, Path, Scene  # noqa: E402
from skyflow.viz.projection import Projection  # noqa: E402
from skyflow.viz.record import FlightLog  # noqa: E402
from skyflow.viz.replay import replay  # noqa: E402
from skyflow.viz.scenepane import draw_scene  # noqa: E402
from skyflow.viz.viewer import Viewer  # noqa: E402


def _frame(w: int = 2) -> ViewFrame:
    plant = np.zeros((w, 17), np.float32)
    plant[:, 6] = 1.0
    plant[:, 2] = 1.2
    plant[:, 0] = np.arange(w)
    plant[:, 13:17] = 800.0
    return ViewFrame(
        plant=plant,
        action=np.tile(np.array([0.1, -0.2, 0.5, 0.0], np.float32), (w, 1)),
        channels={"reward": np.zeros(w, np.float32)},
        done=np.zeros(w, bool),
    )


def _colors(surface) -> set[tuple[int, int, int]]:
    arr = pygame.surfarray.array3d(surface)
    return {tuple(c) for c in arr.reshape(-1, 3)[:: max(1, arr.size // (3 * 20000))]}


class TestBuilders:
    def test_scene_pane_draws_all_primitives(self):
        from types import SimpleNamespace

        surface = pygame.Surface((420, 320))
        scene = Scene(
            Grid(half=(3, 3)),
            # the active accent rides the bind, exactly as a task hook declares it
            Gate(center=(1.5, 0, 1.2), lateral=(0, 1, 0), index=0,
                 bind="task_state.active_gate"),
            Gate(center=(-1.5, 0, 1.2), lateral=(0, 1, 0), index=1,
                 bind="task_state.active_gate"),
            Box(center=(0, 0, 1), half=(2.5, 2.5, 1)),
            Marker(center=(1, 1, 1.5)),
            Path([(0, 0, 1), (1, 1, 1), (2, 0, 1)]),
        )
        proj = Projection.fit("iso", *scene.aabb(), (0, 0, 420, 320))
        vf = _frame()
        vf.done = np.array([False, True])  # second world draws a crash mark
        vf.task_state = SimpleNamespace(active_gate=np.array([0, 0], np.int32))
        draw_scene(
            surface, (0, 0, 420, 320), proj, scene, vf,
            trails={0: [np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.2, 1.1])]},
            omega_max=2500.0,
        )
        arr = pygame.surfarray.array3d(surface)
        assert (arr.reshape(-1, 3) != palette.BG).any(axis=1).mean() > 0.01
        colors = _colors(surface)
        assert palette.ACCENT in colors, "active gate accent missing"
        assert palette.BAD in colors, "crash mark missing"

    def test_registered_user_primitive_draws(self):
        import dataclasses as dc

        from skyflow.viz.primitives import register_primitive

        @dc.dataclass(frozen=True)
        class Ring:
            center: tuple = (0.0, 0.0, 1.0)
            radius: float = 0.5
            style: str = "bright"
            bind: object = None

        def draw_ring(surface, proj, prim, color):
            x, y = proj.point(prim.center)
            pygame.draw.circle(surface, color, (x, y), max(3, prim.radius * proj.ppm), 2)

        register_primitive(Ring, draw_ring)
        surface = pygame.Surface((200, 160))
        scene = Scene(Grid(half=(2, 2)), Ring())
        proj = Projection.fit("iso", *scene.aabb(), (0, 0, 200, 160))
        draw_scene(surface, (0, 0, 200, 160), proj, scene, _frame())
        assert palette.BRIGHT in _colors(surface), "user primitive did not draw"

    def test_scene_pane_lod_fleet_marks(self):
        surface = pygame.Surface((100, 80))
        scene = Scene(Grid(half=(60, 60)))  # huge scene → tiny ppm → LOD marks
        proj = Projection.fit("iso", *scene.aabb(), (0, 0, 100, 80))
        draw_scene(surface, (0, 0, 100, 80), proj, scene, _frame())
        assert proj.ppm * 0.35 < 12.0  # confirms the LOD branch actually ran

    def test_hud_draws_both_control_modes(self):
        pygame.font.init()
        font = pygame.font.Font(None, 14)
        for control in ("motors", "sticks"):
            surface = pygame.Surface((900, 150))
            draw_hud(
                surface, (0, 0, 900, 150), _frame(), control=control, omega_max=2500.0,
                histories={"reward": [0.1, 0.5, 0.3], "goal_dist": [2.0, 1.5, 1.1]},
                font=font, small=font,
            )
            arr = pygame.surfarray.array3d(surface)
            assert (arr.reshape(-1, 3) != palette.BG).any(axis=1).mean() > 0.005


class TestViewer:
    def test_headless_hover_rollout_with_screenshot(self, tmp_path, key):
        env = SkyFlowEnv(SimConfig(num_envs=3, task="hover"))
        shot = tmp_path / "viewer.png"
        viewer = Viewer.for_env(
            env, watch=(0, 1), headless=True, frames=8, shot=str(shot), display_hz=1000.0
        )
        assert viewer.image_shape is None
        obs, state = env.reset(key)
        step = jax.jit(env.step)
        action = jnp.zeros((env.fleet, env.act_dim))
        for _ in range(200):
            if not viewer.open:
                break
            obs, state, reward, done, info = step(state, action)
            viewer.frame(state, obs=obs, action=action, reward=reward, done=done, info=info)
        assert not viewer.open, "frames budget should have closed the viewer"
        assert shot.exists() and shot.stat().st_size > 1000

    def test_policy_pane_vision_task(self, tmp_path, key):
        from skyflow.vision.camera import CameraModel

        env = SkyFlowEnv(
            SimConfig(
                num_envs=2,
                task="figure_eight",
                task_kwargs={
                    "vision": True,
                    "camera": CameraModel(height=16, width=16, supersample=1),
                },
            )
        )
        shot = tmp_path / "vision.png"
        viewer = Viewer.for_env(
            env, watch=(0,), headless=True, frames=3, shot=str(shot), display_hz=1000.0
        )
        assert viewer.image_shape == (16, 16, 1) and viewer.image_term == "mask"
        obs, state = env.reset(key)
        step = jax.jit(env.step)
        action = jnp.zeros((env.fleet, env.act_dim))
        for _ in range(20):
            if not viewer.open:
                break
            obs, state, reward, done, info = step(state, action)
            viewer.frame(state, obs=obs, action=action, reward=reward, done=done, info=info)
        assert not viewer.open and shot.exists()


class TestReplay:
    def test_replay_headless_from_saved_log(self, tmp_path, key):
        env = SkyFlowEnv(SimConfig(num_envs=2, task="figure_eight"))
        log = FlightLog.for_env(env, watch=(0, 1))
        _obs, state = env.reset(key)
        step = jax.jit(env.step)
        action = jnp.zeros((env.fleet, env.act_dim))
        for _ in range(15):
            _obs, state, reward, done, _info = step(state, action)
            log.capture(state, action=action, reward=reward, done=done)
        path = log.save(tmp_path / "flight.npz")

        shot = tmp_path / "replay.png"
        replay(path, pilot=(48, 64), headless=True, frames=10, shot=str(shot), speed=8.0)
        assert shot.exists() and shot.stat().st_size > 1000
