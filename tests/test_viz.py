"""
viz, pygame-free half (DESIGN.md §11, §13): primitives + serde, projection, the FPV
composite, FlightLog round-trips, and the core-must-not-import-viz boundary scan.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import skyflow
from skyflow.viz import palette
from skyflow.viz.fpv import compose, obs_mask, upscale
from skyflow.viz.frame import ViewFrame, quat_to_rot
from skyflow.viz.primitives import Box, Gate, Grid, Marker, Scene, register_primitive, resolve
from skyflow.viz.primitives import Path as VizPath
from skyflow.viz.projection import KINDS, Projection
from skyflow.viz.record import FlightLog, gateset_from_dict, gateset_to_dict


def _frame(w: int = 3, focus: int = 0) -> ViewFrame:
    plant = np.zeros((w, 17), np.float32)
    plant[:, 6] = 1.0  # identity quaternion
    plant[:, 2] = 1.0
    return ViewFrame(plant=plant, focus=focus)


class TestPrimitives:
    def test_serde_round_trip(self):
        scene = Scene(
            Grid(half=(4, 2)),
            VizPath([(0, 0, 1), (1, 1, 1)], closed=True, dashed=False),
            Gate(center=(1, 0, 1.5), lateral=(0, 1, 0), half_w=0.4, half_h=0.3, index=2),
            Box(center=(0, 0, 1), half=(2, 2, 1), style="dim"),
            Marker(center=(1, 2, 3), bind="task_state.goal"),
        )
        back = Scene.from_json(scene.to_json())
        assert list(back) == list(scene)

    def test_callable_bind_is_live_only(self):
        scene = Scene(Marker(bind=lambda vf: (1.0, 2.0, 3.0)))
        d = scene.to_dicts()[0]
        assert "bind" not in d  # callables drop from serde; string binds survive
        assert Scene.from_dicts([d])  # and the dict still reconstructs

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown primitive"):
            Scene.from_dicts([{"type": "teapot"}])

    def test_bind_string_takes_focus_row(self):
        vf = _frame(w=3, focus=1)
        vf.task_state = SimpleNamespace(goal=np.array([[0, 0, 0], [4, 5, 6], [9, 9, 9]], np.float32))
        m = resolve(Marker(bind="task_state.goal"), vf)
        assert m is not None and m.center == (4.0, 5.0, 6.0)

    def test_bind_none_hides_and_dict_overrides(self):
        vf = _frame()
        assert resolve(Marker(bind=lambda f: None), vf) is None
        g = resolve(Gate(center=(0, 0, 1), lateral=(0, 1, 0), bind=lambda f: {"style": "accent"}), vf)
        assert g is not None and g.style == "accent"

    def test_gate_bind_reads_active_index(self):
        vf = _frame(w=2, focus=1)
        vf.task_state = SimpleNamespace(active_gate=np.array([0, 3], np.int32))
        gates = [
            Gate(center=(g, 0, 1), lateral=(0, 1, 0), index=g, bind="task_state.active_gate")
            for g in range(4)
        ]
        styles = []
        for g in gates:
            r = resolve(g, vf)
            assert r is not None
            styles.append(r.style)
        assert styles == ["wire", "wire", "wire", "accent"]  # focus world's row wins

    def test_register_primitive_is_public_and_serde_round_trips(self):
        import dataclasses as dc

        @dc.dataclass(frozen=True)
        class Disc:
            center: tuple = (0.0, 0.0, 0.0)
            radius: float = 0.5
            style: str = "wire"
            bind: object = None

        register_primitive(Disc, lambda surface, proj, prim, color: None)
        back = Scene.from_dicts(Scene(Disc(center=(1, 2, 3))).to_dicts())
        assert isinstance(next(iter(back)), Disc)
        with pytest.raises(TypeError, match="dataclass"):
            register_primitive(int, lambda *a: None)

    def test_from_gateset_matches_world_readback(self):
        from skyflow.vision.gates import figure_eight

        gs = figure_eight(2)
        prims = Gate.from_gateset(gs)
        assert len(prims) == len(gs)
        np.testing.assert_allclose(
            np.array([p.center for p in prims]), np.asarray(gs.centers_world), atol=1e-6
        )
        assert [p.index for p in prims] == list(range(len(gs)))

    def test_aabb_covers_geometry(self):
        scene = Scene(Grid(half=(3, 2)), Box(center=(5, 0, 1), half=(1, 1, 1)))
        lo, hi = scene.aabb()
        assert lo[0] <= -3 and hi[0] >= 6 and hi[2] >= 2


class TestProjection:
    def test_altitude_draws_up_everywhere(self):
        for kind in KINDS:
            if kind == "top":
                continue
            proj = Projection(kind, 10.0, (0, 0))
            assert proj.point((0, 0, 2))[1] < proj.point((0, 0, 0))[1]

    def test_preset_bases(self):
        top = Projection("top", 10.0, (0, 0))
        np.testing.assert_allclose(top.points(np.array([[1.0, 0, 0]]))[0], [10, 0], atol=1e-9)
        np.testing.assert_allclose(top.points(np.array([[0, 1.0, 0]]))[0], [0, -10], atol=1e-9)
        prof = Projection("profile", 10.0, (0, 0))
        np.testing.assert_allclose(prof.points(np.array([[0, 0, 1.0]]))[0], [0, -10], atol=1e-9)

    def test_orbit_keeps_pivot_anchored(self):
        proj = Projection("iso", 10.0, (50, 50))
        pivot = (1.0, 2.0, 0.5)
        before = proj.points(np.asarray(pivot).reshape(1, 3))[0].copy()
        proj.orbit(37.0, -12.0, pivot=pivot)
        after = proj.points(np.asarray(pivot).reshape(1, 3))[0]
        np.testing.assert_allclose(after, before, atol=1e-9)
        assert proj.orbited

    def test_orbit_near_side_follows_a_rightward_drag(self):
        # the viewer sends orbit(-dx, ...) for a rightward drag: the ground point on
        # the camera's NEAR side must move right with the cursor (turntable feel)
        proj = Projection("iso", 10.0, (0, 0))
        az = np.radians(proj.azim)
        near = np.array([[np.cos(az), np.sin(az), 0.0]])
        u0 = proj.points(near)[0][0]
        proj.orbit(-4.0, 0.0)  # what a rightward drag sends
        assert proj.points(near)[0][0] > u0

    def test_orbit_clamps_elevation(self):
        proj = Projection("profile", 10.0, (0, 0))
        proj.orbit(0.0, 500.0)
        assert proj.elev == 89.0
        proj.orbit(0.0, -500.0)
        assert proj.elev == -89.0

    def test_pan_shifts_pixels(self):
        proj = Projection("top", 10.0, (50, 50))
        before = proj.point((1, 2, 0))
        proj.pan(7, -3)
        after = proj.point((1, 2, 0))
        assert after == (before[0] + 7, before[1] - 3)

    def test_zoom_at_keeps_anchor_fixed(self):
        proj = Projection("iso", 10.0, (50, 50))
        ax, ay = proj.point((1.0, -2.0, 0.5))
        proj.zoom_at(ax, ay, 1.5)
        bx, by = proj.point((1.0, -2.0, 0.5))
        assert bx == pytest.approx(ax) and by == pytest.approx(ay)
        assert proj.ppm == pytest.approx(15.0)

    def test_zoom_at_clamps_scale(self):
        proj = Projection("top", 10.0, (0, 0))
        proj.zoom_at(0, 0, 1e9)
        assert proj.scale <= 5000.0
        proj.zoom_at(0, 0, 1e-9)
        assert proj.scale >= 0.05

    def test_fit_keeps_aabb_inside_rect(self):
        lo, hi = np.array([-5.0, -2.0, 0.0]), np.array([5.0, 2.0, 3.0])
        rect = (10, 20, 400, 300)
        for kind in KINDS:
            proj = Projection.fit(kind, lo, hi, rect)
            corners = np.array(
                [[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]
            )
            uv = proj.points(corners)
            assert uv[:, 0].min() >= rect[0] and uv[:, 0].max() <= rect[0] + rect[2]
            assert uv[:, 1].min() >= rect[1] and uv[:, 1].max() <= rect[1] + rect[3]


class TestFpv:
    def test_compose_classes(self):
        mask = np.zeros((4, 4), np.float32)
        floor = np.zeros((4, 4), np.float32)
        mask[0, 0] = 1.0
        floor[3, :] = 1.0
        img = compose(mask, floor)
        assert img.dtype == np.uint8 and img.shape == (4, 4, 3)
        assert tuple(img[0, 0]) == palette.FPV_GATE
        assert tuple(img[3, 1]) == palette.FPV_FLOOR
        assert tuple(img[1, 1]) == palette.FPV_SKY

    def test_upscale_nearest(self):
        img = compose(np.eye(2, dtype=np.float32))
        big = upscale(img, 3)
        assert big.shape == (6, 6, 3)
        assert (big[0:3, 0:3] == img[0, 0]).all()

    def test_obs_mask_slices_verbatim(self):
        h = w = 4
        layout = {"mask": slice(0, h * w), "prop": slice(h * w, h * w + 3)}
        obs_row = np.arange(h * w + 3, dtype=np.float32)
        m = obs_mask(obs_row, (h, w, 1), layout)
        np.testing.assert_array_equal(m, obs_row[: h * w].reshape(h, w))

    def test_obs_mask_term_is_a_parameter(self):
        h = w = 2
        layout = {"depth": slice(0, h * w)}
        obs_row = np.arange(h * w, dtype=np.float32)
        m = obs_mask(obs_row, (h, w, 1), layout, term="depth")
        assert m.shape == (h, w)
        with pytest.raises(KeyError, match="depth"):
            obs_mask(obs_row, (h, w, 1), {"mask": slice(0, 4)}, term="depth")

    def test_quat_to_rot_identity_and_yaw(self):
        np.testing.assert_allclose(quat_to_rot(np.array([1, 0, 0, 0])), np.eye(3), atol=1e-9)
        yaw90 = quat_to_rot(np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
        np.testing.assert_allclose(yaw90 @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-7)


class TestFlightLog:
    def _synthetic(self, t: int = 12, w: int = 2) -> FlightLog:
        rng = np.random.default_rng(0)
        log = FlightLog(watch=(0, 1), every=1, header={"dt": 0.01, "control": "motors"})
        plant = rng.normal(size=(t, w, 17)).astype(np.float32)
        log.extend(
            plant,
            action=rng.uniform(-1, 1, (t, w, 4)).astype(np.float32),
            reward=rng.normal(size=(t, w)).astype(np.float32),
            channels={"goal_dist": rng.uniform(0, 3, (t, w)).astype(np.float32)},
            done=np.zeros((t, w), bool),
            binds={"task_state.active_gate": np.zeros((t, w), np.int32)},
        )
        return log

    def test_round_trip(self, tmp_path):
        log = self._synthetic()
        path = log.save(tmp_path / "flight.npz")
        back = FlightLog.load(path)
        assert len(back) == 12 and back.plant.shape == (12, 2, 17)
        assert back.header["control"] == "motors" and back.dt == pytest.approx(0.01)
        assert back.action is not None and back.action.shape == (12, 2, 4)
        assert set(back.channels) == {"reward", "goal_dist"}  # reward is just a channel
        assert back.channels["goal_dist"].shape == (12, 2)
        assert back.binds["task_state.active_gate"].shape == (12, 2)

    def test_extend_slices_fleet_chunks(self):
        log = FlightLog(watch=(0, 2), header={"dt": 0.01})
        plant = np.arange(1 * 4 * 17, dtype=np.float32).reshape(1, 4, 17)
        log.extend(plant)
        assert len(log) == 1
        np.testing.assert_array_equal(log._plant[0], plant[0, [0, 2]])

    def test_capture_respects_every_and_watch(self):
        state = SimpleNamespace(
            plant=np.arange(3 * 17, dtype=np.float32).reshape(3, 17),
            task_state=SimpleNamespace(active_gate=np.array([4, 5, 6], np.int32)),
        )
        # the scene's string bind tells capture which task values to record — no
        # field names live in the recorder itself
        header = {"dt": 0.01, "scene": [{"type": "gate", "bind": "task_state.active_gate"}]}
        log = FlightLog(watch=(1,), every=2, header=header)
        for _ in range(5):
            log.capture(state, reward=np.zeros(3, np.float32))
        assert len(log) == 3  # steps 0, 2, 4
        np.testing.assert_array_equal(log._plant[0], state.plant[[1]])
        assert int(log._binds["task_state.active_gate"][0][0]) == 5
        assert len(log._channels["reward"]) == 3

    def test_for_env_header_and_gateset_round_trip(self, key):
        from skyflow import SimConfig, SkyFlowEnv

        env = SkyFlowEnv(SimConfig(num_envs=2, task="figure_eight"))
        log = FlightLog.for_env(env, watch=(0,))
        assert log.header["task"] == "figure_eight" and log.header["dt"] == pytest.approx(0.01)
        assert log.header["scene"], "the task hook should populate the scene"
        from skyflow.tasks.gate_course import GateCourseTask

        task = env.task
        assert isinstance(task, GateCourseTask)  # narrows the Task protocol for .gates
        gs = gateset_from_dict(log.header["gateset"])
        np.testing.assert_allclose(
            np.asarray(gs.centers_world),
            np.asarray(task.gates.centers_world),
            atol=1e-6,
        )

    def test_capture_unwraps_wrapped_task_state(self):
        """Sticks mode wraps the task pytree in the firmware carry; the env accessor
        passed as task_state_of must make binds resolve anyway (DESIGN.md §10, §13)."""
        carry = SimpleNamespace(task=SimpleNamespace(active_gate=np.array([7], np.int32)))
        state = SimpleNamespace(plant=np.zeros((1, 17), np.float32), task_state=carry)
        log = FlightLog(
            watch=(0,),
            header={"scene": [{"type": "gate", "bind": "task_state.active_gate"}]},
            task_state_of=lambda s: s.task_state.task,
        )
        log.capture(state)
        assert int(log._binds["task_state.active_gate"][0][0]) == 7

    def test_env_task_state_accessor(self, key):
        from skyflow import SimConfig, SkyFlowEnv

        env = SkyFlowEnv(SimConfig(num_envs=2, task="figure_eight"))
        _obs, state = env.reset(key)
        assert hasattr(env.task_state(state), "active_gate")  # identity in motors mode

    def test_gateset_dict_round_trip(self):
        from skyflow.vision.gates import figure_eight

        gs = figure_eight(3)
        back = gateset_from_dict(gateset_to_dict(gs))
        for name in ("centers", "yaws", "inner_half", "outer_half", "normals"):
            np.testing.assert_allclose(
                np.asarray(getattr(back, name)), np.asarray(getattr(gs, name)), atol=1e-6
            )


class TestBoundary:
    def test_core_never_imports_viz(self):
        """DESIGN.md §13: no core module imports skyflow.viz (the viewer is optional)."""
        root = Path(skyflow.__file__).parent
        pattern = re.compile(r"^\s*(from|import)\s+skyflow\.viz", re.MULTILINE)
        offenders = [
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if "viz" not in p.parts and pattern.search(p.read_text())
        ]
        assert not offenders, f"core modules importing skyflow.viz: {offenders}"

    def test_task_hooks_return_serde_form(self):
        from skyflow.tasks.gate_course import GateCourseTask
        from skyflow.tasks.hover import HoverTask

        for task in (HoverTask(), GateCourseTask()):
            scene = Scene.from_dicts(task.viz_scene())
            assert len(scene) >= 2
        gates = [p for p in Scene.from_dicts(GateCourseTask().viz_scene()) if isinstance(p, Gate)]
        assert [g.index for g in gates] == list(range(len(gates)))
