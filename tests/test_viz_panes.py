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

    def test_show_glyphs_promotes_unfocused_marks(self):
        # default: unfocused watched worlds draw fleet marks at ANY zoom (here the ppm
        # is large); show_glyphs=True (the viewer's X key) draws them as full glyphs
        scene = Scene(Grid(half=(2, 2)))
        proj = Projection.fit("iso", *scene.aabb(), (0, 0, 400, 300))
        painted = []
        for show in (False, True):
            surface = pygame.Surface((400, 300))
            draw_scene(surface, (0, 0, 400, 300), proj, scene, _frame(),
                       omega_max=2500.0, show_glyphs=show)
            arr = pygame.surfarray.array3d(surface).reshape(-1, 3)
            painted.append(int((arr != palette.BG).any(axis=1).sum()))
        assert painted[1] > painted[0], "X should add pixels: marks became full glyphs"

    def test_focused_glyph_never_collapses(self):
        # tiny ppm: unfocused worlds are marks, the focused world keeps the full
        # glyph — its accent heading line is the fingerprint (marks draw body color)
        surface = pygame.Surface((100, 80))
        scene = Scene(Grid(half=(60, 60)))
        proj = Projection.fit("iso", *scene.aabb(), (0, 0, 100, 80))
        assert proj.ppm * 0.35 < 12.0
        draw_scene(surface, (0, 0, 100, 80), proj, scene, _frame(), omega_max=2500.0)
        arr = pygame.surfarray.array3d(surface).reshape(-1, 3)
        assert (arr == palette.ACCENT).all(axis=1).any(), "focused glyph lost its accent"

    def test_fleet_mark_flag_breaks_roll_symmetry(self):
        # pre-flag, the mark was a dot + body-x tick: a roll about body x moved nothing
        renders = []
        for half_roll in (0.0, np.pi / 6):
            vf = _frame()
            vf.plant[1, 6] = np.cos(half_roll)
            vf.plant[1, 7] = np.sin(half_roll)  # roll about body x, world 1 (unfocused)
            surface = pygame.Surface((400, 300))
            scene = Scene(Grid(half=(4, 4)))
            proj = Projection.fit("iso", *scene.aabb(), (0, 0, 400, 300))
            assert proj.ppm * 0.35 < 12.0  # the mark branch, not the full glyph
            draw_scene(surface, (0, 0, 400, 300), proj, scene, vf)
            renders.append(pygame.surfarray.array3d(surface))
        assert (renders[0] != renders[1]).any(), "the up-flag should move with roll"

    def test_fleet_scatter_gated_by_show_fleet(self):
        # a pushed frame carries positions; the G key must be able to hide them
        scene = Scene(Grid(half=(6, 6)))
        proj = Projection.fit("iso", *scene.aabb(), (0, 0, 400, 300))
        painted = []
        for show in (True, False):
            vf = _frame()
            vf.positions = np.tile(np.array([[5.0, 5.0, 0.5]]), (50, 1))
            surface = pygame.Surface((400, 300))
            draw_scene(surface, (0, 0, 400, 300), proj, scene, vf, show_fleet=show)
            arr = pygame.surfarray.array3d(surface).reshape(-1, 3)
            painted.append(int((arr != palette.BG).any(axis=1).sum()))
        assert painted[0] > painted[1], "show_fleet=False should hide the scatter"

    def test_hud_compass_turns_with_yaw(self):
        # pure yaw changes no other instrument, so any pixel delta comes from the compass
        renders = []
        for half_yaw in (0.0, np.pi / 4):
            f = _frame()
            f.plant[:, 6] = np.cos(half_yaw)
            f.plant[:, 9] = np.sin(half_yaw)
            surface = pygame.Surface((900, 150))
            draw_hud(surface, (0, 0, 900, 150), f, omega_max=2500.0)
            renders.append(pygame.surfarray.array3d(surface))
        assert (renders[0] != renders[1]).any()

    def test_hud_level_markers_light_when_level(self):
        # each level marker turns GOOD only while its own axis is level
        def good_pixels(half_roll: float, half_pitch: float) -> int:
            f = _frame()
            f.plant[:, 6] = np.cos(half_roll) * np.cos(half_pitch)
            f.plant[:, 7] = np.sin(half_roll) * np.cos(half_pitch)
            f.plant[:, 8] = np.cos(half_roll) * np.sin(half_pitch)
            f.plant[:, 9] = -np.sin(half_roll) * np.sin(half_pitch)
            surface = pygame.Surface((900, 150))
            draw_hud(surface, (0, 0, 900, 150), f, omega_max=2500.0)
            arr = pygame.surfarray.array3d(surface).reshape(-1, 3)
            return int((arr == palette.GOOD).all(axis=1).sum())

        level, rolled, pitched = good_pixels(0, 0), good_pixels(0.2, 0), good_pixels(0, 0.2)
        both_off = good_pixels(0.2, 0.2)
        assert level > rolled > both_off == 0 and level > pitched > both_off
        assert rolled != pitched, "the roll and pitch markers are separate marks"

    def test_hud_episode_reward_and_throughput_panels(self):
        pygame.font.init()
        font = pygame.font.Font(None, 14)
        rect = (0, 0, 1600, 150)
        renders = []
        for ep_rewards, throughput, total in (
            (None, None, None),
            ([-3.0, 1.0, 5.0], None, None),  # negative totals hang below a zero line
            (None, [900.0, 1000.0, 1100.0], None),
            (None, None, [1000.0, 2000.0, 3000.0]),
        ):
            surface = pygame.Surface(rect[2:])
            draw_hud(surface, rect, _frame(), histories={"reward": [0.1, 0.3]},
                     ep_rewards=ep_rewards, throughput=throughput, total_steps=total,
                     total_now=None if total is None else 3500.0, fleet=4096,
                     font=font, small=font)
            renders.append(pygame.surfarray.array3d(surface))
        assert (renders[0] != renders[1]).any(), "the episode-reward panel should draw"
        assert (renders[0] != renders[2]).any(), "the throughput panel should draw"
        assert (renders[0] != renders[3]).any(), "the total-steps panel should draw"

    def test_hud_dials_and_armed_lamp(self):
        pygame.font.init()
        font = pygame.font.Font(None, 14)

        def render(speed: float, armed):
            f = _frame()
            f.plant[:, 3] = speed
            surface = pygame.Surface((900, 150))
            draw_hud(surface, (0, 0, 900, 150), f, control="sticks", omega_max=2500.0,
                     armed=armed, font=font, small=font)
            return pygame.surfarray.array3d(surface)

        slow, fast, off = render(1.0, True), render(3.0, True), render(1.0, False)
        assert (slow != fast).any(), "the speed dial should move"
        assert (slow != off).any(), "the arm lamp should change with the state"
        assert palette.GOOD in {tuple(c) for c in slow.reshape(-1, 3)}, "lamp not GOOD"

    def test_hud_climb_dial_is_signed(self):
        renders = []
        for vz in (-2.0, 0.0, 2.0):
            f = _frame()
            f.plant[:, 5] = vz  # vertical velocity only; the other instruments hold
            surface = pygame.Surface((900, 150))
            draw_hud(surface, (0, 0, 900, 150), f, omega_max=2500.0)
            renders.append(pygame.surfarray.array3d(surface))
        assert (renders[0] != renders[2]).any(), "climb and sink must draw differently"
        assert (renders[0] != renders[1]).any() and (renders[1] != renders[2]).any()

    def test_hud_episode_bars(self):
        base, bars = [], None
        for episodes in (None, [10.0, 20.0, 40.0, 80.0]):
            surface = pygame.Surface((900, 150))
            draw_hud(surface, (0, 0, 900, 150), _frame(), episodes=episodes)
            base.append(pygame.surfarray.array3d(surface))
        bars = base[1]
        assert (base[0] != bars).any(), "the episode panel should draw"

    def test_hud_gauge_ranges_grow_only(self):
        ranges: dict[str, float] = {}
        seen = []
        for speed in (1.0, 9.0, 2.0):
            f = _frame()
            f.plant[:, 3] = speed
            surface = pygame.Surface((900, 150))
            draw_hud(surface, (0, 0, 900, 150), f, ranges=ranges)
            seen.append(ranges["SPD m/s"])
        assert seen == [2.0, 10.0, 10.0]  # grows on 9, never shrinks back

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
        assert viewer.wait_closed(10.0), "frames budget should have closed the viewer"
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
        assert viewer.wait_closed(10.0) and shot.exists()

    def test_viewer_tracks_episode_lengths(self):
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid()), headless=True, threaded=False)
        try:
            def vf_at(step: int, done: bool = False) -> ViewFrame:
                f = _frame()
                f.step = step
                f.done = np.array([done, False])
                return f

            # GLOBAL step counter (training viz): the first episode was joined midway,
            # so only the second (steps 111..118) is charted
            for f in (vf_at(100), vf_at(105), vf_at(110, done=True),
                      vf_at(111), vf_at(118, done=True)):
                viewer._track(f)
            assert viewer._eps.vals == [8.0]

            # PER-EPISODE counter (eval): the drop 9 -> 1 reveals a missed done
            viewer._eps.clear()
            viewer.discontinuity()
            for f in (vf_at(3), vf_at(9), vf_at(1), vf_at(5), vf_at(8, done=True)):
                viewer._track(f)
            assert viewer._eps.vals == [8.0]

            # a host-declared discontinuity (new log, seek): the step jump back is no
            # episode end, and the episode in progress leaves the chart
            viewer._eps.clear()
            for f in (vf_at(1), vf_at(2), vf_at(3, done=True), vf_at(4), vf_at(5)):
                viewer._track(f)
            viewer.discontinuity()
            for f in (vf_at(0), vf_at(1), vf_at(2, done=True), vf_at(3), vf_at(4, done=True)):
                viewer._track(f)
            assert viewer._eps.vals == [3.0, 2.0]  # the cut-short episode is not charted

            # a repeated row (paused or held replay) counts once
            viewer._eps.clear()
            viewer.discontinuity()
            frames = [vf_at(10), vf_at(11), vf_at(12, done=True)]
            for f in (frames[0], frames[1], frames[1], frames[1], frames[2], frames[2]):
                viewer._track(f)
            for f in (vf_at(13), vf_at(14), vf_at(14), vf_at(15, done=True)):
                viewer._track(f)
            assert viewer._eps.vals == [3.0]
        finally:
            viewer.close()

    def test_viewer_tracks_episode_rewards(self):
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid()), headless=True, threaded=False)
        try:
            def vf_at(step: int, r: float, done: bool = False, info=None) -> ViewFrame:
                f = _frame()
                f.step = step
                f.done = np.array([done, False])
                f.channels = {"reward": np.array([r, 0.0], np.float32)}
                f.info = info
                return f

            # replay: every step drawn, no step info — the reward channel sums exactly
            for f in (vf_at(1, 1.0), vf_at(2, 2.0), vf_at(3, 3.0, done=True),
                      vf_at(4, -1.0), vf_at(5, -1.0, done=True)):
                viewer._track(f)
            assert viewer._ep_rets.vals == [-2.0]  # the joined-midway first is not charted
            assert viewer._eps.vals == [2.0]

            # live: the env's exact accumulators win over the sampled reward, even for
            # an episode joined midway; the done row carries the finished totals while
            # its step counter already reset
            viewer._ep_rets.clear()
            viewer._eps.clear()
            viewer.discontinuity()

            def info(n: int, ret: float, done: bool) -> dict:
                return {"ep_len": np.array([n, 0]), "ep_return": np.array([ret, 0.0]),
                        "terminated": np.array([done, False]),
                        "truncated": np.array([False, False])}

            for f in (vf_at(10, 0.5, info=info(10, 4.0, False)),
                      vf_at(30, 0.5, info=info(30, 9.0, False)),
                      vf_at(0, 0.5, done=True, info=info(31, 9.5, True)),
                      vf_at(7, 0.1, info=info(7, 0.7, False)),
                      # a done merged from a dropped frame whose totals were lost: this
                      # row is the next episode, the cut-short one is not charted
                      vf_at(2, 0.1, done=True, info=info(2, 0.2, False)),
                      vf_at(5, 0.1, info=info(5, 0.5, False)),
                      vf_at(0, 0.1, done=True, info=info(6, 0.6, True))):
                viewer._track(f)
            assert viewer._ep_rets.vals == [9.5, 0.6]
            assert viewer._eps.vals == [31.0, 6.0]
        finally:
            viewer.close()

    def test_viewer_throughput_and_fleet(self):
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid()), headless=True, threaded=False)
        try:
            t = [0.0]
            import skyflow.viz.viewer as vmod

            real = vmod.time.perf_counter
            vmod.time.perf_counter = lambda: t[0]
            try:
                for i in range(7):  # 100 steps every 0.25 s = 400 steps/s per sim
                    f = _frame()
                    f.step, f.fleet = 100 * i, 64
                    t[0] = 0.25 * i
                    viewer._track(f)
            finally:
                vmod.time.perf_counter = real
            # the first window (first-draw compiles) is dropped
            assert viewer._sps.vals == [pytest.approx(400.0), pytest.approx(400.0)]
            assert viewer._fleet == 64

            # measured mode totals the steps seen since the viewer opened
            assert viewer._steps.vals == [400.0, 600.0] and viewer._steps_now == 600.0

            viewer.report_throughput(1234.0, fleet=128, steps=9000.0)  # the host knows better
            assert not viewer.measure_sps and viewer._sps.vals[-1] == 1234.0
            assert viewer._fleet == 128
            assert viewer._steps.vals[-1] == 9000.0 and viewer._steps_now == 9000.0
        finally:
            viewer.close()

    def test_dropped_frame_keeps_exact_episode_totals(self):
        """Live viewers drop most frames, so most done rows are never drawn: their
        exact ep_len/ep_return must still reach the episode charts."""
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid()), headless=True, threaded=False)
        try:
            def vf_at(step: int, n: int, ret: float, done: bool) -> ViewFrame:
                f = _frame()
                f.step = step
                f.done = np.array([done, False])
                f.info = {"ep_len": np.array([n, 0]), "ep_return": np.array([ret, 0.0]),
                          "terminated": np.array([done, False]),
                          "truncated": np.array([False, False])}
                return f

            viewer._track(vf_at(40, 40, 3.0, False))
            viewer._stash_done(vf_at(0, 57, 4.5, True))  # the done row, never drawn
            drawn = vf_at(5, 5, 0.4, False)
            viewer._merge_lost_dones(drawn)
            viewer._track(drawn)
            assert viewer._eps.vals == [57.0] and viewer._ep_rets.vals == [4.5]
            assert viewer._ep_len == 5 and viewer._ep_ret == pytest.approx(0.4)
        finally:
            viewer.close()

    def test_later_dropped_done_waits_for_a_later_frame(self):
        """A done stashed AFTER a frame was fed belongs to a later frame: folding it
        into the earlier one charted the episode early, then a second time."""
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid()), headless=True, threaded=False)
        try:
            early = _frame()
            early.step = 10
            late_done = _frame()
            late_done.done = np.array([True, False])
            viewer._stash_done(late_done, seq=5)
            viewer._merge_lost_dones(early, seq=4)  # fed before the done: untouched
            assert early.done is not None and not early.done.any()
            assert len(viewer._lost_dones) == 1
            after = _frame()
            viewer._merge_lost_dones(after, seq=6)
            assert after.done is not None and after.done[0] and not viewer._lost_dones
        finally:
            viewer.close()

    def test_status_pill_and_watch_labels(self, tmp_path):
        from skyflow.viz.primitives import Grid, Scene

        with pytest.raises(ValueError, match="labels"):
            Viewer(Scene(Grid()), watch=(0, 4), labels=("seed 0",), headless=True,
                   threaded=False)
        viewer = Viewer(Scene(Grid()), watch=(0, 4), labels=("seed 0", "seed 1"),
                        headless=True, threaded=False)
        try:
            with pytest.raises(ValueError, match="level"):
                viewer.set_status("x", level="loud")
            shots = []
            for status in (None, ("TRAINING STOPPED", "bad")):
                viewer.set_status(*status) if status else viewer.set_status(None)
                viewer.push(_frame(), force=True)
                path = tmp_path / f"s{len(shots)}.png"
                assert viewer.screenshot(str(path))
                shots.append(pygame.surfarray.array3d(pygame.image.load(str(path))))
            arr = shots[1].reshape(-1, 3)
            assert (arr == palette.BAD).all(axis=1).any(), "the status pill should draw"
            assert not (shots[0].reshape(-1, 3) == palette.BAD).all(axis=1).any()
        finally:
            viewer.close()

    def test_ep_trace_compresses_pairwise(self):
        from skyflow.viz.viewer import _EpTrace

        tr = _EpTrace(cap=8)
        for i in range(32):
            tr.add(float(i))
        assert tr.bin == 8 and len(tr.vals) == 4
        assert tr.vals[0] == pytest.approx(3.5)  # mean of the first bin, 0..7

    def test_follow_centers_the_focused_world(self):
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid(half=(8, 8))), headless=True, threaded=False)
        try:
            viewer.follow = True
            f = _frame()
            f.plant[0, 0:3] = (6.0, -5.0, 2.0)
            viewer.push(f, force=True)
            rx, ry, rw, rh = viewer._scene_rect
            px, py = viewer._proj(viewer._scene_rect).point(f.plant[0, 0:3])
            assert abs(px - (rx + rw / 2)) < 1e-6
            assert abs(py - (ry + rh / 2)) < 1e-6
        finally:
            viewer.close()

    def test_mailbox_reports_replaced_frames(self):
        from skyflow.viz.viewer import _Mailbox

        dropped = []
        mb = _Mailbox(on_drop=dropped.append)
        mb.put("a")
        mb.put("b")  # "a" was never taken: it drops
        assert dropped == ["a"]
        assert mb.take(0.1)[0] == "b"
        mb.put("c")  # "b" WAS taken: no drop
        assert dropped == ["a"]

    def test_dropped_done_frame_still_clears_the_trail(self):
        from skyflow.viz.primitives import Grid, Scene

        viewer = Viewer(Scene(Grid()), headless=True, threaded=False)
        try:
            a = _frame()
            a.plant[0, 0:3] = (3.0, 0.0, 1.0)
            a.step = 5
            b = _frame()
            b.plant[0, 0:3] = (3.1, 0.0, 1.0)
            b.step = 6
            viewer.push(a, force=True)
            viewer.push(b, force=True)
            assert len(viewer._trails[0]) == 2
            crash = _frame()
            crash.step = 7
            crash.done = np.array([True, False])
            viewer._stash_done(crash)  # the crash frame itself was dropped whole
            respawn = _frame()  # back at the origin
            respawn.plant[0, 0:3] = (0.0, 0.0, 1.0)
            respawn.step = 0
            viewer.push(respawn, force=True)
            # the carried done must clear the trail: no crash→origin streak
            assert len(viewer._trails[0]) == 0
        finally:
            viewer.close()

    def test_render_thread_owns_drawing(self, tmp_path):
        """A threaded viewer draws, screenshots, and closes with NO further pushes:
        the render thread keeps the last frame alive and burns the frames budget."""
        from skyflow.viz.primitives import Grid, Scene

        shot = tmp_path / "threaded.png"
        viewer = Viewer(
            Scene(Grid()), headless=True, frames=5, shot=str(shot), display_hz=1000.0
        )
        assert viewer._thread is not None and viewer._thread.is_alive()
        viewer.push(_frame())
        assert viewer.wait_closed(10.0), "render thread should exhaust the budget alone"
        assert shot.exists() and shot.stat().st_size > 1000
        v2 = Viewer(Scene(Grid()), headless=True)
        try:
            with pytest.raises(RuntimeError, match="threaded=False"):
                v2.grab()
        finally:
            v2.close()


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

    def test_mp4_export_plays_realtime(self, tmp_path, key):
        """The mp4 is resampled onto the mp4_fps clock: a 100 Hz log exported at 30 fps
        keeps its wall-clock duration (rows drop), never its row count."""
        iio = pytest.importorskip("imageio.v3")
        env = SkyFlowEnv(SimConfig(num_envs=2, task="figure_eight"))
        log = FlightLog.for_env(env, watch=(0,))
        _obs, state = env.reset(key)
        step = jax.jit(env.step)
        action = jnp.zeros((env.fleet, env.act_dim))
        for _ in range(50):  # 0.5 s at the 100 Hz control default
            _obs, state, reward, done, _info = step(state, action)
            log.capture(state, action=action, reward=reward, done=done)
        path = log.save(tmp_path / "flight.npz")

        mp4 = tmp_path / "flight.mp4"
        replay(path, headless=True, mp4=str(mp4), mp4_fps=30.0)
        assert mp4.exists()
        vid = iio.imread(str(mp4))
        n_expected = round(50 * 0.01 * 30.0)  # duration x playback fps, not 50 rows
        assert abs(vid.shape[0] - n_expected) <= 1
        meta = iio.immeta(str(mp4))
        assert round(meta["fps"]) == 30
