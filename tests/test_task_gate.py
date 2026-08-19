"""
DESIGN.md §11 (task tests, gate part) — GateCourseTask pass/crash/progress logic on
dynamics-free synthetic plant states: plant rows [F,17] are built directly, so nothing
here depends on env.py or the dynamics backend.

The shipped `skyflow.vision` modules (DESIGN.md §2: gates/camera/renderer/mask_noise)
are used as-is.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.tasks.gate_course import GateCourseTask, GateTaskState
from skyflow.vision.gates import GateSet, figure_eight

# -- synthetic plant states ---------------------------------------------------------


def _plant(pos, vel=None) -> jax.Array:
    """[F,17] f32 spec-layout rows: given position/velocity, identity attitude, at rest."""
    pos = jnp.asarray(pos, jnp.float32).reshape(-1, 3)
    f = pos.shape[0]
    vel = jnp.zeros((f, 3)) if vel is None else jnp.asarray(vel, jnp.float32).reshape(f, 3)
    quat = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (f, 1))
    return jnp.concatenate(
        [pos, vel, quat, jnp.zeros((f, 3)), jnp.zeros((f, 4))], axis=-1
    ).astype(jnp.float32)


def _fresh_state(f: int) -> GateTaskState:
    return GateTaskState(
        active_gate=jnp.zeros((f,), jnp.int32), passes=jnp.zeros((f,), jnp.int32)
    )


def _single_gate() -> GateSet:
    """One gate at an arbitrary pose; tests read the built fields, never assume them."""
    return GateSet.build([[2.0, -1.0, 1.5]], [0.7])


def _axes(gates, g: int):
    """Gate frame in PUBLIC z-up coordinates (raw GateSet fields are internal NED)."""
    c = np.asarray(gates.centers_world[g], np.float64)
    n = np.asarray(gates.normals_world[g], np.float64)
    lat = np.asarray(gates.laterals_world[g], np.float64)
    vert = np.asarray(gates.verticals_world[g], np.float64)
    return c, n, lat, vert


def _run_line(task, points):
    """Evaluate a scripted [T,3] single-world trajectory; returns per-step TaskEvals."""
    ev_fn = jax.jit(task.evaluate)
    state = _fresh_state(1)
    out = []
    for i in range(1, points.shape[0]):
        ev = ev_fn(_plant(points[i - 1][None]), _plant(points[i][None]), state)
        state = ev.task_state
        out.append(ev)
    return out


# -- required suite: single-gate pass, frame hit, figure-eight tour ------------------


def test_single_gate_flythrough_passes_exactly_once():
    """Straight centred-ish transit: one pass, centering ∈ (0,1] at the known offset."""
    gates = _single_gate()
    task = GateCourseTask(gates)
    c, n, lat, vert = _axes(gates, 0)
    iw, ih = np.asarray(gates.inner_half[0], np.float64)

    # Offsets at 0.4·iw laterally, 0.2·ih vertically → Chebyshev 0.4 → centering 0.6.
    offset = 0.4 * iw * lat - 0.2 * ih * vert
    ss = np.arange(-1.45, 1.551, 0.1)  # straddles the plane, never lands exactly on it
    points = c + offset + ss[:, None] * n

    evs = _run_line(task, jnp.asarray(points, jnp.float32))
    passed = np.array([float(e.info["gate_passed"][0]) for e in evs])
    assert passed.sum() == 1.0, "exactly one pass along a single straight transit"

    step = int(np.nonzero(passed)[0][0])
    centering = float(evs[step].info["gate_centering"][0])
    assert 0.0 < centering <= 1.0
    np.testing.assert_allclose(centering, 0.6, atol=2e-3)

    assert not any(bool(e.crash[0]) for e in evs), "clean transit never crashes"
    # single gate ⇒ the pass is the last-gate pass ⇒ success on exactly that step
    success = [bool(e.success[0]) for e in evs]
    assert success[step] and sum(success) == 1
    # active gate clips at the last index
    assert int(evs[-1].task_state.active_gate[0]) == 0
    # reward contract: [F] float32
    assert evs[step].reward.shape == (1,) and evs[step].reward.dtype == jnp.float32
    # approaching the pre-gate point pays positive progress (ω = 0 ⇒ no rate penalty)
    assert float(evs[0].reward[0]) > 0.0


def test_offset_trajectory_through_frame_crashes():
    """A transit through the frame band (between inner and outer edges) is a crash."""
    gates = _single_gate()
    task = GateCourseTask(gates)
    c, n, lat, _vert = _axes(gates, 0)
    iw = float(gates.inner_half[0, 0])
    ow = float(gates.outer_half[0, 0])

    offset = 0.5 * (iw + ow) * lat  # mid-band: inside the frame, outside the opening
    ss = np.arange(-1.45, 1.551, 0.1)
    points = c + offset + ss[:, None] * n

    evs = _run_line(task, jnp.asarray(points, jnp.float32))
    assert any(bool(e.crash[0]) for e in evs), "frame transit must register a crash"
    assert sum(float(e.info["gate_passed"][0]) for e in evs) == 0.0
    assert not any(bool(e.success[0]) for e in evs)
    assert int(evs[-1].task_state.active_gate[0]) == 0


def test_wide_flyaround_of_active_gate_is_a_miss_crash():
    """Crossing the active plane beyond the outer edge touches no solid but is a miss.

    The miss event (DESIGN.md §9 "miss/frame-hit = task crash"): pass and miss
    partition every forward transit of the active centre plane, so flying around the
    gate ends the attempt exactly like hitting the frame does.
    """
    gates = _single_gate()
    task = GateCourseTask(gates)
    c, n, lat, _vert = _axes(gates, 0)
    ow = float(gates.outer_half[0, 0])

    offset = (ow + 0.5) * lat  # well clear of the physical frame
    ss = np.arange(-1.45, 1.551, 0.1)
    points = c + offset + ss[:, None] * n

    evs = _run_line(task, jnp.asarray(points, jnp.float32))
    assert any(bool(e.crash[0]) for e in evs), "wide fly-around must register a miss crash"
    assert sum(float(e.info["gate_missed"][0]) for e in evs) == 1.0
    assert sum(float(e.info["gate_passed"][0]) for e in evs) == 0.0
    assert not any(bool(e.success[0]) for e in evs)


def _figure_eight_course():
    """The shipped builder at the task's default course size (2·3 gates)."""
    return figure_eight(3)


def test_figure_eight_tour_progresses_active_gate_monotonically():
    """A scripted pre→post waypoint tour advances active_gate 0 → G-1 without regression.

    World 0 flies the tour; world 1 stays parked to prove per-world isolation of the
    course state. Asymmetric pre/post offsets keep samples off the exact gate planes.
    """
    gates = _figure_eight_course()
    g_count = len(gates)
    assert g_count >= 2, "figure-eight must be a multi-gate course"
    task = GateCourseTask(gates)

    waypoints = []
    for g in range(g_count):
        c, n, _lat, _vert = _axes(gates, g)
        waypoints.append(c - 0.63 * n)
        waypoints.append(c + 0.47 * n)
    path = [waypoints[0]]
    for wp in waypoints[1:]:
        start = path[-1]
        leg = wp - start
        steps = max(2, math.ceil(np.linalg.norm(leg) / 0.2) + 1)
        for t in np.linspace(0.0, 1.0, steps)[1:]:
            path.append(start + t * leg)
    path = np.asarray(path, np.float32)

    parked = np.asarray(waypoints[0] + np.array([0.0, 0.0, 1.0]), np.float32)
    ev_fn = jax.jit(task.evaluate)
    state = _fresh_state(2)
    actives, passes_total, success_seen = [0], 0.0, False
    for i in range(1, path.shape[0]):
        prev_rows = np.stack([path[i - 1], parked])
        cur_rows = np.stack([path[i], parked])
        ev = ev_fn(_plant(prev_rows), _plant(cur_rows), state)
        a0 = int(ev.task_state.active_gate[0])
        assert a0 >= actives[-1], "active gate must never regress"
        assert int(ev.task_state.active_gate[1]) == 0, "parked world must not advance"
        actives.append(a0)
        passes_total += float(ev.info["gate_passed"][0])
        assert float(ev.info["gate_passed"][1]) == 0.0
        success_seen = success_seen or bool(ev.success[0])
        assert not bool(ev.success[1]) and not bool(ev.crash[1])
        state = ev.task_state

    assert actives[-1] == g_count - 1, "tour must reach the last gate"
    assert passes_total == g_count, "every gate passed exactly once while active"
    assert success_seen, "passing the last gate is success"
    assert int(state.passes[0]) == g_count and int(state.passes[1]) == 0


# -- protocol surface: spawn, state obs, vision obs ----------------------------------


def test_spawn_shapes_and_start_side(nominal_params):
    gates = _single_gate()
    task = GateCourseTask(gates)
    n = nominal_params.shape[0]
    plant, state = task.spawn(jax.random.PRNGKey(3), n, nominal_params)

    assert plant.shape == (n, 17) and plant.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(plant)))
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(plant[:, 6:10], axis=-1)), 1.0, atol=1e-6
    )
    assert state.active_gate.shape == (n,) and state.active_gate.dtype == jnp.int32
    assert bool(jnp.all(state.active_gate == 0)) and bool(jnp.all(state.passes == 0))
    # podium spawn sits on the approach (-normal) side of gate 0
    c, nrm, _lat, _vert = _axes(gates, 0)
    signed = (np.asarray(plant[:, 0:3], np.float64) - c) @ nrm
    assert (signed < 0.0).all()

    spread = GateCourseTask(_figure_eight_course(), spawn_mode="spread")
    _, sstate = spread.spawn(jax.random.PRNGKey(4), 256, nominal_params[:1])
    drawn = np.asarray(sstate.active_gate)
    assert drawn.min() >= 0 and drawn.max() <= spread.num_gates - 1
    assert len(np.unique(drawn)) > 1, "spread mode must start worlds at multiple gates"


def test_state_obs_layout_and_frames():
    """Identity attitude ⇒ body frame == world frame: obs blocks are directly checkable."""
    gates = GateSet.build([[2.0, 0.0, 1.5], [4.0, 0.0, 1.5]], [0.0, 0.0])
    task = GateCourseTask(gates)
    assert task.image_shape is None
    assert task.obs_spec.dim == 25
    assert list(task.obs_spec.layout) == [
        "gate_rel", "gate_normal", "next_gate_rel", "vel_body", "rot_matrix", "last_action",
    ]

    pos = jnp.asarray([[0.5, 0.25, 1.0]], jnp.float32)
    vel = jnp.asarray([[1.0, -2.0, 0.5]], jnp.float32)
    last_action = jnp.asarray([[0.1, 0.2, 0.3, 0.4]], jnp.float32)
    obs, state = task.observe(
        _plant(pos, vel), _fresh_state(1), (jnp.zeros((1, 3)), jnp.zeros((1, 3))),
        last_action, jax.random.PRNGKey(0), fresh_spawn=False,
    )
    assert obs.shape == (1, 25) and obs.dtype == jnp.float32
    assert isinstance(state, GateTaskState)

    lay = task.obs_spec.layout
    o = np.asarray(obs[0], np.float64)
    np.testing.assert_allclose(
        o[lay["gate_rel"]], np.asarray(gates.centers_world[0]) - np.asarray(pos[0]), atol=1e-5
    )
    np.testing.assert_allclose(
        o[lay["gate_normal"]], np.asarray(gates.normals_world[0]), atol=1e-5
    )
    np.testing.assert_allclose(
        o[lay["next_gate_rel"]], np.asarray(gates.centers_world[1]) - np.asarray(pos[0]),
        atol=1e-5,
    )
    np.testing.assert_allclose(o[lay["vel_body"]], np.asarray(vel[0]), atol=1e-5)
    np.testing.assert_allclose(o[lay["rot_matrix"]], np.eye(3).ravel(), atol=1e-5)
    np.testing.assert_allclose(o[lay["last_action"]], np.asarray(last_action[0]), atol=1e-6)


def test_vision_obs_variant_shapes():
    """vision=True swaps the gate blocks for the flattened mask and sets image_shape."""
    task = GateCourseTask(_single_gate(), vision=True)
    assert task.image_shape is not None and task.image_shape[2] == 1
    h, w, _ = task.image_shape
    assert task.obs_spec.layout["mask"] == slice(0, h * w)
    assert task.obs_spec.dim == h * w + 16

    pos = jnp.asarray([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], jnp.float32)
    obs, _ = task.observe(
        _plant(pos), _fresh_state(2), (jnp.zeros((2, 3)), jnp.zeros((2, 3))),
        jnp.zeros((2, 4), jnp.float32), jax.random.PRNGKey(0), fresh_spawn=False,
    )
    assert obs.shape == (2, h * w + 16) and obs.dtype == jnp.float32
    mask = np.asarray(obs[:, : h * w])
    assert (mask >= 0.0).all() and (mask <= 1.0).all(), "coverage mask must lie in [0,1]"


def test_metrics_are_fleet_shaped():
    task = GateCourseTask(_single_gate())
    m = task.metrics(_fresh_state(4))
    assert set(m) == {"gate/active_idx", "gate/passes"}
    for v in m.values():
        assert v.shape == (4,) and v.dtype == jnp.float32
