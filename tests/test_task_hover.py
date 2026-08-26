"""
DESIGN.md §11 (task tests, hover part) — HoverTask spawn/obs/reward logic on
dynamics-free synthetic plant states: plant rows [F,17] are built directly, so nothing
here depends on env.py or the dynamics backend. The task is imported directly (not via
the tasks/ registry, which test_registry.py covers).
"""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.tasks.hover import HoverTask, HoverTaskState


def _plant(pos, vel=None, omega=None) -> jax.Array:
    """[F,17] f32 spec-layout rows: given position/velocity/rates, identity attitude."""
    pos = jnp.asarray(pos, jnp.float32).reshape(-1, 3)
    f = pos.shape[0]
    vel = jnp.zeros((f, 3)) if vel is None else jnp.asarray(vel, jnp.float32).reshape(f, 3)
    omega = (
        jnp.zeros((f, 3)) if omega is None else jnp.asarray(omega, jnp.float32).reshape(f, 3)
    )
    quat = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (f, 1))
    return jnp.concatenate(
        [pos, vel, quat, omega, jnp.zeros((f, 4))], axis=-1
    ).astype(jnp.float32)


def _state(goal, hold=0) -> HoverTaskState:
    goal = jnp.asarray(goal, jnp.float32).reshape(-1, 3)
    f = goal.shape[0]
    return HoverTaskState(
        goal=goal,
        hold=jnp.full((f,), hold, jnp.int32),
        dist=jnp.zeros((f,), jnp.float32),
    )


GOAL = [0.0, 0.0, 1.5]


def _eval_at(task, d_prev: float, d: float, vel=None, omega=None):
    """Evaluate a transition along +x at the given distances from GOAL."""
    prev = _plant([[GOAL[0] + d_prev, GOAL[1], GOAL[2]]])
    cur = _plant([[GOAL[0] + d, GOAL[1], GOAL[2]]], vel=vel, omega=omega)
    return task.evaluate(prev, cur, _state([GOAL]))


# -- protocol surface ---------------------------------------------------------------------


def test_class_contract():
    task = HoverTask()
    assert task.success_terminates is False  # holding is the point (DESIGN.md §9)
    assert task.image_shape is None
    assert task.obs_spec.dim == 19
    assert list(task.obs_spec.layout) == ["rel_pos", "vel", "rot_matrix", "last_action"]


def test_constructor_validation():
    with pytest.raises(ValueError, match="safe box"):
        HoverTask(goal_xy_m=5.0, safe_xy_m=4.0)
    with pytest.raises(ValueError, match="safe box"):
        HoverTask(goal_z_max_m=6.0, safe_z_m=4.0)
    with pytest.raises(ValueError, match="goal_z"):
        HoverTask(goal_z_min_m=2.0, goal_z_max_m=1.0)
    with pytest.raises(ValueError, match="spawn jitter"):
        HoverTask(spawn_xy_m=2.0, spawn_dr_scale=3.0, safe_xy_m=4.0)


# -- spawn -----------------------------------------------------------------------------------


def test_spawn_pad_rows_and_goal_box(nominal_params):
    task = HoverTask()
    n = 64
    plant, state = task.spawn(jax.random.PRNGKey(2), n, nominal_params[:1])
    assert plant.shape == (n, 17) and plant.dtype == jnp.float32
    p = np.asarray(plant)
    np.testing.assert_array_equal(p[:, 2], 0.0)  # on the pad
    assert np.abs(p[:, 0:2]).max() <= task.spawn_xy_m + 1e-6
    assert np.abs(p[:, 0:2]).std() > 0.0  # XY genuinely jittered
    np.testing.assert_array_equal(p[:, 3:6], 0.0)
    np.testing.assert_allclose(np.linalg.norm(p[:, 6:10], axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(p[:, 10:13], 0.0)
    np.testing.assert_array_equal(p[:, 13:17], 0.0)  # motors at rest ("near idle")

    g = np.asarray(state.goal)
    assert g.shape == (n, 3) and state.goal.dtype == jnp.float32
    assert np.abs(g[:, 0:2]).max() <= task.goal_xy_m + 1e-6
    assert g[:, 2].min() >= task.goal_z_min_m and g[:, 2].max() <= task.goal_z_max_m
    np.testing.assert_array_equal(np.asarray(state.hold), 0)
    np.testing.assert_allclose(
        np.asarray(state.dist), np.linalg.norm(g - p[:, 0:3], axis=-1), rtol=1e-6
    )


def test_spawn_dr_scale_zero_pins_the_pad(nominal_params):
    task = HoverTask(spawn_dr_scale=0.0)
    plant, _ = task.spawn(jax.random.PRNGKey(2), 8, nominal_params[:1])
    np.testing.assert_array_equal(np.asarray(plant[:, 0:2]), 0.0)


# -- observe ----------------------------------------------------------------------------------


def _no_imu(f):
    return (jnp.zeros((f, 3), jnp.float32), jnp.zeros((f, 3), jnp.float32))


def test_observe_layout_identity_attitude():
    task = HoverTask()
    pos = jnp.asarray([[0.5, -0.25, 1.0]], jnp.float32)
    vel = jnp.asarray([[1.0, -2.0, 0.5]], jnp.float32)
    last_action = jnp.asarray([[0.1, 0.2, 0.3, 0.4]], jnp.float32)
    obs, state = task.observe(
        _plant(pos, vel), _state([GOAL]), _no_imu(1), last_action,
        jax.random.PRNGKey(0), fresh_spawn=False,
    )
    assert obs.shape == (1, 19) and obs.dtype == jnp.float32
    lay = task.obs_spec.layout
    o = np.asarray(obs[0], np.float64)
    np.testing.assert_allclose(o[lay["rel_pos"]], np.asarray(GOAL) - np.asarray(pos[0]), atol=1e-6)
    np.testing.assert_allclose(o[lay["vel"]], np.asarray(vel[0]), atol=1e-6)
    np.testing.assert_allclose(o[lay["rot_matrix"]], np.eye(3).ravel(), atol=1e-6)
    np.testing.assert_allclose(o[lay["last_action"]], np.asarray(last_action[0]), atol=1e-6)
    assert int(state.hold[0]) == 1  # the hold clock ticked


def test_observe_sanitizes_diverged_worlds():
    task = HoverTask()
    plant = _plant([[jnp.nan, 1e12, -1e12]], vel=[[jnp.inf, -jnp.inf, 0.0]])
    obs, _ = task.observe(
        plant, _state([GOAL]), _no_imu(1), jnp.zeros((1, 4), jnp.float32),
        jax.random.PRNGKey(0), fresh_spawn=False,
    )
    o = np.asarray(obs)
    assert np.isfinite(o).all() and np.abs(o).max() <= 100.0


def test_goal_resamples_every_hold_steps():
    task = HoverTask(goal_hold_s=0.05, control_hz=100.0)  # hold_steps = 5
    assert task.hold_steps == 5
    f = 4
    plant = _plant(np.zeros((f, 3)))
    la = jnp.zeros((f, 4), jnp.float32)
    _, state = task.spawn(jax.random.PRNGKey(7), f, jnp.zeros((1, 1)))
    goal0 = np.asarray(state.goal).copy()

    for t in range(1, 5):
        _, state = task.observe(
            plant, state, _no_imu(f), la, jax.random.PRNGKey(100 + t), fresh_spawn=False
        )
        np.testing.assert_array_equal(np.asarray(state.goal), goal0)
        assert int(state.hold[0]) == t
    _, state = task.observe(
        plant, state, _no_imu(f), la, jax.random.PRNGKey(200), fresh_spawn=False
    )
    assert not np.array_equal(np.asarray(state.goal), goal0), "goal must resample"
    np.testing.assert_array_equal(np.asarray(state.hold), 0)

    # fresh-spawn observations neither tick nor resample (spawn just drew these goals)
    goal1 = np.asarray(state.goal).copy()
    _, state = task.observe(
        plant, state, _no_imu(f), la, jax.random.PRNGKey(300), fresh_spawn=True
    )
    np.testing.assert_array_equal(np.asarray(state.goal), goal1)
    np.testing.assert_array_equal(np.asarray(state.hold), 0)


# -- evaluate ----------------------------------------------------------------------------------


def test_reward_increases_as_distance_falls():
    """Static hovers (no progress/penalty terms): r(d) = w_pos·e^(-3d) + w_hold·e^(-50d)
    is strictly decreasing in d, so closer must always pay more."""
    task = HoverTask()
    rewards = [float(_eval_at(task, d, d).reward[0]) for d in [2.0, 1.5, 1.0, 0.5, 0.2, 0.05]]
    assert all(b > a for a, b in pairwise(rewards)), rewards


def test_progress_term_pays_squared_distance_shrink():
    task = HoverTask()
    moving = float(_eval_at(task, 1.0, 0.5).reward[0])
    static = float(_eval_at(task, 0.5, 0.5).reward[0])
    assert moving - static == pytest.approx(task.w_prog * (1.0**2 - 0.5**2), abs=1e-5)


def test_speed_and_rate_penalties():
    task = HoverTask()
    base = float(_eval_at(task, 0.5, 0.5).reward[0])
    with_vel = float(_eval_at(task, 0.5, 0.5, vel=[[3.0, 0.0, 4.0]]).reward[0])
    with_rate = float(_eval_at(task, 0.5, 0.5, omega=[[0.0, 2.0, 0.0]]).reward[0])
    assert base - with_vel == pytest.approx(task.w_vel * 5.0, abs=1e-5)
    assert base - with_rate == pytest.approx(task.w_rate * 2.0, abs=1e-5)


def test_success_inside_radius_only():
    task = HoverTask()
    assert bool(_eval_at(task, 0.05, 0.05).success[0])
    assert not bool(_eval_at(task, 0.2, 0.2).success[0])


def test_crash_is_leaving_the_safe_box():
    task = HoverTask()  # safe box: |x|,|y| <= 4, z <= 4
    inside = _plant([[3.9, -3.9, 3.9]])
    for column, value in [(0, 5.0), (1, -5.0), (2, 5.0)]:
        out = inside.at[0, column].set(value)
        ev = task.evaluate(inside, out, _state([GOAL]))
        assert bool(ev.crash[0]), f"column {column} escape must crash"
    assert not bool(task.evaluate(inside, inside, _state([GOAL])).crash[0])


def test_evaluate_contract_shapes_and_state():
    task = HoverTask()
    f = 3
    prev = _plant(np.tile([1.0, 0.0, 1.5], (f, 1)))
    cur = _plant(np.tile([0.5, 0.0, 1.5], (f, 1)))
    ev = jax.jit(task.evaluate)(prev, cur, _state(np.tile(GOAL, (f, 1))))
    assert ev.reward.shape == (f,) and ev.reward.dtype == jnp.float32
    assert ev.success.dtype == jnp.bool_ and ev.crash.dtype == jnp.bool_
    assert set(ev.info) == {"hover/dist", "hover/success"}
    for v in ev.info.values():
        assert v.shape == (f,) and v.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(ev.task_state.dist), 0.5, atol=1e-6)


def test_metrics_are_fleet_shaped():
    task = HoverTask()
    m = task.metrics(_state(np.tile(GOAL, (4, 1)), hold=7))
    assert set(m) == {"hover/dist", "hover/goal_hold"}
    for v in m.values():
        assert v.shape == (4,) and v.dtype == jnp.float32
