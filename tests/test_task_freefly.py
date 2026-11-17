"""
The `freefly` playground task: protocol surface, the gate layout (split-S double gate
included), pass/hit counting that never crashes, and an env session under FREEFLY_ENV
where a hard landing does not reset the world.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.env import DomainRand, SimConfig, SkyFlowEnv
from skyflow.tasks import build_task
from skyflow.tasks.freefly import FREEFLY_ENV, FreeflyTask, FreeflyTaskState


def _plant(pos) -> jax.Array:
    pos = jnp.asarray(pos, jnp.float32).reshape(-1, 3)
    f = pos.shape[0]
    quat = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (f, 1))
    return jnp.concatenate([pos, jnp.zeros((f, 3)), quat, jnp.zeros((f, 7))], axis=-1)


def _state(f: int) -> FreeflyTaskState:
    zeros = jnp.zeros((f,), jnp.int32)
    return FreeflyTaskState(passes=zeros, hits=zeros)


def test_registered_and_contract():
    task = build_task("freefly")
    assert isinstance(task, FreeflyTask)
    assert task.success_terminates is False
    assert task.image_shape is None
    assert task.obs_spec.dim == 19
    plant, ts = task.spawn(jax.random.PRNGKey(0), 3, jnp.zeros((3, 1)))
    assert plant.shape == (3, 17) and plant.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(plant[:, 0:3]), 0.0)  # on the pad
    imu = (jnp.zeros((3, 3)), jnp.zeros((3, 3)))
    obs, _ = task.observe(plant, ts, imu, jnp.zeros((3, 4)), jax.random.PRNGKey(1), True)
    assert obs.shape == (3, 19)


def test_split_s_is_two_stacked_openings_facing_opposite_ways():
    gates = FreeflyTask().gates
    c = np.asarray(gates.centers_world)
    n = np.asarray(gates.normals_world)
    top, bottom = 2, 3
    np.testing.assert_allclose(c[top, :2], c[bottom, :2])
    gap = c[top, 2] - c[bottom, 2]
    outer_h = np.asarray(gates.outer_half)[top, 1]
    np.testing.assert_allclose(gap, 2.0 * outer_h, atol=1e-5)  # frame on frame
    np.testing.assert_allclose(n[top], -n[bottom], atol=1e-6)
    # every gate sits fully above ground (a pitched gate's vertical extent shrinks)
    lo = c[:, 2] - np.abs(np.asarray(gates.verticals_world)[:, 2]) * np.asarray(
        gates.outer_half
    )[:, 1]
    assert np.all(lo > 0.0)


def test_split_s_line_passes_both_openings_and_never_crashes():
    task = FreeflyTask()
    c = np.asarray(task.gates.centers_world)
    top, bottom = c[2], c[3]
    # in through the top along +x, then back out through the bottom along -x
    dx = np.array([1.0, 0.0, 0.0])
    prev = _plant([top - dx, bottom + dx])
    cur = _plant([top + dx, bottom - dx])
    ev = task.evaluate(prev, cur, _state(2))
    np.testing.assert_array_equal(np.asarray(ev.reward), [1.0, 1.0])
    assert not bool(jnp.any(ev.crash)) and not bool(jnp.any(ev.success))
    np.testing.assert_array_equal(np.asarray(ev.task_state.passes), [1, 1])


def test_frame_hit_is_counted_not_fatal():
    task = FreeflyTask()
    g0 = np.asarray(task.gates.centers_world)[0]
    bar = g0 + np.array([0.0, 0.0, 0.83])  # through the top bar of gate 0
    dx = np.array([0.5, 0.0, 0.0])
    ev = task.evaluate(_plant([bar - dx]), _plant([bar + dx]), _state(1))
    assert int(ev.task_state.hits[0]) == 1
    assert float(ev.reward[0]) == 0.0
    assert not bool(ev.crash[0])


def test_env_session_survives_a_hard_landing():
    """Motors mode under FREEFLY_ENV: drop from 3 m with idle motors, then sit for far
    longer than the default stuck/episode limits — no done, ever."""
    cfg = SimConfig(
        num_envs=2, task="freefly", dr=DomainRand().off(), **FREEFLY_ENV
    )
    env = SkyFlowEnv(cfg)
    step = jax.jit(env.step)
    _, state = env.reset(jax.random.PRNGKey(1))
    state = state.replace(plant=state.plant.at[:, 2].set(3.0))
    idle = -jnp.ones((2, 4), jnp.float32)
    for _ in range(1200):  # > default max_episode_steps (1000) and stuck_steps (200)
        _, state, _, done, _ = step(state, idle)
        assert not bool(jnp.any(done))
    assert bool(jnp.all(state.plant[:, 2] == 0.0))


def test_freefly_env_keys_are_simconfig_fields():
    names = {f.name for f in dataclasses.fields(SimConfig)}
    assert set(FREEFLY_ENV) <= names
