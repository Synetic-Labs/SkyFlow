"""
The sticks production axis, end to end on the real CPU SITL (TECH_DEBT R4). The
shipped vanishing-gates bug lived exactly here — control="sticks" + gate task +
viz/record — and no test walked the combination. Small fleet, fixed keys; a short
max_episode_steps forces the in-jit auto-reset through the REAL firmware reset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import DomainRand, SimConfig, SkyFlowEnv

FLEET = 2
EP_STEPS = 8


def _skip_without_cpu_sitl() -> None:
    pytest.importorskip("cudaflight")
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        pytest.skip(f"libcpuflight unavailable: {e}")


@pytest.fixture(scope="module")
def sticks_gate_env():
    """Sticks + figure-eight gate course on the CPU SITL — the production shape."""
    _skip_without_cpu_sitl()
    env = SkyFlowEnv(SimConfig(
        num_envs=FLEET, task="figure_eight", control="sticks", firmware="cpu",
        max_episode_steps=EP_STEPS, dr=DomainRand().off(),
    ))
    yield env
    env.close()


def _aetr(throttle: float) -> jax.Array:
    return jnp.tile(jnp.array([0.0, 0.0, throttle, 0.0], jnp.float32), (FLEET, 1))


def test_sticks_gate_multistep_with_auto_reset(sticks_gate_env):
    env = sticks_gate_env
    _obs, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    action = _aetr(0.2)  # above hover idle: the fleet actually flies
    saw_done = False
    for _ in range(2 * EP_STEPS + 2):
        obs, state, reward, done, info = step(state, action)
        assert bool(jnp.isfinite(obs).all())
        assert reward.shape == (FLEET,) and done.shape == (FLEET,)
        assert "armed" in info and np.asarray(info["armed"]).dtype == np.bool_
        if bool(done.any()):
            saw_done = True
            # done worlds came back respawned: step counter re-zeroed in-jit,
            # and the REAL firmware reset ran for exactly those worlds
            assert bool((state.steps[np.asarray(done)] == 0).all())
    assert saw_done  # max_episode_steps drove at least one truncation round

    ts = env.task_state(state)
    assert ts.active_gate.shape == (FLEET,)


def test_sticks_record_binds_resolve(tmp_path, sticks_gate_env):
    """FlightLog on a REAL sticks env: the scene's task_state binds must resolve
    through the accessor and survive a save/load round trip — the exact axis the
    shipped bug silently broke."""
    from skyflow.viz.record import FlightLog

    env = sticks_gate_env
    log = FlightLog.for_env(env, watch=(0,))
    assert log.header["scene"], "gate task must declare a viz scene"
    _obs, state = env.reset(jax.random.PRNGKey(1))
    for _ in range(3):
        _obs, state, reward, done, _info = env.step(state, _aetr(0.2))
        log.capture(state, action=_aetr(0.2), reward=reward, done=done)
    path = log.save(tmp_path / "flight.npz")

    back = FlightLog.load(path)
    assert "task_state.active_gate" in back.binds
    assert back.binds["task_state.active_gate"].shape == (3, 1)


def test_sticks_snapshot_needs_the_accessor(sticks_gate_env):
    """snapshot() with task_state=env.task_state(state) resolves gate binds; the
    raw fallback RAISES on a sticks state instead of hiding every primitive."""
    from skyflow.viz.frame import snapshot
    from skyflow.viz.primitives import Gate, resolve

    env = sticks_gate_env
    _obs, state = env.reset(jax.random.PRNGKey(2))

    vf = snapshot(state, (0,), dt=env.dt_control, task_state=env.task_state(state))
    assert vf.task_state.active_gate.shape == (1,)
    gate = Gate(center=(0, 0, 1.2), lateral=(0, 1, 0), index=0,
                bind="task_state.active_gate")
    assert resolve(gate, vf) is not None  # the bind resolves — gates render

    with pytest.raises(TypeError, match="task_carry"):
        snapshot(state, (0,), dt=env.dt_control)  # raw read of the carry


# NOTE: the both-modes contract test lives in test_task_carry.py — this module's
# module-scoped fixture holds the ONE allowed CPU SITL fleet open, so no test in
# this file may construct another (the lifecycle guard rightly refuses).
