"""
Structural R3: SimState stores the task slot in `task_carry`; `task_state` is a
guarded VIEW that raises on the sticks-mode firmware carry instead of handing it
out. The shipped vanishing-gates bug read the carry as task fields — it compiled
fine and silently hid every bound primitive. After this fix a raw read in sticks
mode is a TypeError that names the accessor; motors mode is unchanged.
"""

import jax
import jax.numpy as jnp
import pytest

from skyflow import DomainRand, SimConfig, SkyFlowEnv
from skyflow.types import FirmwareCarry


def _skip_without_cpu_sitl() -> None:
    pytest.importorskip("cudaflight")
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        pytest.skip(f"libcpuflight unavailable: {e}")


def test_motors_task_state_is_the_bare_pytree():
    with SkyFlowEnv(SimConfig(num_envs=2, task="hover", dr=DomainRand().off())) as env:
        _obs, state = env.reset(jax.random.PRNGKey(0))
        assert not isinstance(state.task_carry, FirmwareCarry)
        assert state.task_state is state.task_carry  # the view passes bare pytrees through
        assert env.task_state(state) is state.task_carry  # accessor identity in motors mode


def test_sticks_raw_task_state_read_raises():
    _skip_without_cpu_sitl()
    with SkyFlowEnv(SimConfig(
        num_envs=1, task="hover", control="sticks", firmware="cpu",
        dr=DomainRand().off(),
    )) as env:
        _obs, state = env.reset(jax.random.PRNGKey(0))
        assert isinstance(state.task_carry, FirmwareCarry)
        with pytest.raises(TypeError, match="task_carry"):
            _ = state.task_state
        assert hasattr(env.task_state(state), "goal")  # the accessor unwraps

        # the carry survives a step and the view stays guarded
        aetr = jnp.tile(jnp.array([0.0, 0.0, -1.0, 0.0], jnp.float32), (1, 1))
        _o, state2, _r, _d, _i = env.step(state, aetr)
        assert isinstance(state2.task_carry, FirmwareCarry)
        with pytest.raises(TypeError, match="task_carry"):
            _ = state2.task_state


def test_sticks_step_rejects_a_bare_task_pytree():
    """A state whose carry was replaced by the bare task pytree (e.g. rebuilt from
    a motors-mode state) must not step the firmware on task data."""
    _skip_without_cpu_sitl()
    with SkyFlowEnv(SimConfig(
        num_envs=1, task="hover", control="sticks", firmware="cpu",
        dr=DomainRand().off(),
    )) as env:
        _obs, state = env.reset(jax.random.PRNGKey(0))
        bad = state.replace(task_carry=env.task_state(state))
        aetr = jnp.tile(jnp.array([0.0, 0.0, -1.0, 0.0], jnp.float32), (1, 1))
        with pytest.raises(TypeError, match="firmware"):
            env.step(bad, aetr)


def test_control_mode_contract(control_mode):
    """One contract pass through BOTH control modes (the parametrized conftest
    fixture skips sticks where the SITL cannot boot): construct, reset, step,
    finite, close. Lives here — not in test_sticks_axis.py — because that module's
    module-scoped fixture holds the one allowed CPU SITL fleet open."""
    if control_mode == "sticks":
        cfg = SimConfig(num_envs=2, task="hover", control="sticks", firmware="cpu",
                        dr=DomainRand().off())
    else:
        cfg = SimConfig(num_envs=2, task="hover", control="motors",
                        dr=DomainRand().off())
    with SkyFlowEnv(cfg) as env:
        obs, state = env.reset(jax.random.PRNGKey(3))
        assert obs.shape == (2, env.obs_dim)
        action = jnp.tile(
            jnp.array([0.0, 0.0, -1.0 if control_mode == "sticks" else 0.0, 0.0],
                      jnp.float32),
            (2, 1),
        )
        for _ in range(3):
            obs, state, _reward, _done, _info = env.step(state, action)
            assert bool(jnp.isfinite(obs).all())
