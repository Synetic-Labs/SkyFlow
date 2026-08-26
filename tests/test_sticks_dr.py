"""
The new DR knobs on the sticks axis (TECH_DEBT §7 D10/D11): battery_sag,
cmd_drop_prob, obs_error and the factor stage each run through the REAL CPU SITL
for a few jitted steps, and the estimator leaves clear on auto-reset. Each test
opens and closes its own fleet (one CPU fleet per process).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import DomainRand, SimConfig, SkyFlowEnv

FLEET = 2


def _skip_without_cpu_sitl() -> None:
    pytest.importorskip("cudaflight")
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        pytest.skip(f"libcpuflight unavailable: {e}")


_KNOBS = {
    "battery_sag": {"battery_sag": 0.15},
    "cmd_drop_prob": {"cmd_drop_prob": 0.3},
    "obs_error": {"obs_error": {"profile": "mocap"}},
    "factors": {"factors": {}},
}


@pytest.mark.parametrize("knob", sorted(_KNOBS))
def test_dr_knob_steps_through_the_real_firmware(knob):
    _skip_without_cpu_sitl()
    dr = DomainRand(body_scale=0.0, **_KNOBS[knob])
    cfg = SimConfig(num_envs=FLEET, task="hover", control="sticks", firmware="cpu", dr=dr)
    with SkyFlowEnv(cfg) as env:
        obs, state = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        aetr = jnp.tile(jnp.array([0.0, 0.0, 0.2, 0.0], jnp.float32), (FLEET, 1))
        for _ in range(5):
            obs, state, reward, _done, info = step(state, aetr)
            assert bool(jnp.isfinite(obs).all()) and bool(jnp.isfinite(reward).all())
            assert "armed" in info
        assert bool(jnp.isfinite(state.plant).all())


def test_estimator_leaves_clear_on_auto_reset():
    """obs_error drift/hold state is per-episode: a done world comes back with the OU
    state and the dropout hold at zero (the auto-reset blend covers the new leaves)."""
    dr = DomainRand(body_scale=0.0, obs_error={"profile": "vio"})
    cfg = SimConfig(num_envs=4, task="hover", max_episode_steps=3, dr=dr)
    with SkyFlowEnv(cfg) as env:
        _obs, state = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        a = jnp.zeros((4, env.act_dim), jnp.float32)
        done = jnp.zeros(4, bool)
        for _ in range(3):
            _obs, state, _r, done, _i = step(state, a)
        assert bool(done.all())  # max_episode_steps=3 truncates every world
        np.testing.assert_array_equal(np.asarray(state.est_ou), 0.0)
        np.testing.assert_array_equal(np.asarray(state.est_hold), 0)
        assert bool((state.steps == 0).all())
