"""
Per-world dynamics identity (DESIGN.md §6, §7): the `params_sampler` hook replaces the
multiplicative sampler at reset; `pin_dynamics` keeps each world's rows and delay draw
through auto-reset; `motor_model="asymmetric"` reduces to first_order when the spin-up
and spin-down coefficients equal 1/tau_m.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.dynamics import pack_params, param_slices
from skyflow.env import MOTOR_MODELS, DomainRand, SimConfig, SkyFlowEnv
from skyflow.params import AIRFRAMES

FLEET = 8
SLICES = param_slices(4)


def _nominal_rows(f: int) -> jax.Array:
    nominal = jnp.asarray(pack_params(AIRFRAMES["crazyflie"].values), jnp.float32)
    return jnp.broadcast_to(nominal, (f, nominal.shape[0]))


def _mass_sampler(key, f):
    """Per-world mass factors in [0.8, 1.2] on the nominal row."""
    factor = 1.0 + 0.2 * jax.random.uniform(key, (f, 1), jnp.float32, -1.0, 1.0)
    return _nominal_rows(f).at[:, SLICES["mass"]].multiply(factor)


def _env(pin: bool, sampler=None, **overrides) -> SkyFlowEnv:
    kwargs = dict(
        num_envs=FLEET,
        task="hover",
        control="motors",
        physics_hz=1000,
        control_hz=100.0,
        dr=DomainRand(scale=0.0, delay_steps=(0, 3)),
        max_episode_steps=3,
        stuck_steps=1000,
        pin_dynamics=pin,
    )
    kwargs.update(overrides)
    return SkyFlowEnv(SimConfig(**kwargs), params_sampler=sampler)


def _run(env: SkyFlowEnv, state, n: int, action: float = 0.0):
    step = jax.jit(env.step)
    a = jnp.full((FLEET, 4), action, jnp.float32)
    for _ in range(n):
        _, state, _, _, _ = step(state, a)
    return state


def test_sampler_rows_are_used_at_reset():
    _, s = _env(False, _mass_sampler).reset(jax.random.PRNGKey(0))
    masses = np.asarray(s.params[:, SLICES["mass"][0]])
    assert len(np.unique(masses)) > 1
    nominal = float(_nominal_rows(1)[0, SLICES["mass"][0]])
    assert np.all(np.abs(masses / nominal - 1.0) <= 0.2 + 1e-6)


def test_pinned_identity_survives_auto_reset():
    env = _env(True, _mass_sampler)
    _, s0 = env.reset(jax.random.PRNGKey(0))
    s = _run(env, s0, 4)  # every world truncates at step 3 and respawns
    assert int(s.steps.max()) == 1
    np.testing.assert_array_equal(np.asarray(s.params), np.asarray(s0.params))
    np.testing.assert_array_equal(np.asarray(s.delay_idx), np.asarray(s0.delay_idx))


def test_unpinned_identity_is_redrawn_at_auto_reset():
    env = _env(False, _mass_sampler)
    _, s0 = env.reset(jax.random.PRNGKey(0))
    s = _run(env, s0, 4)
    assert not np.array_equal(np.asarray(s.params), np.asarray(s0.params))


def test_sampler_shape_is_checked():
    env = _env(False, lambda key, f: jnp.zeros((f, 3), jnp.float32))
    with pytest.raises(ValueError, match="params_sampler"):
        env.reset(jax.random.PRNGKey(0))


def test_default_sampler_path_unchanged_by_pin_flag():
    """pin_dynamics only changes respawn; the reset draw is the §6 sampler either way."""
    key = jax.random.PRNGKey(3)
    _, a = _env(False).reset(key)
    _, b = _env(True).reset(key)
    np.testing.assert_array_equal(np.asarray(a.params), np.asarray(b.params))
    np.testing.assert_array_equal(np.asarray(a.plant), np.asarray(b.plant))


def test_asymmetric_motor_model_reduces_to_first_order():
    tau_m = float(AIRFRAMES["crazyflie"].values["tau_m"])

    def asym_rows(key, f):
        rows = _nominal_rows(f)
        for name in ("ka1", "kd1"):
            rows = rows.at[:, SLICES[name]].set(1.0 / tau_m)
        return rows

    key = jax.random.PRNGKey(1)
    env_first = _env(False, max_episode_steps=1000)
    env_asym = _env(False, asym_rows, max_episode_steps=1000, motor_model="asymmetric")
    _, s_first = env_first.reset(key)
    _, s_asym = env_asym.reset(key)
    s_first = _run(env_first, s_first, 20, action=0.4)
    s_asym = _run(env_asym, s_asym, 20, action=0.4)
    np.testing.assert_allclose(
        np.asarray(s_asym.plant), np.asarray(s_first.plant), rtol=1e-4, atol=1e-4
    )
    assert np.all(np.isfinite(np.asarray(s_asym.plant)))


def test_bad_motor_model_is_rejected():
    assert "asymmetric" in MOTOR_MODELS
    with pytest.raises(ValueError, match="motor_model"):
        _env(False, motor_model="bogus")
