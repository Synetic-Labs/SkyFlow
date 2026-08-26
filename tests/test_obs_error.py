"""
ERRORS.md — the L5 estimator-error model (errors.py) and its env threading:
profile resolution, structural invariants (unit quaternions, OU stationarity),
value-bit-exactness when off, dropout staleness, and the per-episode bias trait.
CPU motors-mode fleets only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import errors
from skyflow.env import DomainRand, SimConfig, SkyFlowEnv

FLEET = 16


def _env(obs_error, scale: float = 1.0, **dr_kwargs):
    dr = DomainRand(scale=scale, body_scale=0.0, obs_error=obs_error, **dr_kwargs)
    cfg = SimConfig(num_envs=FLEET, task="hover", control="motors",
                    physics_hz=1000, control_hz=100.0, dr=dr)
    return SkyFlowEnv(cfg)


# -- resolution / validation ---------------------------------------------------------

def test_unknown_profile_and_keys_raise():
    with pytest.raises(ValueError, match="profile"):
        errors.resolve_obs_error({"profile": "gps"})
    with pytest.raises(ValueError, match=r"unknown dr\.obs_error keys"):
        errors.resolve_obs_error({"profile": "mocap", "sigma": 1.0})
    with pytest.raises(ValueError, match="bias_frac"):
        errors.resolve_obs_error({"profile": "mocap", "bias_frac": -1.0})
    with pytest.raises(ValueError, match="p_drop"):
        errors.resolve_obs_error({"profile": "mocap", "p_drop": 1.0})
    assert errors.resolve_obs_error(None) is None


def test_fracs_scale_the_profile_widths():
    base = errors.resolve_obs_error({"profile": "mocap"})
    half = errors.resolve_obs_error({"profile": "mocap", "white_frac": 0.5})
    assert base is not None and half is not None
    np.testing.assert_allclose(np.asarray(half.white), 0.5 * np.asarray(base.white))
    assert half.rotor_rel == pytest.approx(0.5 * base.rotor_rel)
    assert half.bias == base.bias  # other components untouched


def test_master_scale_folds_into_the_fracs():
    dr = DomainRand(scale=0.5, obs_error={"profile": "mocap"}).effective()
    assert dr.obs_error is not None
    assert dr.obs_error["white_frac"] == pytest.approx(0.5)
    assert dr.obs_error["bias_frac"] == pytest.approx(0.5)
    # event knobs never scale
    assert "p_drop" not in dr.obs_error or dr.obs_error["p_drop"] == \
        errors.PROFILES["mocap"]["p_drop"]


# -- structural invariants -----------------------------------------------------------

def test_corrupted_quaternion_is_a_unit_quaternion():
    spec = errors.resolve_obs_error({"profile": "vio", "bias_frac": 5.0})
    assert spec is not None
    key = jax.random.PRNGKey(0)
    plant = jnp.zeros((64, 17), jnp.float32).at[:, 6].set(1.0)  # identity quats
    bias = errors.draw_bias(jax.random.PRNGKey(1), 64, spec)
    ou = jnp.zeros((64, 12), jnp.float32)
    est = errors.corrupt_plant(plant, bias, ou, key, spec)
    norms = np.linalg.norm(np.asarray(est[:, 6:10]), axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    assert not np.allclose(np.asarray(est[:, 6:10]), np.asarray(plant[:, 6:10]))


def test_rotor_estimate_is_relative_and_floored():
    spec = errors.resolve_obs_error({"profile": "mocap"})
    assert spec is not None
    plant = jnp.zeros((32, 17), jnp.float32).at[:, 6].set(1.0).at[:, 13:17].set(2000.0)
    est = errors.corrupt_plant(
        plant, jnp.zeros((32, 12)), jnp.zeros((32, 12)), jax.random.PRNGKey(2), spec
    )
    rel = np.asarray(est[:, 13:17]) / 2000.0 - 1.0
    assert np.abs(rel).max() <= spec.rotor_rel + 1e-6
    assert np.abs(rel).max() > 0.0
    zero = plant.at[:, 13:17].set(0.0)
    est0 = errors.corrupt_plant(
        zero, jnp.zeros((32, 12)), jnp.zeros((32, 12)), jax.random.PRNGKey(2), spec
    )
    assert np.asarray(est0[:, 13:17]).min() >= 0.0


def test_ou_stationary_std_matches_sigma():
    spec = errors.resolve_obs_error({"profile": "vio"})
    assert spec is not None
    ou = jnp.zeros((256, 12), jnp.float32)
    key = jax.random.PRNGKey(3)
    for i in range(4000):  # 40 s at 100 Hz >> max tau 5 s
        ou = errors.advance_ou(ou, jax.random.fold_in(key, i), 0.01, spec)
    got = np.asarray(ou).std(axis=0)
    want = np.asarray(spec.ou_sigma)
    on = want > 0
    np.testing.assert_allclose(got[on], want[on], rtol=0.15)
    np.testing.assert_allclose(got[~on], 0.0, atol=1e-7)


# -- env threading -------------------------------------------------------------------

def test_none_profile_is_value_identical_to_off():
    """The zero-width profile consumes fold_in keys only, so the legacy stream is
    untouched and every observation VALUE matches the obs_error=None env."""
    key = jax.random.PRNGKey(0)
    env_off, env_none = _env(None), _env({"profile": "none"})
    obs_off, s_off = env_off.reset(key)
    obs_none, s_none = env_none.reset(key)
    np.testing.assert_array_equal(np.asarray(obs_off), np.asarray(obs_none))
    step_off, step_none = jax.jit(env_off.step), jax.jit(env_none.step)
    a = jnp.zeros((FLEET, 4), jnp.float32)
    for _ in range(3):
        obs_off, s_off, _, _, _ = step_off(s_off, a)
        obs_none, s_none, _, _, _ = step_none(s_none, a)
    np.testing.assert_array_equal(np.asarray(obs_off), np.asarray(obs_none))
    np.testing.assert_array_equal(np.asarray(s_off.plant), np.asarray(s_none.plant))


def test_bias_is_a_per_episode_trait():
    env = _env({"profile": "mocap"})
    _, state = env.reset(jax.random.PRNGKey(0))
    bias0 = np.asarray(state.dr_state.est_bias)
    assert np.abs(bias0).max() > 0.0
    step = jax.jit(env.step)
    a = jnp.zeros((FLEET, 4), jnp.float32)
    done = jnp.zeros(FLEET, bool)
    for _ in range(3):
        _, state, _, done, _ = step(state, a)
    assert not bool(np.asarray(done).any())
    np.testing.assert_array_equal(np.asarray(state.dr_state.est_bias), bias0)


def test_position_estimate_differs_from_truth_but_tracks_it():
    env = _env({"profile": "vio"})
    _, state = env.reset(jax.random.PRNGKey(0))
    err = np.asarray(state.est_held[:, 0:3]) - np.asarray(state.plant[:, 0:3])
    assert np.abs(err).max() > 0.0
    assert np.abs(err).max() < 0.2  # cm-class profile, not meters


def test_dropout_holds_the_last_estimate():
    env = _env({"profile": "mocap", "p_drop": 0.999, "drop_mean_steps": 50.0})
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    a = jnp.zeros((FLEET, 4), jnp.float32)
    _, state, _, _, _ = step(state, a)  # holds begin here
    held1 = np.asarray(state.est_held)
    hold1 = np.asarray(state.est_hold)
    assert (hold1 > 0).mean() > 0.9
    _, state, _, _, _ = step(state, a)
    held2 = np.asarray(state.est_held)
    still = np.asarray(state.est_hold) > 0
    np.testing.assert_array_equal(held2[still & (hold1 > 0)], held1[still & (hold1 > 0)])
