"""
ERRORS.md — measured floors: for any scale > 0 a floored knob resolves to
floor + scale·(value − floor), so no dial setting produces a sim cleaner than
the measured hardware. Scale exactly 0 and knobs at exactly 0 keep the legacy
bit-exact zeros. The "obs_error_fracs" floor pins the estimator profile at
measured reality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.env import DomainRand, SimConfig, SkyFlowEnv

FLEET = 8


def test_floored_knob_interpolates_from_the_floor():
    dr = DomainRand(scale=0.25, gyro_noise_rps=0.02,
                    floors={"gyro_noise_rps": 0.0072}).effective()
    assert dr.gyro_noise_rps == pytest.approx(0.0072 + 0.25 * (0.02 - 0.0072))
    full = DomainRand(scale=1.0, gyro_noise_rps=0.02,
                      floors={"gyro_noise_rps": 0.0072}).effective()
    assert full.gyro_noise_rps == pytest.approx(0.02)  # scale 1 = the ceiling


def test_unfloored_knob_keeps_the_legacy_fold():
    dr = DomainRand(scale=0.25, gyro_noise_rps=0.02, wind_mean_mps=6.0,
                    floors={"gyro_noise_rps": 0.0072}).effective()
    assert dr.wind_mean_mps == pytest.approx(1.5)  # plain 0.25 * 6


def test_scale_zero_and_knob_zero_stay_exact_zero():
    dr = DomainRand(scale=0.0, gyro_noise_rps=0.02,
                    floors={"gyro_noise_rps": 0.0072}).effective()
    assert dr.gyro_noise_rps == 0.0
    dr = DomainRand(scale=0.5, gyro_noise_rps=0.0,
                    floors={"gyro_noise_rps": 0.0072}).effective()
    assert dr.gyro_noise_rps == 0.0  # off is off


def test_bad_floors_raise():
    with pytest.raises(ValueError, match="unknown dr.floors"):
        DomainRand(floors={"wind_tau_s": 0.5}).effective()  # a clock is not floorable
    with pytest.raises(ValueError, match="below its floor"):
        DomainRand(gyro_noise_rps=0.005, floors={"gyro_noise_rps": 0.0072}).effective()
    with pytest.raises(ValueError, match=">= 0"):
        DomainRand(floors={"gyro_noise_rps": -1.0}).effective()


def test_obs_error_fracs_floor_pins_the_profile_at_reality():
    base = {"profile": "mocap"}
    dr = DomainRand(scale=0.5, obs_error=dict(base),
                    floors={"obs_error_fracs": 1.0}).effective()
    assert dr.obs_error["white_frac"] == pytest.approx(1.0)  # reality at any scale
    stress = DomainRand(scale=0.5, obs_error={**base, "white_frac": 2.0},
                        floors={"obs_error_fracs": 1.0}).effective()
    assert stress.obs_error["white_frac"] == pytest.approx(1.5)  # 1 + 0.5*(2-1)
    with pytest.raises(ValueError, match="below the"):
        DomainRand(scale=0.5, obs_error={**base, "white_frac": 0.5},
                   floors={"obs_error_fracs": 1.0}).effective()
    legacy = DomainRand(scale=0.5, obs_error=dict(base)).effective()
    assert legacy.obs_error["white_frac"] == pytest.approx(0.5)  # no floor = legacy


def test_floored_env_scale_zero_is_bit_exact_with_the_plain_env():
    key = jax.random.PRNGKey(0)
    floors = {"gyro_noise_rps": 0.0072, "gyro_bias_rps": 7e-5}
    cfg_a = SimConfig(num_envs=FLEET, task="hover", control="motors",
                      physics_hz=1000, control_hz=100.0,
                      dr=DomainRand(scale=0.0, gyro_noise_rps=0.02))
    cfg_b = SimConfig(num_envs=FLEET, task="hover", control="motors",
                      physics_hz=1000, control_hz=100.0,
                      dr=DomainRand(scale=0.0, gyro_noise_rps=0.02, floors=floors))
    env_a, env_b = SkyFlowEnv(cfg_a), SkyFlowEnv(cfg_b)
    obs_a, s_a = env_a.reset(key)
    obs_b, s_b = env_b.reset(key)
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
    a = jnp.zeros((FLEET, 4), jnp.float32)
    for _ in range(3):
        _, s_a, _, _, _ = jax.jit(env_a.step)(s_a, a)
        _, s_b, _, _, _ = jax.jit(env_b.step)(s_b, a)
    np.testing.assert_array_equal(np.asarray(s_a.plant), np.asarray(s_b.plant))
