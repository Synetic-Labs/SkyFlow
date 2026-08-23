"""
DESIGN.md §7 — the battery-sag trait: a one-sided per-episode draw on the rotor-speed
ceiling (a pack only sags; the firmware sees nothing, full stick simply buys less
rotor speed). Measured anchor: Air75 II Racer battery_hover sysid, ~13% per pack.
CPU motors-mode fleets only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.env import DomainRand, SimConfig, SkyFlowEnv
from skyflow.params import AIRFRAMES

FLEET = 16
W_MAX = AIRFRAMES["crazyflie"].rotor_speed_max


def _env(sag: float, scale: float = 1.0):
    dr = DomainRand(scale=scale, body_scale=0.0, battery_sag=sag)
    cfg = SimConfig(num_envs=FLEET, task="hover", control="motors",
                    physics_hz=1000, control_hz=100.0, dr=dr)
    return SkyFlowEnv(cfg)


def test_sag_zero_keeps_the_airframe_ceiling_everywhere():
    env = _env(0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(state.dr_state.w_max),
                                  np.full(FLEET, W_MAX, np.float32))


def test_sag_draw_is_one_sided_and_spread():
    env = _env(0.13)
    _, state = env.reset(jax.random.PRNGKey(0))
    ratio = np.asarray(state.dr_state.w_max) / W_MAX
    assert ratio.max() <= 1.0 + 1e-6
    assert ratio.min() >= 0.87 - 1e-6
    assert ratio.std() > 0.01  # per-world trait, not a constant


def test_master_scale_folds_into_the_sag():
    env = _env(0.4, scale=0.5)
    _, state = env.reset(jax.random.PRNGKey(0))
    ratio = np.asarray(state.dr_state.w_max) / W_MAX
    assert ratio.min() >= 0.8 - 1e-6  # 0.5 * 0.4 = 0.2 worst case
    assert _env(0.4, scale=0.0).reset(jax.random.PRNGKey(0))[1].dr_state.w_max[0] == W_MAX


def test_full_throttle_saturates_at_the_per_world_ceiling():
    env = _env(0.3)
    _, state = env.reset(jax.random.PRNGKey(1))
    step = jax.jit(env.step)
    a = jnp.ones((FLEET, 4), jnp.float32)  # motors mode: full throttle every rotor
    for _ in range(60):  # ~6 motor time constants at 100 Hz
        _, state, _, _, _ = step(state, a)
    rotors = np.asarray(state.plant[:, 13:17])
    ceilings = np.broadcast_to(np.asarray(state.dr_state.w_max)[:, None], rotors.shape)
    np.testing.assert_allclose(rotors, ceilings, rtol=5e-3)


def test_sag_of_one_or_more_is_rejected():
    with pytest.raises(ValueError, match="battery_sag"):
        _env(1.0)
