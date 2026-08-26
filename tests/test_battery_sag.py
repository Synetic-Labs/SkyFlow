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


def test_tw_guard_prices_the_sagged_ceiling():
    """With factors on, the mass clamp must see the SAGGED ceiling: drawn thrust at
    each world's own w_max lifts the drawn vehicle at TW_FLOOR wherever the
    nominal-mass floor does not bind."""
    from skyflow.dynamics import N_ROTORS, pack_params, param_slices
    from skyflow.params import TW_FLOOR

    dr = DomainRand(scale=1.0, body_scale=1.0, battery_sag=0.3,
                    factors={"mass": (0.0, 0.9), "air_prop": (-0.1, 0.1)},
                    brackets={k: 0.0 for k in
                              ("mass", "inertia", "ct0", "ct1", "ct2", "cq0", "cq1",
                               "cq2", "tau_m", "I_rot", "k_d", "k_z")})
    cfg = SimConfig(num_envs=64, task="hover", control="motors",
                    physics_hz=1000, control_hz=100.0, dr=dr)
    env = SkyFlowEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(3))

    sl = param_slices(N_ROTORS)
    rows = np.asarray(state.params)
    w = np.asarray(state.dr_state.w_max)[:, None]
    thrust = (rows[:, sl["ct0"]].sum(-1) + (rows[:, sl["ct1"]] * w).sum(-1)
              + (rows[:, sl["ct2"]] * w * w).sum(-1))
    tw = thrust / (rows[:, sl["mass"]][:, 0] * 9.81)
    nominal_mass = pack_params(AIRFRAMES["crazyflie"].values)[sl["mass"]][0]
    floor_bound = rows[:, sl["mass"]][:, 0] <= nominal_mass * (1 + 1e-6)
    assert np.all((tw >= TW_FLOOR - 1e-3) | floor_bound)
    assert (tw >= TW_FLOOR - 1e-3).mean() > 0.5  # the guard actually engages
