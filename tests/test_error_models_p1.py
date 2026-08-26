"""
ERRORS.md P1 slice — gyro saturation + scale factor (L4), IMU mount pose trait
(L4, unpinning the generated imu_fn's constants), CoG offset (L1 geometry),
weight-relative + held pokes (L2), battery-sag start-charge shape and
within-episode discharge (L3). Every model off = bit-exact with the plain env.
CPU motors-mode fleets only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import dynamics, sensors
from skyflow.env import DomainRand, SimConfig, SkyFlowEnv
from skyflow.params import AIRFRAMES, apply_cog_offset, sample_params

FLEET = 16
CF = AIRFRAMES["crazyflie"]


def _env(**dr_kwargs):
    dr = DomainRand(scale=dr_kwargs.pop("scale", 1.0), body_scale=0.0, **dr_kwargs)
    cfg = SimConfig(num_envs=FLEET, task="hover", control="motors",
                    physics_hz=1000, control_hz=100.0, dr=dr)
    return SkyFlowEnv(cfg)


def _spinning_plant(f: int, rate: float) -> jax.Array:
    plant = jnp.zeros((f, 17), jnp.float32).at[:, 6].set(1.0)
    plant = plant.at[:, 10].set(rate).at[:, 13:17].set(2000.0)
    return plant


def _nominal(f: int) -> jax.Array:
    return sample_params(jax.random.PRNGKey(0), CF, f, 0.0)


# -- L4: saturation / scale / mount ---------------------------------------------------

def test_gyro_saturation_clips_the_measured_rate():
    plant = _spinning_plant(FLEET, 60.0)  # ~3400 dps true roll rate
    params = _nominal(FLEET)
    cmd = plant[:, 13:17]
    wind = jnp.zeros((FLEET, 3), jnp.float32)
    _, g_free = sensors.measure(plant, cmd, wind, params)
    _, g_sat = sensors.measure(plant, cmd, wind, params, gyro_sat_rps=34.9)
    assert float(jnp.abs(g_free).max()) > 34.9
    assert float(jnp.abs(g_sat).max()) <= 34.9 * (1.0 + 1e-5)
    np.testing.assert_array_equal(np.asarray(g_sat[:, 1:]), np.asarray(g_free[:, 1:]))


def test_gyro_scale_factor_multiplies_per_axis():
    plant = _spinning_plant(FLEET, 5.0)
    params = _nominal(FLEET)
    cmd = plant[:, 13:17]
    wind = jnp.zeros((FLEET, 3), jnp.float32)
    scale = jnp.ones((FLEET, 3), jnp.float32).at[:, 0].set(1.01)
    _, g_ref = sensors.measure(plant, cmd, wind, params)
    _, g_scl = sensors.measure(plant, cmd, wind, params, gyro_scale=scale)
    np.testing.assert_allclose(np.asarray(g_scl[:, 0]), 1.01 * np.asarray(g_ref[:, 0]),
                               rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(g_scl[:, 1:]), np.asarray(g_ref[:, 1:]))


def test_imu_mount_rotation_rotates_the_measurement():
    from skyflow.errors import rotvec_to_mat

    plant = _spinning_plant(FLEET, 2.0)
    params = _nominal(FLEET)
    cmd = plant[:, 13:17]
    wind = jnp.zeros((FLEET, 3), jnp.float32)
    a_ref, g_ref = dynamics.imu(plant, cmd, wind, params)
    rot = rotvec_to_mat(jnp.tile(jnp.asarray([[0.0, 0.0, jnp.pi / 2]]), (FLEET, 1)))
    a_rot, g_rot = dynamics.imu(plant, cmd, wind, params, mount=rot)
    # A pure rotation preserves the norms and moves the roll rate between axes.
    np.testing.assert_allclose(np.linalg.norm(np.asarray(a_rot), axis=1),
                               np.linalg.norm(np.asarray(a_ref), axis=1), rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(g_rot), axis=1),
                               np.linalg.norm(np.asarray(g_ref), axis=1), rtol=1e-5)
    assert not np.allclose(np.asarray(g_rot), np.asarray(g_ref))


def test_imu_offset_prices_the_lever_arm():
    plant = _spinning_plant(FLEET, 10.0)
    params = _nominal(FLEET)
    cmd = plant[:, 13:17]
    wind = jnp.zeros((FLEET, 3), jnp.float32)
    a_ref, _ = dynamics.imu(plant, cmd, wind, params)
    off = jnp.tile(jnp.asarray([[0.0, 0.02, 0.0]], jnp.float32), (FLEET, 1))
    a_off, _ = dynamics.imu(plant, cmd, wind, params, offset=off)
    # omega x (omega x r): 10 rad/s about x with r=2 cm on y -> ~2 m/s^2 centripetal.
    assert float(jnp.abs(a_off - a_ref).max()) > 1.0
    zero = jnp.zeros((FLEET, 3), jnp.float32)
    a_same, _ = dynamics.imu(plant, cmd, wind, params, offset=zero)
    np.testing.assert_allclose(np.asarray(a_same), np.asarray(a_ref), atol=1e-6)


# -- L1: CoG offset --------------------------------------------------------------------

def test_cog_offset_translates_every_rotor_common_mode():
    rows = _nominal(FLEET)
    off = jnp.tile(jnp.asarray([[0.005, -0.003, 0.001]], jnp.float32), (FLEET, 1))
    shifted = apply_cog_offset(rows, off)
    sl = dynamics.param_slices(4)["rotor_pos"]
    d = (np.asarray(shifted[:, sl]) - np.asarray(rows[:, sl])).reshape(FLEET, 4, 3)
    want = np.broadcast_to(-np.asarray(off)[:, None, :], d.shape)
    np.testing.assert_allclose(d, want, atol=1e-7)
    outside = np.ones(rows.shape[1], bool)
    outside[sl] = False
    np.testing.assert_array_equal(np.asarray(shifted[:, outside]),
                                  np.asarray(rows[:, outside]))


def test_cog_trait_reaches_the_params_row():
    env = _env(cog_offset_m=0.01)
    _, state = env.reset(jax.random.PRNGKey(0))
    sl = dynamics.param_slices(4)["rotor_pos"]
    nominal = np.asarray(_nominal(FLEET)[:, sl]).reshape(FLEET, 4, 3)
    got = np.asarray(state.params[:, sl]).reshape(FLEET, 4, 3)
    d = got - nominal
    # Common mode: all four rotors in a world share one delta; deltas vary by world.
    np.testing.assert_allclose(d.std(axis=1), 0.0, atol=1e-7)
    assert float(np.abs(d).max()) > 1e-4
    assert float(np.abs(d).max()) <= 0.01 + 1e-6


# -- L2: pokes -------------------------------------------------------------------------

def test_poke_fracs_resolve_against_the_nominal_weight():
    m_g = CF.values["mass"] * abs(CF.values["grav"])
    env = _env(poke_force_frac=0.5, poke_torque_frac=0.1, poke_prob=0.01)
    assert env._poke_force_n == pytest.approx(0.5 * m_g)
    r_arm = float(np.mean([np.hypot(r[0], r[1]) for r in CF.values["rotor_pos"]]))
    assert env._poke_torque_nm == pytest.approx(0.1 * m_g * r_arm)
    with pytest.raises(ValueError, match="not both"):
        _env(poke_force_n=0.1, poke_force_frac=0.5)


def test_held_pokes_persist_and_expire():
    env = _env(poke_prob=1.0, poke_force_frac=0.3, poke_dur_steps=50.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    a = jnp.zeros((FLEET, 4), jnp.float32)
    _, state, _, _, info = step(state, a)
    assert bool(np.asarray(info["poke_active"]).all())
    held1 = np.asarray(state.poke_fext)
    assert np.abs(held1).max() > 0.0
    left1 = np.asarray(state.poke_left)
    _, state, _, _, _ = step(state, a)
    still = np.asarray(state.poke_left) == left1 - 1
    np.testing.assert_array_equal(np.asarray(state.poke_fext)[still & (left1 > 0)],
                                  held1[still & (left1 > 0)])


# -- L3: battery -----------------------------------------------------------------------

def test_sag_shape_weights_starts_toward_a_full_pack():
    env_u = _env(battery_sag=0.4)
    env_s = _env(battery_sag=0.4, battery_sag_shape=6.0)
    w = CF.rotor_speed_max
    sag_u = 1.0 - np.asarray(env_u.reset(jax.random.PRNGKey(0))[1].dr_state.w_max) / w
    sag_s = 1.0 - np.asarray(env_s.reset(jax.random.PRNGKey(0))[1].dr_state.w_max) / w
    # Uniform mean = 0.2; U^6 mean = 0.4/7 ~ 0.057. Same key -> a paired comparison.
    assert sag_s.mean() < 0.5 * sag_u.mean()
    assert sag_s.max() <= 0.4 + 1e-6


def test_sag_ramp_lowers_the_ceiling_with_flight_time():
    """Deterministic: synthesize a mid-flight state (steps preset) and step once —
    the substep clip must enforce the ramped ceiling immediately. Auto-resets make a
    long full-throttle rollout reset `steps`, so the rollout form cannot pin this."""
    env = _env(battery_sag_rate_ps=0.05)  # ceiling falls 125 rad/s per flight second
    _, state = env.reset(jax.random.PRNGKey(1))
    a = jnp.ones((FLEET, 4), jnp.float32)
    w = CF.rotor_speed_max
    spun = state.replace(plant=state.plant.at[:, 13:17].set(w))  # rotors at the max
    deep = spun.replace(steps=jnp.full(FLEET, 800, jnp.int32))  # 8 s in: -1000 rad/s
    _, s_deep, _, _, _ = env.step(deep, a)
    ceiling = w - 0.05 * w * 8.0
    assert float(np.asarray(s_deep.plant[:, 13:17]).max()) <= ceiling * (1 + 1e-3)
    _, s_zero, _, _, _ = env.step(spun, a)
    assert float(np.asarray(s_zero.plant[:, 13:17]).max()) > 0.95 * w


# -- the off guard ---------------------------------------------------------------------

def test_all_new_knobs_off_is_bit_exact_with_the_plain_env():
    key = jax.random.PRNGKey(0)
    env_a = _env(scale=0.0)
    env_b = _env(scale=0.0, gyro_sat_rps=0.0, gyro_scale_frac=0.0, imu_offset_m=0.0,
                 imu_mount_deg=0.0, cog_offset_m=0.0, poke_force_frac=0.0,
                 poke_torque_frac=0.0, poke_dur_steps=1.0, battery_sag_shape=1.0,
                 battery_sag_rate_ps=0.0)
    obs_a, s_a = env_a.reset(key)
    obs_b, s_b = env_b.reset(key)
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
    step_a, step_b = jax.jit(env_a.step), jax.jit(env_b.step)
    act = jnp.full((FLEET, 4), 0.3, jnp.float32)
    for _ in range(5):
        obs_a, s_a, _, _, _ = step_a(s_a, act)
        obs_b, s_b, _, _, _ = step_b(s_b, act)
    np.testing.assert_array_equal(np.asarray(s_a.plant), np.asarray(s_b.plant))
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
