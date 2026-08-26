"""
DESIGN.md §11 — the DomainRand block: master-scale folding, off() bit-exactness,
bracket overrides, per-episode traits (steady wind, IMU bias) and their respawn
redraw, sensor corruption reaching the firmware rows, and env-applied obs noise.

Fleets are small and keys fixed. Sticks-mode checks inject a row-recorder fleet
(types.FirmwareFleet stand-in) whose fwstate holds the last sensor rows the firmware
saw, so corruption is asserted at the exact boundary the real firmware consumes.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import sensors
from skyflow.dynamics import pack_params
from skyflow.env import DomainRand, SimConfig, SkyFlowEnv
from skyflow.params import AIRFRAMES, Airframe, register_airframe, sample_params
from skyflow.tasks.hover import HoverTask

FLEET = 8

# A draggy Crazyflie variant (c_D > 0) so steady wind bends trajectories: the nominal
# vehicle ships c_D = 0 and a mean-wind response needs a drag term to act through.
if "crazyflie_cd_dr" not in AIRFRAMES:
    _values = copy.deepcopy(AIRFRAMES["crazyflie"].values)
    _values["c_D"] = [0.02, 0.02, 0.02]
    register_airframe(
        "crazyflie_cd_dr",
        Airframe(
            name="crazyflie_cd_dr",
            values=_values,
            rotor_speed_min=AIRFRAMES["crazyflie"].rotor_speed_min,
            rotor_speed_max=AIRFRAMES["crazyflie"].rotor_speed_max,
            throttle_k=AIRFRAMES["crazyflie"].throttle_k,
        ),
    )


def make_env(dr: DomainRand, fleet: int = FLEET, **cfg_kwargs) -> SkyFlowEnv:
    cfg_kwargs.setdefault("stuck_steps", 10**6)
    cfg_kwargs.setdefault("max_episode_steps", 10**6)
    cfg = SimConfig(num_envs=fleet, dr=dr, **cfg_kwargs)
    return SkyFlowEnv(cfg, task=HoverTask(safe_xy_m=100.0, safe_z_m=100.0, goal_hold_s=1e6))


# -- master scale / off() --------------------------------------------------------------------


def test_off_is_bit_exact_nominal(key):
    env = make_env(DomainRand().off())
    _, state = env.reset(key)
    nominal = np.asarray(pack_params(AIRFRAMES["crazyflie"].values), np.float32)
    np.testing.assert_array_equal(
        np.asarray(state.params), np.broadcast_to(nominal, (FLEET, nominal.shape[0]))
    )
    np.testing.assert_array_equal(np.asarray(state.dr_state.wind_mean), 0.0)
    np.testing.assert_array_equal(np.asarray(state.dr_state.imu_bias), 0.0)
    assert bool(jnp.all(state.delay_idx == 0))


def test_scale_zero_disables_every_continuous_knob(key):
    """scale=0 must produce the same rollout as off(), whatever the magnitudes say."""
    hot = DomainRand(
        scale=0.0, wind_mean_mps=5.0, wind_gust_mps=3.0, poke_force_n=1.0,
        gyro_noise_rps=1.0, accel_noise_mps2=1.0, gyro_bias_rps=1.0,
        accel_bias_mps2=1.0, obs_noise=1.0, poke_prob=0.5,
    )
    a = jnp.zeros((FLEET, 4), jnp.float32)

    def rollout(dr):
        env = make_env(dr)
        obs, state = env.reset(key)
        for _ in range(3):
            obs, state, _, _, _ = env.step(state, a)
        return obs, state

    obs_h, s_h = rollout(hot)
    obs_o, s_o = rollout(DomainRand().off())
    np.testing.assert_array_equal(np.asarray(obs_h), np.asarray(obs_o))
    np.testing.assert_array_equal(np.asarray(s_h.plant), np.asarray(s_o.plant))
    np.testing.assert_array_equal(np.asarray(s_h.wind_vel), 0.0)


def test_effective_folds_the_master_scale():
    dr = DomainRand(
        scale=0.5, body_scale=1.0, wind_mean_mps=2.0, wind_gust_mps=1.0,
        wind_tau_s=0.3, poke_prob=0.2, poke_force_n=0.1, poke_torque_nm=0.01,
        delay_steps=(1, 2), gyro_noise_rps=0.02, accel_noise_mps2=0.04,
        gyro_bias_rps=0.01, accel_bias_mps2=0.03, baro_noise_pa=4.0,
        obs_noise=0.06, spawn_scale=2.0,
    )
    eff = dr.effective()
    assert eff.scale == 1.0
    assert eff.body_scale == pytest.approx(0.5)
    assert eff.wind_mean_mps == pytest.approx(1.0)
    assert eff.wind_gust_mps == pytest.approx(0.5)
    assert eff.poke_force_n == pytest.approx(0.05)
    assert eff.gyro_noise_rps == pytest.approx(0.01)
    assert eff.obs_noise == pytest.approx(0.03)
    # never scaled: clocks, event rates, integer delays, spawn spread
    assert eff.wind_tau_s == 0.3
    assert eff.poke_prob == 0.2
    assert eff.delay_steps == (1, 2)
    assert eff.spawn_scale == 2.0


# -- bracket overrides -----------------------------------------------------------------------


def test_bracket_override_pins_a_key_while_others_jitter(key, crazyflie):
    from skyflow.dynamics import N_ROTORS, param_slices

    sl = param_slices(N_ROTORS)
    rows = sample_params(key, crazyflie, 64, 1.0, {"mass": 0.0})
    nominal = np.asarray(pack_params(crazyflie.values), np.float32)
    np.testing.assert_array_equal(
        np.asarray(rows[:, sl["mass"]]),
        np.broadcast_to(nominal[sl["mass"]], (64, len(sl["mass"]))),
    )
    ct2 = np.asarray(rows[:, sl["ct2"]])  # nonzero nominal on the Crazyflie (ct0/ct1 are 0)
    assert np.std(ct2) > 0.0  # everything else still jitters
    assert np.std(ct2, axis=1).max() > 0.0  # per-rotor entries draw independently


def test_bad_bracket_overrides_and_body_scale_fail_loudly():
    with pytest.raises(ValueError, match="unknown SCHEMA key"):
        SkyFlowEnv(SimConfig(num_envs=2, dr=DomainRand(brackets={"nope": 0.1})), task=HoverTask())
    with pytest.raises(ValueError, match="structural key"):
        SkyFlowEnv(SimConfig(num_envs=2, dr=DomainRand(brackets={"grav": 0.1})), task=HoverTask())
    with pytest.raises(ValueError, match="max bracket"):
        SkyFlowEnv(SimConfig(num_envs=2, dr=DomainRand(body_scale=4.0)), task=HoverTask())
    with pytest.raises(ValueError, match="delay"):
        SkyFlowEnv(SimConfig(num_envs=2, dr=DomainRand(delay_steps=(2, 1))), task=HoverTask())


# -- traits: steady wind + IMU bias ------------------------------------------------------------


def test_wind_mean_trait_is_horizontal_and_bounded(key):
    ceiling = 2.0
    env = make_env(DomainRand(body_scale=0.0, wind_mean_mps=ceiling))
    _, state = env.reset(key)
    wm = np.asarray(state.dr_state.wind_mean)
    np.testing.assert_array_equal(wm[:, 2], 0.0)
    mags = np.linalg.norm(wm, axis=-1)
    assert (mags <= ceiling + 1e-6).all() and mags.std() > 0.0
    # gust state stays exactly zero with wind_gust_mps = 0 — the mean is a separate trait
    a = jnp.zeros((FLEET, 4), jnp.float32)
    _, state, _, _, _ = env.step(state, a)
    np.testing.assert_array_equal(np.asarray(state.wind_vel), 0.0)
    assert float(env.metrics(state)["wind_speed_mean"]) > 0.0


def test_wind_mean_reaches_the_aerodynamics(key):
    """Same key, mean wind on vs off, airborne draggy fleet: positions must differ."""

    def run(mean):
        env = make_env(
            DomainRand(body_scale=0.0, wind_mean_mps=mean), airframe="crazyflie_cd_dr"
        )
        _, state = env.reset(key)
        state = state.replace(plant=state.plant.at[:, 2].set(2.0))
        a = jnp.zeros((FLEET, 4), jnp.float32)
        for _ in range(10):
            _, state, _, done, _ = env.step(state, a)
            assert not bool(done.any())
        return np.asarray(state.plant)

    calm = run(0.0)
    windy = run(3.0)
    assert np.abs(windy[:, 0:3] - calm[:, 0:3]).max() > 1e-4


def test_traits_are_redrawn_at_respawn(key):
    env = make_env(
        DomainRand(body_scale=0.0, wind_mean_mps=2.0, gyro_bias_rps=0.05),
        max_episode_steps=1, stuck_steps=10**6,
    )
    _, state = env.reset(key)
    before = np.asarray(state.dr_state.wind_mean)
    bias_before = np.asarray(state.dr_state.imu_bias)
    _, state, _, done, _ = env.step(state, jnp.zeros((FLEET, 4), jnp.float32))
    assert bool(done.all())  # every world truncates at step 1 and respawns in-jit
    assert not np.array_equal(np.asarray(state.dr_state.wind_mean), before)
    assert not np.array_equal(np.asarray(state.dr_state.imu_bias), bias_before)


# -- sensors.measure corruption -----------------------------------------------------------------


def test_measure_bias_and_noise_hooks(key, nominal_params, fleet_size):
    plant = jnp.zeros((fleet_size, 17), jnp.float32).at[:, 6].set(1.0).at[:, 2].set(1.0)
    omega = plant[:, 13:17]
    wind = jnp.zeros((fleet_size, 3), jnp.float32)
    a0, g0 = sensors.measure(plant, omega, wind, nominal_params)
    bias = jnp.tile(jnp.asarray([[0.1, -0.2, 0.3, 0.01, -0.02, 0.03]], jnp.float32), (fleet_size, 1))
    a1, g1 = sensors.measure(plant, omega, wind, nominal_params, imu_bias=bias)
    np.testing.assert_allclose(np.asarray(a1 - a0), np.asarray(bias[:, 0:3]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(g1 - g0), np.asarray(bias[:, 3:6]), atol=1e-6)
    a2, g2 = sensors.measure(
        plant, omega, wind, nominal_params, key=key, accel_noise_std=0.5, gyro_noise_std=0.05
    )
    a3, g3 = sensors.measure(
        plant, omega, wind, nominal_params, key=key, accel_noise_std=0.5, gyro_noise_std=0.05
    )
    np.testing.assert_array_equal(np.asarray(a2), np.asarray(a3))  # same key, same noise
    np.testing.assert_array_equal(np.asarray(g2), np.asarray(g3))
    assert not np.array_equal(np.asarray(a2), np.asarray(a0))
    assert not np.array_equal(np.asarray(g2), np.asarray(g0))


# -- obs noise is env-applied -------------------------------------------------------------------


def test_obs_noise_is_applied_by_the_env(key):
    clean_env = make_env(DomainRand(body_scale=0.0))
    noisy_env = make_env(DomainRand(body_scale=0.0, obs_noise=0.05))
    obs_c, state_c = clean_env.reset(key)
    obs_n, state_n = noisy_env.reset(key)
    d = np.abs(np.asarray(obs_n) - np.asarray(obs_c))
    assert d.max() > 0.0 and d.max() <= 0.05 + 1e-6
    a = jnp.zeros((FLEET, 4), jnp.float32)
    obs_c2, _, _, _, _ = clean_env.step(state_c, a)
    obs_n2, _, _, _, _ = noisy_env.step(state_n, a)
    d2 = np.abs(np.asarray(obs_n2) - np.asarray(obs_c2))
    assert d2.max() > 0.0 and d2.max() <= 0.05 + 1e-6


# -- sticks mode: corruption reaches the firmware rows ------------------------------------------


class _RowRecorderFleet:
    """types.FirmwareFleet stand-in whose fwstate holds the LAST sensor rows fw_step
    saw — after a step, state.task_carry.fwstate is exactly what the firmware would
    have consumed on the final 1 kHz substep. Motors stay at zero (grounded fleet)."""

    act_dim = 4

    def __init__(self, fleet: int):
        self.fleet = fleet

    def fresh_firmware_state(self):
        return jnp.zeros((0,), jnp.uint8), jnp.zeros((self.fleet, 7), jnp.float32)

    def fw_step(self, blob, fwstate, sticks, sensors):
        return blob, sensors, jnp.zeros((self.fleet, 4), jnp.float32), jnp.ones(
            (self.fleet,), jnp.uint8
        )

    def reset(self, blob, fwstate, mask):
        return blob, fwstate

    def close(self):
        pass


def _sticks_env(dr: DomainRand, fleet: int = 4) -> SkyFlowEnv:
    return SkyFlowEnv(
        SimConfig(
            num_envs=fleet, control="sticks", dr=dr,
            stuck_steps=10**6, max_episode_steps=10**6,
        ),
        task=HoverTask(safe_xy_m=100.0, safe_z_m=100.0, goal_hold_s=1e6),
        firmware_fleet=_RowRecorderFleet(fleet),
    )


def _rows_after_step(env, key, n_steps=1):
    _, state = env.reset(key)
    a = jnp.zeros((4, 4), jnp.float32)
    out = []
    for _ in range(n_steps):
        _, state, _, _, _ = env.step(state, a)
        out.append(np.asarray(state.task_carry.fwstate))
    return out


def test_exact_sensing_gives_exactly_zero_gyro_rows_on_the_pad(key):
    (rows,) = _rows_after_step(_sticks_env(DomainRand().off()), key)
    np.testing.assert_array_equal(rows[:, 0:3], 0.0)  # grounded, zero body rates, no DR


def test_gyro_noise_reaches_the_firmware_rows(key):
    r1, r2 = _rows_after_step(
        _sticks_env(DomainRand(body_scale=0.0, gyro_noise_rps=0.05)), key, n_steps=2
    )
    assert np.abs(r1[:, 0:3]).max() > 0.0
    assert not np.array_equal(r1[:, 0:3], r2[:, 0:3])  # a process: new draw every sample


def test_gyro_bias_reaches_the_firmware_rows_and_holds(key):
    half = 0.05
    r1, r2 = _rows_after_step(
        _sticks_env(DomainRand(body_scale=0.0, gyro_bias_rps=half)), key, n_steps=2
    )
    assert np.abs(r1[:, 0:3]).max() > 0.0
    assert np.abs(r1[:, 0:3]).max() <= half + 1e-6
    np.testing.assert_array_equal(r1[:, 0:3], r2[:, 0:3])  # a trait: constant in-episode


def test_baro_noise_reaches_the_firmware_rows(key):
    quiet1, quiet2 = _rows_after_step(_sticks_env(DomainRand().off()), key, n_steps=2)
    np.testing.assert_array_equal(quiet1[:, 6], quiet2[:, 6])
    n1, n2 = _rows_after_step(
        _sticks_env(DomainRand(body_scale=0.0, baro_noise_pa=5.0)), key, n_steps=2
    )
    assert not np.array_equal(n1[:, 6], n2[:, 6])


def test_obs_noise_skips_mask_terms(key):
    """dr.obs_noise is unit-blind (LEGACY knob): a half-width sane for metres used
    to DESTROY the mask pixels riding in the same vector (TECH_DEBT C8).
    Mask-valued terms are excluded; numeric terms keep the blanket."""
    kw: dict = {"num_envs": 2, "task": "figure_eight", "task_kwargs": {"vision": True}}
    with SkyFlowEnv(SimConfig(dr=DomainRand(body_scale=0.0), **kw)) as clean, \
         SkyFlowEnv(SimConfig(dr=DomainRand(body_scale=0.0, obs_noise=0.5), **kw)) as noisy:
        assert clean.obs_spec[0].name == "mask"
        d = clean.obs_spec[0].dim
        obs_c, _ = clean.reset(key)
        obs_n, _ = noisy.reset(key)
        # the mask block is untouched by obs_noise; the numeric tail is corrupted
        np.testing.assert_array_equal(np.asarray(obs_n[:, :d]), np.asarray(obs_c[:, :d]))
        assert float(np.abs(np.asarray(obs_n[:, d:] - obs_c[:, d:])).max()) > 0.0


def test_obs_error_camera_images_from_the_true_pose(key):
    """dr.obs_error corrupts the ESTIMATE the policy reads. A camera is bolted to the
    vehicle: the rendered mask must come from the true pose, so the image block is
    identical with the estimator error on or off (TECH_DEBT §7 D1)."""
    kw: dict = {"num_envs": 2, "task": "figure_eight", "task_kwargs": {"vision": True}}
    err = {"profile": "mocap", "bias_frac": 50.0}  # a metre-class pose error
    with SkyFlowEnv(SimConfig(dr=DomainRand(body_scale=0.0), **kw)) as clean, \
         SkyFlowEnv(SimConfig(dr=DomainRand(body_scale=0.0, obs_error=err), **kw)) as est:
        d = clean.obs_spec[0].dim
        obs_c, _ = clean.reset(key)
        obs_e, _ = est.reset(key)
        np.testing.assert_array_equal(np.asarray(obs_e[:, :d]), np.asarray(obs_c[:, :d]))
        assert float(np.abs(np.asarray(obs_e[:, d:] - obs_c[:, d:])).max()) > 0.0


def test_obs_error_zero_attitude_widths_keep_the_quaternion_bit_exact():
    """A zero attitude width promises the true attitude untouched — no rotation
    compose, no renormalization (TECH_DEBT §7 D2; the old path moved non-identity
    quaternions by ~1e-7)."""
    import dataclasses

    from skyflow import errors

    spec = errors.resolve_obs_error({"profile": "mocap"})
    assert spec is not None

    def zero_att(t):
        return tuple(0.0 if 6 <= i < 9 else v for i, v in enumerate(t))

    spec = dataclasses.replace(
        spec, bias=zero_att(spec.bias), ou_sigma=zero_att(spec.ou_sigma),
        white=zero_att(spec.white),
    )
    q = jax.random.normal(jax.random.PRNGKey(4), (16, 4), jnp.float32)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)  # random unit quaternions
    plant = jnp.zeros((16, 17), jnp.float32).at[:, 6:10].set(q)
    bias = errors.draw_bias(jax.random.PRNGKey(5), 16, spec)
    est = errors.corrupt_plant(plant, bias, jnp.zeros((16, 12), jnp.float32),
                               jax.random.PRNGKey(6), spec)
    np.testing.assert_array_equal(np.asarray(est[:, 6:10]), np.asarray(q))
    assert float(np.abs(np.asarray(est[:, 0:3])).max()) > 0.0  # position still corrupted
