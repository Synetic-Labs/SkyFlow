"""
DESIGN.md §11 — the dynamics adapter is a pure re-batching of the generated
backend. Runs under x64 (module-scoped, restored on teardown) so comparisons hold at the
golden 1e-9 tolerances. This is the one suite that imports skyflow_dynamics directly: its
whole point is composing the backend by hand against skyflow.dynamics.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from skyflow_dynamics.backends import jax as sfd

from skyflow import dynamics, sensors
from skyflow.params import AIRFRAMES, NEVER_JITTER, sample_params
from skyflow.types import DRState, SimState


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Adapter-grade tolerances need f64 — allowed for adapter tests only (DESIGN.md §3)."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _random_fleet(key, fleet, w_max):
    """Random plant rows + backend inputs at moderate, physical magnitudes."""
    ks = jax.random.split(key, 9)
    x = jax.random.normal(ks[0], (fleet, 3))
    v = 2.0 * jax.random.normal(ks[1], (fleet, 3))
    q = jax.random.normal(ks[2], (fleet, 4))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w = 3.0 * jax.random.normal(ks[3], (fleet, 3))
    rotor = jax.random.uniform(ks[4], (fleet, 4), minval=0.2 * w_max, maxval=0.9 * w_max)
    plant = jnp.concatenate([x, v, q, w, rotor], axis=-1)
    omega_cmd = jax.random.uniform(ks[5], (fleet, 4), minval=0.0, maxval=w_max)
    wind = 3.0 * jax.random.normal(ks[6], (fleet, 3))
    f_ext = 0.05 * jax.random.normal(ks[7], (fleet, 3))
    tau_ext = 1e-4 * jax.random.normal(ks[8], (fleet, 3))
    return plant, omega_cmd, wind, f_ext, tau_ext


def _hover_fleet(values, fleet):
    """Exact-hover rows: identity attitude, zero rates, Ω at Σ ct2·Ω² = m·g."""
    w_hover = math.sqrt(values["mass"] * values["grav"] / sum(values["ct2"]))
    row = jnp.concatenate(
        [
            jnp.array([0.0, 0.0, 1.0]),  # position is irrelevant to the IMU
            jnp.zeros(3),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            jnp.zeros(3),
            jnp.full(4, w_hover),
        ]
    )
    plant = jnp.tile(row, (fleet, 1))
    omega_cmd = jnp.full((fleet, 4), w_hover)
    wind = jnp.zeros((fleet, 3))
    params = jnp.tile(dynamics.pack_params(values), (fleet, 1))
    return plant, omega_cmd, wind, params


def test_substep_matches_backend_composition(crazyflie, fleet_size, key):
    params = sample_params(jax.random.PRNGKey(7), crazyflie, fleet_size, 1.0)
    params = params.astype(jnp.float64)
    plant, omega_cmd, wind, f_ext, tau_ext = _random_fleet(key, fleet_size, crazyflie.rotor_speed_max)
    dt = 1e-3

    got = dynamics.substep(
        plant, omega_cmd, wind, f_ext, tau_ext, params, dt,
        crazyflie.rotor_speed_min, crazyflie.rotor_speed_max,
    )
    assert got.shape == (fleet_size, dynamics.STATE_DIM)
    assert got.dtype == jnp.float64

    step = sfd.rk4_step_fn(dynamics.N_ROTORS, "first_order")
    for i in range(fleet_size):
        u_i = jnp.concatenate([omega_cmd[i], wind[i], f_ext[i], tau_ext[i]])
        want = sfd.post_step(
            step(plant[i], u_i, params[i], dt),
            crazyflie.rotor_speed_min, crazyflie.rotor_speed_max,
        )
        np.testing.assert_allclose(np.asarray(got[i]), np.asarray(want), rtol=1e-9, atol=1e-9)


def test_statedot_matches_backend_composition(crazyflie, fleet_size, key):
    params = sample_params(jax.random.PRNGKey(11), crazyflie, fleet_size, 1.0)
    params = params.astype(jnp.float64)
    plant, omega_cmd, wind, f_ext, tau_ext = _random_fleet(key, fleet_size, crazyflie.rotor_speed_max)

    got = dynamics.statedot(plant, omega_cmd, wind, f_ext, tau_ext, params)
    f = sfd.statedot_fn(dynamics.N_ROTORS, "first_order")
    for i in range(fleet_size):
        u_i = jnp.concatenate([omega_cmd[i], wind[i], f_ext[i], tau_ext[i]])
        want = f(plant[i], u_i, params[i])
        np.testing.assert_allclose(np.asarray(got[i]), np.asarray(want), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("k", [0.0, 0.35, 1.0])
def test_throttle_endpoints_and_monotonicity(crazyflie, k):
    w_min, w_max = crazyflie.rotor_speed_min, crazyflie.rotor_speed_max
    lo = dynamics.throttle_to_omega(jnp.zeros((2, 4)), w_min, w_max, k)
    hi = dynamics.throttle_to_omega(jnp.ones((2, 4)), w_min, w_max, k)
    np.testing.assert_allclose(np.asarray(lo), w_min, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(hi), w_max, rtol=1e-12)

    grid = jnp.linspace(0.0, 1.0, 257)[:, None] * jnp.ones((1, 4))
    omega = np.asarray(dynamics.throttle_to_omega(grid, w_min, w_max, k))
    assert np.all(np.diff(omega, axis=0) > 0.0)


def test_imu_exact_hover(crazyflie, fleet_size):
    plant, omega_cmd, wind, params = _hover_fleet(crazyflie.values, fleet_size)
    accel, gyro = dynamics.imu(plant, omega_cmd, wind, params)
    g = crazyflie.values["grav"]
    np.testing.assert_allclose(
        np.asarray(accel), np.tile([0.0, 0.0, g], (fleet_size, 1)), rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(np.asarray(gyro), 0.0, rtol=0.0, atol=1e-9)


def test_measure_wraps_imu_and_noise_hook(crazyflie, fleet_size):
    plant, omega_cmd, wind, params = _hover_fleet(crazyflie.values, fleet_size)
    plant = plant.astype(jnp.float32)
    omega_cmd = omega_cmd.astype(jnp.float32)
    wind = wind.astype(jnp.float32)
    params = params.astype(jnp.float32)

    accel, gyro = sensors.measure(plant, omega_cmd, wind, params)
    accel_ref, gyro_ref = dynamics.imu(plant, omega_cmd, wind, params)
    assert jnp.array_equal(accel, accel_ref) and jnp.array_equal(gyro, gyro_ref)

    noise_key = jax.random.PRNGKey(3)
    a1, g1 = sensors.measure(
        plant, omega_cmd, wind, params, key=noise_key, accel_noise_std=0.1, gyro_noise_std=0.01
    )
    a2, g2 = sensors.measure(
        plant, omega_cmd, wind, params, key=noise_key, accel_noise_std=0.1, gyro_noise_std=0.01
    )
    assert jnp.array_equal(a1, a2) and jnp.array_equal(g1, g2)  # same key → same corruption
    assert not jnp.array_equal(a1, accel_ref) and not jnp.array_equal(g1, gyro_ref)
    assert a1.dtype == jnp.float32 and g1.dtype == jnp.float32


def test_sample_params_structural_keys_untouched(crazyflie):
    fleet = 64
    rows = np.asarray(sample_params(jax.random.PRNGKey(5), crazyflie, fleet, 1.0))
    nominal = np.asarray(dynamics.pack_params(crazyflie.values)).astype(np.float32)
    slices = dynamics.param_slices(dynamics.N_ROTORS)
    for name in NEVER_JITTER:
        assert np.array_equal(rows[:, slices[name]], np.tile(nominal[slices[name]], (fleet, 1)))


def test_sample_params_jitter_within_brackets(crazyflie):
    from skyflow.params import DR_BRACKETS

    fleet = 64
    scale = 1.0
    rows = np.asarray(sample_params(jax.random.PRNGKey(6), crazyflie, fleet, scale))
    nominal = np.asarray(dynamics.pack_params(crazyflie.values)).astype(np.float32)
    slices = dynamics.param_slices(dynamics.N_ROTORS)

    for name, b in DR_BRACKETS.items():
        idx = slices[name]
        nonzero = nominal[idx] != 0.0
        if not np.any(nonzero):
            continue
        ratio = rows[:, idx][:, nonzero] / nominal[idx][nonzero]
        assert np.all(ratio >= 1.0 - scale * b - 1e-5), name
        assert np.all(ratio <= 1.0 + scale * b + 1e-5), name
        if b > 0.0:
            assert np.std(ratio) > 0.0, name  # the draw actually varies across worlds


def test_sample_params_zero_nominals_stay_zero(crazyflie):
    rows = np.asarray(sample_params(jax.random.PRNGKey(8), crazyflie, 64, 1.0))
    nominal = np.asarray(dynamics.pack_params(crazyflie.values)).astype(np.float32)
    assert np.all(rows[:, nominal == 0.0] == 0.0)


def test_sample_params_scale_zero_is_nominal_exactly(crazyflie, key):
    fleet = 16
    rows = sample_params(key, crazyflie, fleet, 0.0)
    assert rows.dtype == jnp.float32
    nominal = np.asarray(dynamics.pack_params(crazyflie.values)).astype(np.float32)
    assert np.array_equal(np.asarray(rows), np.tile(nominal, (fleet, 1)))


def test_simstate_is_a_scannable_pytree(fleet_size, nominal_params, key):
    f = fleet_size
    state = SimState(
        plant=jnp.zeros((f, dynamics.STATE_DIM), jnp.float32),
        params=nominal_params,
        key=key,
        wind_vel=jnp.zeros((f, 3), jnp.float32),
        dr_state=DRState(
            wind_mean=jnp.zeros((f, 3), jnp.float32),
            imu_bias=jnp.zeros((f, 6), jnp.float32),
            w_max=jnp.full((f,), AIRFRAMES["crazyflie"].rotor_speed_max, jnp.float32),
        ),
        act_buf=jnp.zeros((f, 3, 4), jnp.float32),
        delay_idx=jnp.zeros(f, jnp.int32),
        last_action=jnp.zeros((f, 4), jnp.float32),
        steps=jnp.zeros(f, jnp.int32),
        airborne=jnp.zeros(f, bool),
        ep_return=jnp.zeros(f, jnp.float32),
        ep_len=jnp.zeros(f, jnp.int32),
        crash_frac=jnp.zeros((), jnp.float32),
        success_frac=jnp.zeros((), jnp.float32),
        trunc_frac=jnp.zeros((), jnp.float32),
        ep_return_ema=jnp.zeros((), jnp.float32),
        ep_len_ema=jnp.zeros((), jnp.float32),
        task_state={"goal": jnp.zeros((f, 3), jnp.float32)},
    )

    def body(s, _):
        return s.replace(steps=s.steps + 1), s.steps

    out, stacked = jax.lax.scan(body, state, None, length=4)
    assert isinstance(out, SimState)
    assert np.all(np.asarray(out.steps) == 4)
    assert stacked.shape == (4, f)
    assert out.task_state["goal"].shape == (f, 3)
