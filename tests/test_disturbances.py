"""
DESIGN.md §11, suite 3 — disturbances: OU wind stationary statistics, poke rate, and the
proof that wind reaches the aerodynamics through the backend's exogenous inputs (and is
exactly zero when disabled).

Statistical assertions use large fleets with fixed keys and tolerances several standard
errors wide (noted inline), so they are deterministic in practice. Rollouts idle on the
ground pad — the vehicle is irrelevant to the wind/poke processes, and a grounded fleet
guarantees no auto-reset clears the wind state mid-measurement (stuck truncation is
disabled per test).
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.env import SimConfig, SkyFlowEnv
from skyflow.params import AIRFRAMES, Airframe, register_airframe
from skyflow.tasks.hover import HoverTask

# A Crazyflie variant with parasitic drag, for the wind→aerodynamics check: the nominal
# vehicle ships c_D = 0 (RotorPy lineage), so a drag response needs this registration.
# 0.02 N/(m/s)² on 30 g gives ~6 m/s² at a 3 m/s gust — unmissable in a few steps.
if "crazyflie_cd" not in AIRFRAMES:
    _values = copy.deepcopy(AIRFRAMES["crazyflie"].values)
    _values["c_D"] = [0.02, 0.02, 0.02]
    register_airframe(
        "crazyflie_cd",
        Airframe(
            name="crazyflie_cd",
            values=_values,
            rotor_speed_min=AIRFRAMES["crazyflie"].rotor_speed_min,
            rotor_speed_max=AIRFRAMES["crazyflie"].rotor_speed_max,
            throttle_k=AIRFRAMES["crazyflie"].throttle_k,
        ),
    )


def make_env(fleet: int, **cfg_kwargs) -> SkyFlowEnv:
    cfg_kwargs.setdefault("physics_dr_scale", 0.0)
    cfg_kwargs.setdefault("stuck_steps", 10**6)
    cfg_kwargs.setdefault("max_episode_steps", 10**6)
    cfg = SimConfig(num_envs=fleet, **cfg_kwargs)
    return SkyFlowEnv(cfg, task=HoverTask(safe_xy_m=100.0, safe_z_m=100.0, goal_hold_s=1e6))


def _scan_rollout(env, state, action, n_steps: int, collect):
    """jit-scanned rollout collecting `collect(state', info)` per step."""

    def body(s, _):
        _, s2, _, done, info = env.step(s, action)
        return s2, (collect(s2, info), done)

    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps))(state)


# -- OU wind ---------------------------------------------------------------------------------


def test_ou_wind_stationary_std_matches_config(key):
    """Exact OU discretization ⇒ stationary per-axis std equals wind_std_mps. 256 worlds
    x 250 post-burn-in steps x 3 axes at decay ≈ 0.967 give ≈3.2k effective samples: the
    0.12 tolerances sit beyond 4 std errors of both the mean and std estimators."""
    fleet, std, tau = 256, 1.5, 0.3
    env = make_env(fleet, wind_std_mps=std, wind_tau_s=tau, physics_hz=500.0)
    _, state = env.reset(key)
    idle = -jnp.ones((fleet, 4), jnp.float32)
    _, (winds, dones) = _scan_rollout(env, state, idle, 400, lambda s, i: s.wind_vel)
    assert not bool(dones.any()), "a reset would have cleared wind state mid-measurement"

    sample = np.asarray(winds[150:])  # burn-in: var reaches var_inf·(1-decay^300) ≈ var_inf to 0.1%
    assert abs(sample.std() - std) < 0.12
    assert abs(sample.mean()) < 0.12


def test_wind_stays_exactly_zero_when_disabled(key):
    fleet = 16
    env = make_env(fleet, wind_std_mps=0.0)
    _, state = env.reset(key)
    idle = -jnp.ones((fleet, 4), jnp.float32)
    _, (winds, _) = _scan_rollout(env, state, idle, 30, lambda s, i: s.wind_vel)
    assert np.asarray(winds).shape == (30, fleet, 3)
    np.testing.assert_array_equal(np.asarray(winds), 0.0)


# -- pokes -----------------------------------------------------------------------------------


def test_poke_rate_matches_poke_prob(key):
    """poke_force_n = 0 keeps the draws physically inert (no crash/reset interference)
    while info["poke_active"] still reports the Bernoulli gate. n = 256·200 draws ⇒
    se ≈ 0.0016; tolerance 0.015 is ≈9 standard errors."""
    fleet, prob = 256, 0.15
    env = make_env(fleet, poke_prob=prob, poke_force_n=0.0)
    _, state = env.reset(key)
    idle = -jnp.ones((fleet, 4), jnp.float32)
    _, (pokes, _) = _scan_rollout(env, state, idle, 200, lambda s, i: i["poke_active"])
    rate = float(np.asarray(pokes, np.float64).mean())
    assert abs(rate - prob) < 0.015

    env0 = make_env(fleet=16, poke_prob=0.0)
    _, state0 = env0.reset(key)
    _, (pokes0, _) = _scan_rollout(
        env0, state0, -jnp.ones((16, 4), jnp.float32), 20, lambda s, i: i["poke_active"]
    )
    assert not bool(np.asarray(pokes0).any())


def test_pokes_shove_through_exogenous_inputs(key):
    """Same key, same actions, poke on vs off: the trajectories must split — and only
    through the backend's F_ext/τ_ext inputs, which is all the poke branch touches."""
    fleet = 8

    def run(poke_force, poke_torque):
        env = make_env(
            fleet, poke_prob=1.0, poke_force_n=poke_force, poke_torque_nm=poke_torque
        )
        _, state = env.reset(key)
        state = state.replace(plant=state.plant.at[:, 2].set(2.0))  # airborne, clamp inert
        a = jnp.zeros((fleet, 4), jnp.float32)
        for _ in range(3):
            _, state, _, done, _ = env.step(state, a)
            assert not bool(done.any())
        return np.asarray(state.plant)

    quiet = run(0.0, 0.0)
    shoved = run(0.05, 1e-5)
    assert np.abs(shoved[:, 0:6] - quiet[:, 0:6]).max() > 1e-5
    assert np.abs(shoved[:, 10:13] - quiet[:, 10:13]).max() > 1e-4


# -- wind → aerodynamics -----------------------------------------------------------------------


def test_wind_enters_aerodynamics_on_a_draggy_vehicle(key):
    """On the c_D > 0 variant a 3 m/s OU wind must bend the trajectory: same key, same
    actions, wind on vs off, airborne mid-throttle fleet — positions must differ."""
    fleet = 8

    def run(wind_std):
        env = make_env(fleet, airframe="crazyflie_cd", wind_std_mps=wind_std)
        _, state = env.reset(key)
        state = state.replace(plant=state.plant.at[:, 2].set(2.0))
        a = jnp.zeros((fleet, 4), jnp.float32)
        for _ in range(10):
            _, state, _, done, _ = env.step(state, a)
            assert not bool(done.any())
        return np.asarray(state.plant)

    calm = run(0.0)
    windy = run(3.0)
    assert np.abs(windy[:, 0:3] - calm[:, 0:3]).max() > 1e-4, (
        "wind never reached the aerodynamics"
    )
