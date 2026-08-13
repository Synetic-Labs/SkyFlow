"""
DESIGN.md §11, suite 4 — ground contact (§8): the clamp never lets the vehicle
penetrate, a resting vehicle genuinely rests (no creep, no spurious dones), a hard fall
registers as a ground crash, and the spawned-on-pad hover task takes off under full
throttle. physics_dr_scale is 0 throughout: symmetric rotors keep open-loop full
throttle torque-free, so thresholds are exact.
"""

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.env import SimConfig, SkyFlowEnv
from skyflow.tasks.hover import HoverTask

FLEET = 4


def make_env(**cfg_kwargs) -> SkyFlowEnv:
    cfg_kwargs.setdefault("physics_dr_scale", 0.0)
    cfg_kwargs.setdefault("stuck_steps", 10**6)
    cfg_kwargs.setdefault("max_episode_steps", 10**6)
    cfg = SimConfig(num_envs=FLEET, **cfg_kwargs)
    return SkyFlowEnv(cfg, task=HoverTask(safe_xy_m=100.0, safe_z_m=100.0))


def test_no_penetration_and_hard_fall_is_a_ground_crash(key):
    """Free fall from 1 m with idle motors: impact speed ≈ 4.4 m/s. The state must never
    show z < 0 at any control boundary, and the impact must terminate the episode (the
    per-substep crash predicate — the contact clamp itself erases the descent)."""
    env = make_env()
    step = jax.jit(env.step)
    _, state = env.reset(key)
    state = state.replace(plant=state.plant.at[:, 2].set(1.0))
    idle = -jnp.ones((FLEET, 4), jnp.float32)

    crashed = np.zeros(FLEET, bool)
    for _ in range(80):
        _, state, _, _, info = step(state, idle)
        assert bool(jnp.all(state.plant[:, 2] >= 0.0)), "ground penetration"
        crashed |= np.asarray(info["terminated"])
    assert crashed.all(), "a 4.4 m/s impact must register as a ground crash"
    # post-crash worlds respawned on the pad
    assert bool(jnp.all(state.plant[:, 2] >= 0.0))


def test_resting_vehicle_stays_put(key):
    env = make_env()
    step = jax.jit(env.step)
    _, state = env.reset(key)
    start = np.asarray(state.plant[:, 0:3]).copy()
    idle = -jnp.ones((FLEET, 4), jnp.float32)
    for _ in range(50):
        _, state, _, done, _ = step(state, idle)
        assert not bool(done.any())
        np.testing.assert_array_equal(np.asarray(state.plant[:, 2]), 0.0)  # clamped, exact
    plant = np.asarray(state.plant)
    np.testing.assert_allclose(plant[:, 0:2], start[:, 0:2], atol=1e-4)
    np.testing.assert_array_equal(plant[:, 3:6], 0.0)  # velocities zeroed by the clamp
    np.testing.assert_array_equal(plant[:, 10:13], 0.0)  # body rates too
    np.testing.assert_array_equal(plant[:, 13:17], 0.0)  # Ω_c = w_min = 0 ⇒ rotors at rest
    assert not bool(state.airborne.any())


def test_pad_spawn_takes_off_under_full_throttle(key):
    """Nominal Crazyflie at full throttle: thrust/weight ≈ 1.95, motor spin-up ~90 ms,
    so 0.6 s clears half a metre comfortably — and the launch must not trip any of the
    crash or stuck logic on the way up."""
    env = make_env()
    step = jax.jit(env.step)
    _, state = env.reset(key)
    np.testing.assert_array_equal(np.asarray(state.plant[:, 2]), 0.0)  # pad spawn

    full = jnp.ones((FLEET, 4), jnp.float32)
    for _ in range(60):
        _, state, _, done, _ = step(state, full)
        assert not bool(done.any())
    assert bool(jnp.all(state.plant[:, 2] > 0.5)), "full throttle failed to lift off"
    assert bool(state.airborne.all())
    assert bool(jnp.all(state.plant[:, 5] > 0.0))  # still climbing
