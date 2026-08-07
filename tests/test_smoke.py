"""Smoke tests: the package imports, a fleet flies, and gradients flow.

Deliberately small and CPU-only — this is the gate that says "the extraction is intact",
not a physics validation suite. Everything here runs with `control="motors"` (pure JAX),
which is the control mode this distribution ships.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from skyflow import params, plant
from skyflow.env import SkyFlowEnv

FLEET = 8


def test_airframe_params_resolve():
    """Every registered airframe states all of its coefficients."""
    assert params.AIRFRAME_PARAMS, "no airframes registered"
    for name in params.AIRFRAME_PARAMS:
        p = params.airframe_params(name)
        row = p.to_array()
        assert row.shape == (len(params._PARAM_KEYS),)
        assert jnp.all(jnp.isfinite(row)), f"{name} has a non-finite coefficient"


def test_unknown_airframe_is_a_clear_error():
    with pytest.raises(ValueError, match="not-a-drone"):
        params.airframe_params("not-a-drone")


def test_custom_airframe_registers():
    """A downstream airframe registered by name resolves like a built-in."""
    params.register_airframe("smoke_clone", params.air75_ii_racer)
    try:
        assert params.airframe_params("smoke_clone") == params.airframe_params("air75_ii_racer")
    finally:
        params.AIRFRAME_PARAMS.pop("smoke_clone", None)


def test_reset_and_step():
    """A fleet resets and steps, and the obs/reward/done shapes hold."""
    env = SkyFlowEnv(num_envs=FLEET, task="hover", control="motors")
    obs, state = env.jax_reset(jax.random.key(0))
    assert obs.shape == (env.fleet, env.obs_dim)
    assert jnp.all(jnp.isfinite(obs))

    action = jnp.zeros((env.fleet, env.act_dim))
    obs, state, reward, done, info = jax.jit(env.jax_step)(state, action)

    assert obs.shape == (env.fleet, env.obs_dim)
    assert reward.shape == (env.fleet,)
    assert done.shape == (env.fleet,)
    assert jnp.all(jnp.isfinite(obs)) and jnp.all(jnp.isfinite(reward))
    assert isinstance(info, dict)


def test_rollout_is_stable():
    """A short rollout under full throttle stays finite (no NaN blow-up)."""
    env = SkyFlowEnv(num_envs=FLEET, task="hover", control="motors")
    obs, state = env.jax_reset(jax.random.key(1))
    step = jax.jit(env.jax_step)
    action = jnp.full((env.fleet, env.act_dim), 0.5)
    for _ in range(20):
        obs, state, reward, done, _ = step(state, action)
        assert jnp.all(jnp.isfinite(obs)), "obs went non-finite mid-rollout"
        assert jnp.all(jnp.isfinite(reward))


def test_differentiable_rollout_has_gradients():
    """differentiable=True gives a real gradient path through the dynamics.

    Deliberately a MULTI-step rollout. A single step carries no usable gradient and
    that is physics, not a severed path: the action sets a rotor-speed *target*, and
    with a ~39 ms motor lag one 11 ms control step moves the rotor — and therefore the
    position the reward reads — by less than float32 resolution. BPTT/APG differentiate
    through a horizon, which is what this asserts.
    """
    horizon = 30
    env = SkyFlowEnv(num_envs=FLEET, task="hover", control="motors", differentiable=True)
    _, state = env.jax_reset(jax.random.key(2))

    def loss(actions):
        s, total = state, 0.0
        for i in range(horizon):
            _, s, reward, _, _ = env.jax_step(s, actions[i])
            total = total + jnp.sum(reward)
        return -total

    g = jax.grad(loss)(jnp.zeros((horizon, FLEET, 4)))
    assert g.shape == (horizon, FLEET, 4)
    assert jnp.all(jnp.isfinite(g)), "non-finite gradient"
    assert jnp.any(g != 0.0), "gradient is identically zero — the path is severed"


def test_differentiable_rejects_firmware_control():
    """The flag is validated before any firmware import, so the error is clean."""
    with pytest.raises((ValueError, NotImplementedError), match="differentiable"):
        SkyFlowEnv(num_envs=FLEET, task="hover", control="sticks", differentiable=True)


def test_unknown_task_lists_what_is_registered():
    with pytest.raises(ValueError, match="unknown skyflow task"):
        SkyFlowEnv(num_envs=FLEET, task="no-such-task", control="motors")


def test_custom_task_registers():
    """A downstream task registered by name is constructible through the env."""
    from skyflow.tasks import register_task
    from skyflow.tasks.hover import HoverTask

    register_task("smoke_custom", HoverTask)
    env = SkyFlowEnv(num_envs=FLEET, task="smoke_custom", control="motors")
    obs, _ = env.jax_reset(jax.random.key(3))
    assert obs.shape == (env.fleet, env.obs_dim)


def test_param_row_matches_the_dynamics_contract():
    """The flattened coefficient row is the width the vectorised dynamics reads.

    A silent mismatch here would misalign every coefficient in the plant, so it is
    worth asserting directly rather than inferring it from a rollout looking sane.
    """
    row = params.airframe_params("air75_ii_racer").to_array()
    assert row.shape == (len(params._PARAM_KEYS),)
    # the plant indexes the row by name off the same tuple — if these ever diverge,
    # every coefficient after the first inserted key silently shifts.
    assert plant._PK is params._PARAM_KEYS
    assert len(set(params._PARAM_KEYS)) == len(params._PARAM_KEYS), "duplicate param key"
