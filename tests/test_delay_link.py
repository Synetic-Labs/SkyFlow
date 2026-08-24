"""
ERRORS.md L3 — the transport-link error models on the delay ring: per-step jitter
(delay_jitter_steps) and command drops (cmd_drop_prob, the RX's hold-last-command
ZOH). Includes the legacy bit-exactness guard: both knobs at zero reproduce the
plain env exactly. CPU motors-mode fleets only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.env import DomainRand, SimConfig, SkyFlowEnv

FLEET = 16


def _env(**dr_kwargs):
    dr = DomainRand(scale=0.0, **dr_kwargs)
    cfg = SimConfig(num_envs=FLEET, task="hover", control="motors",
                    physics_hz=1000, control_hz=100.0, dr=dr)
    return SkyFlowEnv(cfg)


def test_validation_rejects_bad_knobs():
    with pytest.raises(ValueError, match="delay_jitter_steps"):
        _env(delay_jitter_steps=-1)
    with pytest.raises(ValueError, match="cmd_drop_prob"):
        _env(cmd_drop_prob=1.0)


def test_zero_knobs_are_bit_exact_with_the_plain_env():
    key = jax.random.PRNGKey(0)
    env_a = _env()
    env_b = _env(delay_jitter_steps=0, cmd_drop_prob=0.0)
    obs_a, s_a = env_a.reset(key)
    obs_b, s_b = env_b.reset(key)
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
    step_a, step_b = jax.jit(env_a.step), jax.jit(env_b.step)
    a = jnp.full((FLEET, 4), 0.3, jnp.float32)
    for _ in range(5):
        obs_a, s_a, _, _, _ = step_a(s_a, a)
        obs_b, s_b, _, _, _ = step_b(s_b, a)
    np.testing.assert_array_equal(np.asarray(s_a.plant), np.asarray(s_b.plant))
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))


def test_certain_drop_makes_the_action_stream_irrelevant():
    """cmd_drop_prob ~ 1: no command ever lands, the link holds the reset-neutral
    forever — two different action streams must produce identical plants."""
    env = _env(cmd_drop_prob=0.999999)
    key = jax.random.PRNGKey(0)
    _, s_a = env.reset(key)
    _, s_b = env.reset(key)
    step = jax.jit(env.step)
    k = jax.random.PRNGKey(7)
    for i in range(4):
        act_a = jax.random.uniform(jax.random.fold_in(k, i), (FLEET, 4), jnp.float32, -1, 1)
        act_b = -act_a
        _, s_a, _, _, _ = step(s_a, act_a)
        _, s_b, _, _, _ = step(s_b, act_b)
    np.testing.assert_array_equal(np.asarray(s_a.plant), np.asarray(s_b.plant))


def test_drops_hold_the_previously_applied_command():
    """With drops off the rotors chase each new command; the drop path re-applies
    cmd_prev, so a certain-drop env's rotors keep chasing the FIRST command that
    ever landed (here: none — the neutral), while a no-drop env's rotors move."""
    env_drop = _env(cmd_drop_prob=0.999999)
    env_free = _env()
    key = jax.random.PRNGKey(1)
    _, s_d = env_drop.reset(key)
    _, s_f = env_free.reset(key)
    step_d, step_f = jax.jit(env_drop.step), jax.jit(env_free.step)
    a = jnp.ones((FLEET, 4), jnp.float32)  # full throttle
    for _ in range(30):
        _, s_d, _, _, _ = step_d(s_d, a)
        _, s_f, _, _, _ = step_f(s_f, a)
    assert np.asarray(s_f.plant[:, 13:17]).min() > np.asarray(s_d.plant[:, 13:17]).max()


def test_jitter_runs_and_respects_the_ring():
    """Jitter larger than the delay window still indexes inside the ring (the
    documented edge clip) — the env steps without error and stays finite."""
    env = _env(delay_steps=(1, 2), delay_jitter_steps=5)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    a = jnp.full((FLEET, 4), 0.5, jnp.float32)
    for _ in range(10):
        _, state, _, _, _ = step(state, a)
    assert bool(jnp.isfinite(state.plant).all())


def test_jitter_spreads_the_applied_command_across_worlds():
    """delay (2,2) with jitter 2: after ONE distinctive action followed by
    neutrals, worlds see that action land at different steps — cmd_prev differs
    across the fleet at a fixed step, which a constant delay cannot produce."""
    env = _env(delay_steps=(2, 2), delay_jitter_steps=2)
    _, state = env.reset(jax.random.PRNGKey(3))
    step = jax.jit(env.step)
    mark = jnp.full((FLEET, 4), 0.9, jnp.float32)
    neutral = jnp.zeros((FLEET, 4), jnp.float32)
    _, state, _, _, _ = step(state, mark)
    _, state, _, _, _ = step(state, neutral)
    applied = np.asarray(state.cmd_prev[:, 0], dtype=np.float64)  # 0.9 mark or 0.0
    assert sorted(set(np.round(applied, 3))) == [0.0, 0.9]
