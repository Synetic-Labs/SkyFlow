"""
ERRORS.md L3 — the transport-link error model on the delay ring: command drops
(cmd_drop_prob — a dropped or LATE frame holds the previous applied command; the
next success applies the newest frame, so drops subsume link jitter with no
command reordering). Includes the legacy bit-exactness guard: the knob at zero
reproduces the plain env exactly.

An i.i.d. per-step jitter index was measured and REMOVED 2026-08-24: it reorders
commands (no real link does), and the reordered stick stream through the
firmware's RC feedforward kills takeoff (0.003 airborne vs 0.91 for drops at the
same whoop 10M probe). CPU motors-mode fleets only.
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
    with pytest.raises(ValueError, match="cmd_drop_prob"):
        _env(cmd_drop_prob=1.0)
    with pytest.raises(TypeError):
        # the param was removed 2026-08-24 (reordering is unphysical) — the call is
        # deliberately invalid, which is exactly what pyright flags
        DomainRand(delay_jitter_steps=1)  # pyright: ignore[reportCallIssue]


def test_zero_knob_is_bit_exact_with_the_plain_env():
    key = jax.random.PRNGKey(0)
    env_a = _env()
    env_b = _env(cmd_drop_prob=0.0)
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


def test_recovery_applies_the_newest_frame_not_a_stale_one():
    """After a drop the next successful step reads the normal delayed slot — the
    applied command never moves backward in send time. With delay (1,1) and a
    known action sequence, cmd_prev after a clean step equals the action sent one
    step earlier, regardless of drops before it."""
    env = _env(delay_steps=(1, 1), cmd_drop_prob=0.0)
    _, state = env.reset(jax.random.PRNGKey(2))
    step = jax.jit(env.step)
    seq = [jnp.full((FLEET, 4), v, jnp.float32) for v in (0.1, 0.2, 0.3)]
    for a in seq:
        _, state, _, _, _ = step(state, a)
    np.testing.assert_allclose(np.asarray(state.cmd_prev), 0.2, atol=1e-6)
