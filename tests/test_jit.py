"""
DESIGN.md §11 (jit checks, vision half) — the vision task rolls out jitted, rendering
its coverage masks in-trace, with finite float32 observations and masks in [0, 1].

The env half of the jit checks — the jitted 50-step rollout with the NaN and retrace checks —
lives with the platform contract in
test_env_contract.py::test_jitted_50_step_rollout_no_nan_no_retrace. This file also
builds its env through the SimConfig(task=..., task_kwargs=...) registry path, so the
smoke covers the full §7 construction surface, not just an injected task.
"""

import jax
import jax.numpy as jnp

from skyflow.env import DomainRand, SimConfig, SkyFlowEnv

FLEET = 2
STEPS = 10


def test_vision_task_jitted_rollout_smoke():
    env = SkyFlowEnv(
        SimConfig(
            num_envs=FLEET,
            task="figure_eight",
            task_kwargs={"vision": True},
            dr=DomainRand(body_scale=0.0),
        )
    )
    assert env.image_shape is not None and env.image_shape[2] == 1
    h, w, _ = env.image_shape
    assert env.obs_spec.layout["mask"] == slice(0, h * w)
    assert env.obs_dim == h * w + 16  # mask + vel_body(3) + rot_matrix(9) + last_action(4)

    obs, state = env.reset(jax.random.PRNGKey(7))
    jstep = jax.jit(env.step)
    action = jnp.zeros((FLEET, 4), jnp.float32)  # mid throttle, no differential
    for _ in range(STEPS):
        obs, state, reward, _done, _info = jstep(state, action)
        assert bool(jnp.isfinite(reward).all()) and reward.dtype == jnp.float32

    assert obs.shape == (FLEET, env.obs_dim) and obs.dtype == jnp.float32
    assert bool(jnp.isfinite(obs).all())
    mask = obs[:, : h * w]
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
