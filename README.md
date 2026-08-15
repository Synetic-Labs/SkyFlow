# SkyFlow

An accurate and fast quadrotor simulator in JAX. Advanced physics, massive parallel training.

The dynamics is generated from the [SkyFlow-Dynamics](https://github.com/Synetic-Labs/SkyFlow-Dynamics).
That maintains the symbolic spec, and SkyFlow is the harness.

```python
import jax
import jax.numpy as jnp
from skyflow import SimConfig, SkyFlowEnv

env = SkyFlowEnv(SimConfig(num_envs=4096, task="hover"))

obs, state = env.reset(jax.random.PRNGKey(0))
step = jax.jit(env.step)  # pure functions — the caller jits

action = jnp.zeros((env.fleet, env.act_dim))  # [F,4] in [-1,1]
obs, state, reward, done, info = step(state, action)
```
## What it is

### Environment

The env owns the world and everything that happens to the vehicle: RK4 physics, 
per-world parameter randomization, Ornstein-Uhlenbeck wind, random pokes, 
command transport delay, ground contact, crash detection, and in-jit auto-reset. 
A task owns the objective; spawn distribution, observation, reward, 
task-specific terminals.

### Airframe

The built-in airframe is the spec's Crazyflie reference row;
`register_airframe` adds vehicles from spec parameter rows.

### Vision inside jit

`figure_eight` in vision mode renders analytic ray-cast coverage masks of the gate
frames directly from pose — no rasterizer, no host round-trip, batched over the fleet
inside jit.

### firmware in the loop

A stick-level policy can fly through Betaflight, `control="sticks"` closes
the loop through the real firmware (the `cudaflight` SITL, `firmware` extra)

## Install

```bash
uv sync                     # CPU
uv sync --extra cuda        # CUDA 13 wheels for JAX
uv sync --extra firmware    # + cudaflight for control="sticks"
```

Runtime dependencies: `jax`, `numpy`, `skyflow-dynamics[jax]`.

## Performance

Single-vehicle fleets on an NVIDIA RTX 4090:

| worlds | physics steps/s | env steps/s |
|---|---|---|
| 64 | 1.3 M | 107 K |
| 1024 | 23.7 M | 2.0 M |
| 16384 | 487 M | 22.5 M |
| 65536 | 1.08 B | 35.5 M |

Physics steps are bare RK4 substeps of the plant at 1 kHz. 
Env steps are 100 Hz control with observation, reward, termination, and in-jit reset.

## Tasks

`hover` and `figure_eight` ship as reference tasks.

```python
from skyflow import SimConfig, SkyFlowEnv, register_task

register_task("my_task", MyTask)
env = SkyFlowEnv(SimConfig(num_envs=1024, task="my_task", task_kwargs={...}))
```


## Credits

SkyFlow stands on work that came before it:

- **[crazyflow](https://github.com/utiasDSL/crazyflow)** — the JAX-first, fleet-batched
  quadrotor-RL environment design. The sensor-synthesis seam and the general shape of a
  firmware-in-the-loop JAX rollout follow its lead.
- **SkyDreamer** — *"SkyDreamer: Interpretable End-to-End Vision-Based Drone Racing
  with Model-Based Reinforcement Learning"* (Diermayr et al., 2025,
  [arXiv:2510.14783](https://arxiv.org/abs/2510.14783)) — the gate-course reward shape
  ports its pass/centering machinery.
- Physics and coefficient provenance is tracked per term in the SkyFlow-Dynamics
  registry, source by source.

## License

MIT — see [LICENSE](LICENSE).
