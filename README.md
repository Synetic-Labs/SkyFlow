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

- The env is the world: RK4 physics, per-world parameter randomization, Ornstein-Uhlenbeck wind, random pokes, 
command transport delay, ground contact, crash detection, and in-jit auto-reset. 
- A task is the objective; spawn distribution, observation, reward, task-specific terminals.

### Airframe

The built-in airframe is the spec's Crazyflie reference row;
`register_airframe` adds vehicles from spec parameter rows.

### Vision

`figure_eight` in vision mode renders analytic ray-cast coverage masks of the gate
frames directly from pose without rasterizer or host round-trips, and batched over the fleet
inside jit.

### Firmware

A stick-level policy can fly through Betaflight, `control="sticks"` which closes
the loop through the real firmware (the `cudaflight` SITL, `firmware` extra)

### Viewer

The `viz` extra adds a live viewer. The camera is an analytic function of pose,
 and watching a GPU training run costs one host pull per chunk.

```bash
uv run python examples/fly_hover.py --seconds 30 --view
uv run python examples/fly_teleop.py --sticks joystick --record dvr/
uv run python -m skyflow.viz.replay dvr/lap_00.npz --pilot-cam 384x288
```

```python
from skyflow.viz import Viewer, FlightLog, Box, Marker

viewer = Viewer.for_env(env, watch=(0, 1))           # grid + glyphs + the task's scene
viewer.scene.add(Box(center=(3, -2, 0.5), half=(0.5, 0.5, 0.5)))   # your own props
...
viewer.frame(state, obs=obs, action=action, reward=reward, done=done, info=info)

log = FlightLog.for_env(env, watch=range(8), every=2)  # training side: arrays in,
log.extend(plant_buf, action=action_buf)               # one host pull per chunk
log.save("runs/042/flight.npz")
```

## Install

```bash
uv sync                     # CPU
uv sync --extra cuda        # CUDA 13 wheels for JAX
uv sync --extra firmware    # + cudaflight for control="sticks"
uv sync --extra viz         # + pygame for the viewer / teleop / replay
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

Physics steps are RK4 substeps of the plant at 1 kHz. 
Env steps are 100 Hz control with observation, reward, termination, and reset.

## Tasks

`hover` and `figure_eight` ship as reference tasks.

```python
from skyflow import SimConfig, SkyFlowEnv, register_task

register_task("my_task", MyTask)
env = SkyFlowEnv(SimConfig(num_envs=1024, task="my_task", task_kwargs={...}))
```

## Inspiration

- **[crazyflow](https://github.com/utiasDSL/crazyflow)** — the JAX-first, fleet-batched
  quadrotor-RL environment design. The sensor-synthesis seam and the general shape of a
  firmware-in-the-loop JAX rollout follow its lead.
- **SkyDreamer** — *"SkyDreamer: Interpretable End-to-End Vision-Based Drone Racing
  with Model-Based Reinforcement Learning"* (Diermayr et al., 2025,
  [arXiv:2510.14783](https://arxiv.org/abs/2510.14783)) — the gate-course reward shape
  ports its pass/centering machinery.
- Physics and coefficient provenance is tracked per term in the SkyFlow-Dynamics registry.

