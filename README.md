# SkyFlow

An accurate and fast quadrotor simulator in JAX. Fully compiled for a pure GPU environment training simulation.

The physics dynamics is generated from the separate repo here:
[SkyFlow-Dynamics](https://github.com/Synetic-Labs/SkyFlow-Dynamics).
This maintains the symbolic spec, where each term is verified against published sources and per backend.
SkyFlow is the harness around that plant and inherits from it.

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

Every world steps together, nothing leaves the device, and done worlds respawn in-jit:
the pre-reset observation and flags come back through `info["final_obs"]` /
`info["terminated"]` / `info["truncated"]`, so a training loop never sees a dead state.

## What it is

### Environment

The env owns the world and everything that happens to the vehicle: RK4 physics, 
per-world parameter randomization, Ornstein–Uhlenbeck wind, random pokes, 
command transport delay, ground contact, crash detection, and in-jit auto-reset. 
A task owns the objective; spawn distribution, observation, reward, 
task-specific terminals.

### Airframe

The built-in airframe is the spec's Crazyflie reference row;
`register_airframe` adds vehicles from spec parameter rows.

### Vision inside jit

`gate_course` in vision mode renders analytic ray-cast coverage masks of the gate
frames directly from pose — no rasterizer, no host round-trip, batched over the fleet
inside jit. `skyflow.vision.mask_noise` corrupts the clean render with the artifact families;
branding holes, occluders, glow, speckle.

### firmware in the loop

A stick-level policy can fly through Betaflight, `control="sticks"` closes
the loop through the real firmware (the `cudaflight` SITL, `firmware` extra): AETR
sticks in, per-motor duties out, ticked at 1 kHz inside the substep scan. `control="motors"`
is pure raw motor output.

## Install

```bash
uv sync                     # CPU
uv sync --extra cuda        # CUDA 13 wheels for JAX
uv sync --extra firmware    # + cudaflight for control="sticks"
```

Requires Python 3.12+. Runtime dependencies: `jax`, `numpy`, `skyflow-dynamics[jax]`.

## Tasks

`hover` and `gate_course` (with a figure-eight course builder) ship as reference tasks.
Implement the `Task` protocol from `skyflow.types` and register a builder:

```python
from skyflow import SimConfig, SkyFlowEnv, register_task

register_task("my_task", MyTask)
env = SkyFlowEnv(SimConfig(num_envs=1024, task="my_task", task_kwargs={...}))
```

The env only ever reaches a task through the protocol, so a registered task is a
first-class citizen.`task_kwargs` passes to the builder unmodified; 
the env-owned `spawn_dr_scale` and `control_hz` are forwarded to builders that name them.

Course geometry for gate tasks lives in `skyflow.vision.gates`: `from_waypoints` takes
z-up world rows, and `line`/`circle`/`figure_eight` generate the standard shapes.

## Conventions

World frame right-handed z-up; body FLU; quaternions wxyz scalar-first Hamilton body→world;
 SI units, rotor speeds in rad/s. Every array the env creates is float32 with the fleet axis
`[F, ...]` leading. NED/FRD used only inside `vision/` internals and for firmware sensor.

## Development

```bash
uv run pytest -q         # full suite, CPU, deterministic keys
uv run ruff check .
uv run python examples/fly_hover.py
uv run python examples/fly_figure_eight.py --save-masks 6
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
