# SkyFlow

A fleet-batched quadrotor **simulator** in pure JAX. Physics is generated from the
[SkyFlow-Dynamics](https://github.com/Synetic-Labs/SkyFlow-Dynamics) symbolic spec;
SkyFlow is the harness around it — stepping, domain randomization, disturbances,
sensors, vision, tasks. SkyFlow contains zero handwritten continuous dynamics and zero
training code (DESIGN.md is the contract of record).

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

**A platform plus tasks.** The env owns the plant scan (RK4 physics at 1 kHz under a
100 Hz control loop by default), per-world parameter randomization, OU wind, random
pokes, command transport delay, the ground-contact heuristic, the generic crash set,
and in-jit auto-reset. A **task** owns the objective: spawn distribution, observation,
reward, task-specific terminals. One platform, many tasks.

**Generated physics.** Every force, torque, and sensor equation lives in the
SkyFlow-Dynamics symbolic spec — verified against published sources and golden-tested
per backend. If a physics term is missing it is added there and consumed here, never
implemented here. The built-in airframe is the spec's system-identified Crazyflie;
`register_airframe` adds vehicles from spec parameter rows.

**Vision without a rasterizer.** `gate_course` in vision mode renders analytic ray-cast
coverage masks of the gate frames directly from pose, batched over the fleet, inside
jit — a segmentation mask, not RGB, which is what a perception front-end hands a policy
at deploy time. Persistent mask-corruption families ship in `skyflow.vision.mask_noise`.

**Firmware in the loop.** `control="sticks"` closes the loop through real Betaflight
firmware (the `cudaflight` SITL, `firmware` extra): AETR sticks in, per-motor duties
out, ticked at 1 kHz inside the substep scan. `control="motors"` is pure JAX and never
touches that seam. Consuming repos with their own GPU firmware fleet inject it via
`SkyFlowEnv(cfg, firmware_fleet=...)`.

## Install

```bash
uv sync                     # CPU
uv sync --extra cuda        # CUDA 13 wheels for JAX
uv sync --extra firmware    # + cudaflight for control="sticks"
```

Requires Python 3.12+. Runtime dependencies: `jax`, `numpy`, `skyflow-dynamics[jax]`.

## Tasks

`hover` and `gate_course` (with a figure-eight course builder) ship as reference tasks.
Research tasks live in the consuming repo: implement the `Task` protocol from
`skyflow.types` and register a builder —

```python
from skyflow import SimConfig, SkyFlowEnv, register_task

register_task("my_task", MyTask)
env = SkyFlowEnv(SimConfig(num_envs=1024, task="my_task", task_kwargs={...}))
```

The env only ever reaches a task through the protocol, so a registered task is a
first-class citizen — nothing special-cases the built-ins. Names are refused on
collision (variants register under their own names). `task_kwargs` passes to the
builder unmodified; the env-owned `spawn_dr_scale` and `control_hz` are forwarded to
builders that name them.

Course geometry for gate tasks lives in `skyflow.vision.gates`: `from_waypoints` takes
z-up world rows so a course is config data, and `line`/`circle`/`figure_eight` generate
the standard shapes.

## Conventions

World frame right-handed z-up; body FLU; quaternions wxyz scalar-first Hamilton
body→world; SI units, rotor speeds in rad/s — identical to SkyFlow-Dynamics, states
pass through untranslated. Every array the env creates is float32 with the fleet axis
`[F, ...]` leading. NED/FRD survives only inside `vision/` internals and at the
firmware sensor boundary.

## Scope

Deliberately **not** here:

- **Training code.** No RL algorithms, losses, replay, obs normalization, policy
  networks, or config frameworks. Training repos import SkyFlow and own all of that.
- **A differentiability claim.** The design avoids blockers (pure functions, no host
  state on the motors path) and the roadmap includes a differentiable variant, but
  `differentiable=True` raises `NotImplementedError("planned")` today rather than
  promising gradients that haven't been verified.

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

Any errors in the adaptation are ours, not theirs.

## License

MIT — see [LICENSE](LICENSE).
