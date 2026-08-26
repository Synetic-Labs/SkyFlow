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

### Viewer

The `viz` extra (pygame only) adds a live viewer — wireframe scene, drone glyphs, an
honest FPV pane pair, instruments — plus gamepad/keyboard/UDP teleop and flight-log
replay. The scene is data: five primitives (`Grid`, `Path`, `Gate`, `Box`, `Marker`)
that any task or user composes, so it draws whatever you are training. Logs store poses,
never pixels — the camera is a pure analytic function of pose, so a recorded flight
re-renders at any resolution later, and watching a GPU training run costs one small
host pull per chunk.

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

#### Window controls

| input | action |
|---|---|
| left-drag | orbit the camera about the scene centre (the near side follows the cursor) |
| right/middle-drag | pan |
| wheel | zoom about the cursor |
| `Space` | pause / resume |
| `Tab` | focus the next watched world |
| `V` | view preset: iso / top / profile (each switch lands on a fresh fit) |
| `C` | follow the focused world (keeps it centred; orbit/zoom still apply) |
| `X` | full glyphs for ALL watched worlds (default: unfocused worlds are dots) |
| `G` | whole-fleet scatter on/off (a dot per world, strided on device) |
| `P` | print a screenshot to the working directory |
| `R` | reset request (consumed by hosts that call `take_reset`) |
| `←` `→` | scrub one step (replay, paused) |
| `[` `]` | playback speed |
| `Esc` | quit |

The key list is always visible in a bar along the window's bottom edge. The top bar
shows the clock, the step, drawn/fed fps, and the snapshot-drop count.

#### Viewer options

`Viewer.for_env(env, watch=(0,), **kw)` wires everything below from the env; pass any
of them to override. `Viewer(scene, **kw)` takes the same options directly.

| option | default | meaning |
|---|---|---|
| `scene` | task's `viz_scene()` + `Grid` | the display world (primitives) |
| `watch` | `(0,)` | fleet rows to snapshot and draw (`Tab` cycles them) |
| `camera` | task's `camera` | lens/mount for the FPV panes |
| `gates` | task's `gates` | gate geometry for the pilot cam; `None` = floor/horizon |
| `image_shape`, `obs_layout` | from env (vision tasks) | enable the verbatim policy-obs pane |
| `image_term` | `"mask"` | name of the image block inside the obs layout |
| `omega_max` | airframe's `rotor_speed_max` | rotor-speed normaliser for arcs and bars |
| `control` | env's | `"sticks"` (AETR crosses + arm lamp) or `"motors"` (action bars) |
| `dt` | env's control period | turns steps into the clock readout |
| `task_state_of` | `env.task_state` | state → task pytree (scene binds resolve against it) |
| `title` | task · control | window caption |
| `size` | `(1280, 800)` | window size in pixels |
| `display_hz` | `60.0` | draw-rate cap; `frame()` calls beyond it return without drawing |
| `headless` | `False` | SDL dummy driver (CI, screenshots, export) |
| `frames` | `None` | auto-close after N drawn frames |
| `shot` | `None` | screenshot path written when `frames` runs out |
| `threaded` | `True` | render thread owns the window; `False` = frame-exact draws (replay/export) |
| `pilot`, `policy_floor` | analytic renderers | FPV renderer overrides (`.camera` + `.render(pos, quat)`) |

#### Replay CLI

`python -m skyflow.viz.replay <flight.npz>` opens the viewer on a saved log:

| option | default | meaning |
|---|---|---|
| `path` | — | `flight.npz` written by `FlightLog.save` |
| `--pilot-cam WxH` | `256x192` | pilot-cam resolution |
| `--speed` | `1.0` | initial playback speed (`[` `]` at runtime) |
| `--start` | `0` | first logged row to show |
| `--headless` | off | SDL dummy driver (CI) |
| `--frames N` | — | auto-close after N frames |
| `--shot PATH` | — | screenshot saved when `--frames` ends |
| `--mp4 PATH` | — | export the whole log to mp4 instead of opening a window |
| `--mp4-fps` | `60.0` | mp4 playback fps; rows resample onto this clock (realtime playback) |

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
