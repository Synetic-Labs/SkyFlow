# SkyFlow

A differentiable, fleet-batched quadrotor simulator in pure JAX — built for training
reinforcement-learning flight policies fast enough to iterate on.

The whole rollout — physics, sensors, camera, reward — is one `lax.scan` on the
accelerator. Thousands of drones step together, nothing leaves the device, and the entire
step is differentiable end-to-end, so the same environment serves PPO, SAC, and
analytic-policy-gradient / BPTT methods without a second implementation.

```python
import jax
from skyflow.env import SkyFlowEnv

env = SkyFlowEnv(num_envs=4096, task="hover", control="motors", control_hz=90.0)

key = jax.random.key(0)
obs, state = env.jax_reset(key)

action = jax.numpy.zeros((env.fleet, env.act_dim))
obs, state, reward, done, info = jax.jit(env.jax_step)(state, action)
```

## What it is

**An analytic plant, not a game engine.** The dynamics are a 46-coefficient rigid-body
quadrotor model — thrust curve with motor lag, body drag, axial inflow, per-motor
roll/pitch/yaw torque, rate damping, gyroscopic coupling — integrated with RK4 at 1 kHz
under a policy stepping at 90 Hz. There is no renderer in the loop and no physics engine
to marshal state in and out of, which is what makes the whole thing traceable by JAX.

**Coefficients that came from a real drone.** The bundled `air75_ii_racer` airframe is a
BETAFPV Air75 II Racer with every coefficient system-identified from Vicon motion-capture
flight, not guessed from a datasheet. `PlantParams` has no field defaults on purpose: an
airframe states all of its numbers or it does not exist, so nothing silently inherits
another drone's physics.

**Differentiable.** `differentiable=True` (with `control="motors"`) makes the full substep
rollout a gradient path, for short-horizon BPTT and analytic policy gradients. Vision mode
is *visual BPTT*: the mask is input-only for gradients — the coverage render is piecewise
constant in pose and explicitly stop-gradiented — so credit flows through the dynamics and
the proprioceptive tail while the CNN still trains through the policy pathway.

**Vision without a rasterizer.** Gates are rendered analytically to a coverage mask
directly from pose, batched over the fleet, with a camera model, configurable rate
(a 30 Hz camera under a 90 Hz control loop), and persistent mask-noise domain
randomization. It is a segmentation mask, not RGB — which is exactly the input a
perception front-end hands a policy at deploy time.

**Built to transfer, not just to score.** Domain randomization over every plant
coefficient, sample-and-hold sensor staleness, transport latency, OU wind, and spawn
jitter are all first-class knobs rather than afterthoughts.

## Install

```bash
pip install -e .            # CPU
pip install -e '.[cuda]'    # CUDA 13 wheels for JAX
```

Requires Python 3.12+. The only runtime dependencies are `jax` and `numpy`.

## Tasks

The environment is the *platform* — plant, randomization, disturbances, latency, the
rollout scan, the generic crash set, and the in-jit auto-reset. A **task** is the
*objective*: spawn distribution, observation, reward, and task-specific terminals. One
platform, many tasks.

`hover` ships with the package. Add your own against the `Task` protocol in
`skyflow.tasks.base` and register it:

```python
from skyflow.tasks import register_task
register_task("my_task", MyTask)

env = SkyFlowEnv(num_envs=1024, task="my_task", control="motors")
```

The env only ever reaches a task through the protocol, so a registered task is a
first-class citizen — nothing special-cases the built-in. Registering an existing name
replaces it, which is how you override `hover` without forking.

Course geometry for gate-based tasks lives in `skyflow.render.courses`: `from_waypoints`
takes `[north, east, alt, yaw_deg]` rows so a course is config data, and `line`/`circle`
generate the standard shapes.

## Scope

Two things are deliberately **not** in this distribution:

- **Betaflight-in-the-loop control.** `control="motors"` (direct per-motor thrust) is pure
  JAX and is what ships. The stick and rate control modes close the loop through real
  Betaflight firmware compiled to an XLA custom call, which needs a prebuilt `cudaflight`
  wheel distributed separately. Requesting those modes raises with an explanation.
- **Research tasks.** Gate-racing objectives with filter-in-the-loop observations,
  state-estimator overlays, and competition-specific course formats live downstream in the
  projects that use them. The `Task` protocol is the seam.

## Credits

SkyFlow stands on work that came before it:

- **[crazyflow](https://github.com/utiasDSL/crazyflow)** — the JAX-first, fleet-batched
  quadrotor-RL environment design. The NED conventions, the sensor-synthesis seam, and the
  general shape of a firmware-in-the-loop JAX rollout follow its lead.
- **[RotorPy](https://github.com/spencerfolk/rotorpy)** — a reference for aerodynamic
  modelling beyond simple thrust-and-torque: the drag, inflow and wake terms that make a
  quadrotor model hold up at speed rather than only near hover.
- **SkyDreamer** — the analytic quadrotor model this plant implements, from
  *"SkyDreamer: Interpretable End-to-End Vision-Based Drone Racing with Model-Based
  Reinforcement Learning"* (Diermayr et al., 2025, [arXiv:2510.14783](https://arxiv.org/abs/2510.14783)).
  The plant structure and its Table II reference coefficients are the published starting
  point that the bundled airframe re-fits against real flight.

Any errors in the adaptation are ours, not theirs.

## License

MIT — see [LICENSE](LICENSE).
