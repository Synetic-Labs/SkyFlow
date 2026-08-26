# SkyFlow — design of record

Status: implementation contract for the v0.2 rewrite. Interfaces here are FROZEN for the
build; change requires editing this file first. Style: this document is normative — "must"
means must.

## 1. Identity

SkyFlow is a fleet-batched quadrotor **simulator** in pure JAX. It is only the sim:

- **Physics comes from `skyflow-dynamics[jax]`** (the generated backend). SkyFlow contains
  **zero handwritten continuous dynamics**. If a physics term is missing, it is added to
  SkyFlow-Dynamics first (INTAKE protocol) and consumed here — never implemented here.
- **No training code.** No RL algorithms, losses, optimizers, replay, obs normalization,
  policy networks, checkpoints, logging frameworks, or Hydra/omegaconf. Training repos
  import SkyFlow and own all of that.
- **Tasks are examples.** SkyFlow ships the Task protocol plus two reference tasks
  (`hover`, `figure_eight`). Real research tasks live in the
  downstream project and register against the protocol.
- Differentiability is **not claimed** in docs or API. The design avoids blockers
  (pure functions, no in-place host state on the motors path), and a `differentiable`
  flag exists but must raise `NotImplementedError("planned")` for now.

Boundary test for any new line of code: "does it decide what the vehicle does or senses?"
→ SkyFlow. "Does it decide how a policy learns?" → training repo. "Is it a force/torque/
sensor equation?" → SkyFlow-Dynamics.

## 2. Repo shape

```
pyproject.toml          # uv + hatchling, src layout, python >=3.12
src/skyflow/
  __init__.py           # re-exports: SkyFlowEnv, SimConfig, tasks.register_task, __version__
  types.py              # shared types: PlantState alias, SimState, StepInfo, ObsSpec, TaskEval, Task protocol
  dynamics.py           # THE ONLY module importing skyflow_dynamics: fleet-vmapped step/statedot/imu/throttle
  params.py             # airframe registry (spec SCHEMA rows + limits + command map), DR brackets + sampling
  sensors.py            # IMU packaging: generated imu_fn + noise/scale DR hooks (harness-side by spec charter)
  env.py                # SkyFlowEnv platform: substep scan, delay buffer, wind, pokes, ground, crash set, auto-reset
  tasks/
    __init__.py         # registry: register_task/build_task; registers hover + figure_eight
    base.py             # obs helpers (world_to_body, finalize_obs), NO protocol here (protocol lives in types.py)
    hover.py            # HoverTask
    gate_course.py      # GateCourseTask, registered as "figure_eight" (state or vision obs) + gate pass/progress reward
  vision/
    __init__.py
    camera.py           # CameraModel (intrinsics, mount, rays)
    gates.py            # GateSet + course builders: from_waypoints/line/circle/figure_eight + classify_crossings
    renderer.py         # analytic ray-cast coverage masks (gates, floor)
    mask_noise.py       # persistent mask corruption families
  firmware.py           # control="sticks" seam against cudaflight (§10)
  viz/                  # OPTIONAL viewer, extra "viz" (§13). Core never imports it.
    __init__.py         # lazy re-exports: Viewer, FlightLog, Scene + the five primitives
    palette.py          # shared colors (dark ground, mask-orange accent)
    primitives.py       # Scene + Grid/Path/Gate/Box/Marker — pure data + serde, no pygame
    frame.py            # ViewFrame: host-side numpy snapshot of the watched worlds
    projection.py       # iso/top/profile projection, fitted once from the scene AABB
    fpv.py              # mask+floor composite (pure numpy) + PilotCam pose re-renderer
    scenepane.py        # scene + drone-glyph drawing onto a pygame surface
    hud.py              # instrument strip drawing
    viewer.py           # the live pygame host: render thread, pacing, keys, panes, shots
    record.py           # FlightLog — pose logs → self-describing flight.npz
    replay.py           # python -m skyflow.viz.replay: scrub/replay host + mp4 export
examples/
  fly_hover.py          # tiny PD hover demo (examples are demos, not package code)
  fly_figure_eight.py   # benchmark env from a drone config (default) + course tracer (--trace)
  fly_teleop.py         # hand-fly control="sticks": keyboard / joystick / UDP + viewer
tests/                  # pytest; §11 lists the required suites
```

Dependencies: `jax>=0.11`, `numpy>=2`, `skyflow-dynamics[jax]` (uv source: the public git
URL). Extras: `cuda` → `jax[cuda13]`;
`firmware` → `cudaflight>=0.6.0` (public GitHub release-wheel URL source); `viz` → nothing but
`pygame-ce>=2.5` (§13; the maintained fork, imports as `pygame`). Dev group: `pytest>=9`,
`ruff>=0.16`. License MIT. Version 0.2.0.

## 3. Conventions (one frame inside)

- **World**: right-handed, ẑ up. Gravity -ẑ. **Body**: FLU (x forward, y left, z up).
  **Quaternion**: wxyz scalar-first, Hamilton, body→world. **Units**: SI, rad/s rotor
  speeds. Identical to SkyFlow-Dynamics — states pass through untranslated.
- NED/FRD exists in exactly two contained places: (a) inside `vision/` internals (the
  renderer's internal math; conversion at its public entry points), (b) at the firmware
  boundary (Betaflight wants FRD sensors). Public APIs speak z-up FLU only.
- dtype: float32 on every array the env creates. Precision of the dynamics follows the
  ambient JAX config (tests may enable x64 for tolerance checks of adapters only).
- Fleet axis: leading `[F, ...]` on every batched array. The env is natively batched;
  `dynamics.py` vmaps the single-vehicle backend functions once, centrally.

## 4. types.py (frozen)

```python
class ObsTerm(NamedTuple): name: str; dim: int; units: str = ""  # units+frame string; "" keeps 2-field call sites valid
class ObsSpec(tuple[ObsTerm, ...]):        # .dim (sum), .layout (name→slice)
class DRState(NamedTuple):   # per-episode DomainRand trait draws (§7); new traits go HERE, never as new SimState leaves
    wind_mean: Array         # [F,3] f32 steady wind velocity, world frame, z = 0
    imu_bias: Array          # [F,6] f32 additive IMU bias: accel(3) m/s², gyro(3) rad/s
class TaskEval(NamedTuple):
    reward: Array            # [F] f32
    success: Array           # [F] bool
    crash: Array             # [F] bool — task-specific fatal condition
    info: dict[str, Array]   # scalarizable diagnostics, all [F]
    task_state: Any
class StepInfo(TypedDict):   # info returned by env.step
    terminated: Array; truncated: Array; final_obs: Array; ...task info merged
class Task(Protocol):
    obs_spec: ObsSpec
    image_shape: tuple[int, int, int] | None   # (H, W, C) when vision obs present, else None
    success_terminates: bool
    def spawn(self, key, n, params) -> tuple[Array, Any]            # plant [n,17], task_state
    def observe(self, plant, task_state, imu, last_action, key,
                fresh_spawn: bool) -> tuple[Array, Any]             # obs [n,obs_dim] f32
    def evaluate(self, prev_plant, plant, task_state) -> TaskEval   # reward on the transition
    def metrics(self, task_state) -> dict[str, Array]
```

`SimState` (registered pytree dataclass, all leaves `[F, ...]` unless noted):

```python
plant [F,17] f32      # spec layout: x(3) v(3) q_wxyz(4) ω(3) Ω(4) rad/s
params [F,P] f32      # per-world randomized flat spec params (pack_params order)
key                   # jax PRNG key (env-owned; split every step)
wind_vel [F,3]        # OU gust deviation state (world frame), zero-mean; every consumer
                      # sees the total wind dr_state.wind_mean + wind_vel
dr_state: DRState     # per-episode trait draws (§7), redrawn at auto-reset respawn
act_buf [F,D+1,4]     # transport-delay ring, newest first
delay_idx [F] int32   # per-world delay draw
last_action [F,4]
steps [F] int32; airborne [F] bool
ep_return [F]; ep_len [F] int32
crash_frac 0-d; success_frac 0-d; trunc_frac 0-d   # f32 outcome-fraction EMAs over completed episodes (§7 step 10)
ep_return_ema 0-d; ep_len_ema 0-d                  # f32 completed-episode return/length EMAs (§7 step 10)
task_state: Any       # opaque task pytree
```

## 5. dynamics.py — the only physics importer

Wraps `skyflow_dynamics.backends.jax` (aliased `sfd`). Provides, all fleet-batched via one
central `jax.vmap`:

```python
N_ROTORS = 4; STATE_DIM = 17
substep(plant [F,17], omega_cmd [F,4], wind_vel [F,3], f_ext [F,3], tau_ext [F,3],
        params [F,P], dt) -> [F,17]        # sfd.rk4_step_fn + sfd.post_step (renorm+clip via limits passed in)
statedot(...) -> [F,17]                    # for diagnostics
imu(plant, omega_cmd, wind_vel, params) -> (accel [F,3], gyro [F,3])   # sfd.imu_fn, identity mount
throttle_to_omega(u [F,4] in [0,1], w_min, w_max, k) -> [F,4]          # sfd.throttle_to_speed_fn
pack_params / param_slices re-exported from sfd
```

Motor model fixed to `first_order` for v0.2 (asymmetric available behind a config field,
same backend). Inputs assembly (`Ω_c, v_wind, F_ext, τ_ext`) happens here, matching the
backend's input layout — no other module touches the flat layouts.

## 6. params.py

```python
@dataclass(frozen=True)
class Airframe:
    name: str
    values: dict            # spec SCHEMA row (validated by skyflow_dynamics pack_params)
    rotor_speed_min: float; rotor_speed_max: float
    throttle_k: float       # throttle-curve blend for the command map
AIRFRAMES = {"crazyflie": ...}   # from skyflow_dynamics.spec.parameters.CRAZYFLIE (+limits)
register_airframe(name, airframe)
```

Domain randomization: `sample_params(key, airframe, fleet, scale, brackets=None) -> [F,P]`
— multiplicative log-uniform-style jitter `1 + scale*U(-b, b)` per stored entry with a
bracket table `DR_BRACKETS = {"mass": 0.10, "inertia": 0.15, "ct0/1/2": 0.15,
"cq0/1/2": 0.15, "tau_m": 0.20, "k_d": 0.30, "k_z": 0.30, "r_prop": 0.0, ...}`. Per-rotor
entries (ct*/cq*, rotor geometry) draw independently, so motor-to-motor asymmetry is
covered. `brackets` overrides individual half-widths per key — a measured airframe
shrinks the brackets on its identified parameters; unknown or structural keys fail
loudly. `max_bracket(brackets)` reports the largest effective half-width (the env
validates `body_scale·max_bracket < 1` so factors stay positive). Keys NEVER jittered:
`spin`, `axis`, `grav` (masked via `param_slices`). Zero-valued nominals stay zero
(multiplicative). Same routine used at reset and auto-reset respawn, driven by the §7
DomainRand block: `scale = dr.scale·dr.body_scale`.

## 7. env.py — the platform

```python
@dataclass(frozen=True)
class DomainRand:                        # ALL training-robustness randomization; one instance = one setting
    scale: float = 1.0                   # master dial over every continuous magnitude
    # body — the vehicle (trait, §6 sample_params)
    body_scale: float = 1.0              # multiplies the bracket half-widths / factor limits
    brackets: dict | None = None         # per-key half-width overrides (§6); base table is
                                         # DR_BRACKETS, or RESIDUAL_BRACKETS once factors is on
    factors: dict | None = None          # correlated factor stage (§6): None = off (legacy
                                         # independent draws, bit-exact); {} = FACTOR_LIMITS
    # world — wind and shocks
    wind_mean_mps: float = 0.0           # trait: steady horizontal wind, magnitude ceiling, m/s
    wind_gust_mps: float = 0.0           # process: OU gust stationary std per axis, m/s
    wind_tau_s: float = 0.5              # OU correlation time, s (a clock — never scaled)
    poke_prob: float = 0.0               # event rate per control step per world (never scaled)
    poke_force_n: float = 0.0; poke_torque_nm: float = 0.0   # magnitude ceilings
    # actuation — command transport
    delay_steps: tuple[int, int] = (0, 0)  # trait: (min, max) control steps (never scaled)
    cmd_drop_prob: float = 0.0           # event: dropped packet holds the previous APPLIED
                                         # command (the RX's ZOH; never scaled)
    battery_sag: float = 0.0             # trait: rotor-speed-ceiling factor 1 - U(0, sag)
    # sensing — the IMU/baro rows the firmware (§10) and IMU-observing tasks consume
    gyro_noise_rps: float = 0.0; accel_noise_mps2: float = 0.0  # process: white, per 1 kHz sample
    gyro_bias_rps: float = 0.0; accel_bias_mps2: float = 0.0    # trait: constant per-axis half-width
    baro_noise_pa: float = 0.0           # process: white on the sticks baro row
    # observation — applied by the ENV after task.observe (tasks own semantics, the env owns corruption)
    obs_noise: float = 0.0               # LEGACY stress knob: unit-blind uniform half-width
                                         # (mask-valued terms excluded); prefer obs_error
    obs_error: dict | None = None        # L5 estimator-error model (errors.py; ERRORS.md)
    # initial state — task variety, not model error (never scaled)
    spawn_scale: float = 1.0             # forwarded to task builders that name spawn_dr_scale
    def off(self): ...                   # scale=0, delay=(0,0), cmd_drop_prob=0,
                                         # obs_error=None: bit-exact nominal (§11 test)
    def effective(self): ...             # scale folded into every continuous magnitude

@dataclass(frozen=True)
class SimConfig:
    num_envs: int = 1024
    task: str = "hover"; task_kwargs: dict = field(default_factory=dict)
    airframe: str = "crazyflie"
    control: str = "motors"            # "motors" | "sticks" (§10)
    firmware: str = "auto"             # sticks backend: "auto" | "cpu" | "gpu" (§10)
    control_hz: float = 100.0          # physics fixed at physics_hz
    physics_hz: float = 1000.0         # sticks mode requires exactly 1000 (§10: 1 kHz firmware tick)
    differentiable: bool = False       # raises NotImplementedError("planned") if True
    dr: DomainRand = DomainRand()      # ALL randomization / disturbance — nothing else randomizes
    eeprom: str | None = None          # sticks boot config: CLI `dump all` path, rendered
    eeprom_overrides: str | None = None  # at construction (§10); motors mode rejects both
    # episode / safety
    max_episode_steps: int = 1000; stuck_steps: int = 200
    bounds_xy_m: float = 20.0; bounds_z_m: float = 8.0
    max_speed_mps: float = 30.0; max_rate_rps: float = 50.0
    ground_tilt_limit_rad: float = pi/3
```

**The DomainRand contract.** Robustness training = the sim world deliberately differing
from the nominal model, in the ways the real world differs. Rules:

- Randomness enters ONLY at loop boundaries — the params row (§6), exogenous forces, the
  delay ring, the sensor rows (§2 sensors.py), the observation vector. The ODE and the
  firmware stay exact: the controller you will fly is never randomized, its inputs are.
- Three draw classes: **traits** (once per episode at reset/respawn, constant within it —
  params row, steady wind, IMU bias, delay), **processes** (fresh every sample — gusts,
  sensor noise, obs noise) and **events** (Bernoulli per control step — pokes). Trait
  storage is the single `SimState.dr_state` leaf (§4).
- Every knob is a magnitude in physical units. `scale` multiplies every continuous
  magnitude and never touches clocks, event rates, integer delays or spawn spread;
  `scale=0` with `delay_steps=(0,0)` (= `off()`) is bit-exact nominal — a §11 invariant.
  Defaults reproduce a plain config: body DR on, everything else off.
- One object = one robustness setting: a curriculum ramps `scale`; an outer controller
  (learned DR later) emits a `DomainRand` per training phase — a phase change recompiles
  once. Per-world adaptive levels, if ever needed, move into `DRState` as traced arrays;
  the schema stays.
- Defaults are honest: a knob exists only if it does something (no dead fields).

`SkyFlowEnv(cfg)` exposes `fleet`, `obs_spec`, `obs_dim`, `act_dim=4`, `image_shape`,
`decimation` (=round(physics_hz/control_hz)), `dt_control`, and:

```python
reset(key) -> (obs [F,obs_dim] f32, SimState)
step(state, action [F,4] in [-1,1]) -> (obs, state', reward [F], done [F], info: StepInfo)
metrics(state) -> dict[str, Array]   # scalar means: outcome fractions, ep stats
task_state(state) -> Any             # the task's OWN pytree (unwraps the §10 sticks carry)
```

Pure functions; the caller jits. **Step pipeline (order is normative):**

1. Split `state.key`. Push `action` into `act_buf`; read delayed action per `delay_idx`.
2. Command map: motors mode → `u = (a+1)/2`, `Ω_c = throttle_to_omega(u, ...)`.
3. Advance the OU gust deviation (exact discretization: decay `exp(-dt/τ)` + kick).
   Every consumer sees the total wind `dr_state.wind_mean + wind_vel`.
4. Poke sampling: with prob `dr.poke_prob` per control step draw world-frame `F_ext` and
   body `τ_ext` (uniform ball · magnitudes); else zeros. These pass through the backend's
   exogenous inputs — never write velocity state directly.
5. `lax.scan` over `decimation` substeps: `plant = dynamics.substep(...)` (ZOH on Ω_c,
   wind, F_ext, τ_ext) then ground contact (§8). Sticks mode corrupts each substep's
   sensor rows per DomainRand (bias + fresh per-sample noise) BEFORE the firmware tick —
   the firmware filters what the real one filters (§10).
6. Airborne latch; crash set: flyaway (|x|,|y| > bounds_xy, z > bounds_z, speed >
   max_speed, rate > max_rate), ground crash (z < 0.05 and airborne and (descent > 1 m/s or
   tilt > limit)). Task `evaluate` on (prev_plant, plant): reward, success, task crash.
7. `terminated = crash or task_crash or (success and success_terminates)`;
   `truncated = steps ≥ max_episode_steps or stuck`; `done = terminated or truncated`.
8. Observe: `imu = sensors.measure(...)` with DomainRand bias/noise;
   `obs, task_state = task.observe(...)`; then the env adds `dr.obs_noise`.
9. Auto-reset in-jit: for done worlds — fresh spawn (task), fresh DomainRand draws
   (params §6, traits `dr_state`, delay), cleared buffers/gust, re-observe with
   `fresh_spawn=True`; blend the per-world state leaves with
   `tree_where(done, reset_leaf, leaf)`. Two exclusions, by design: the step-10 EMA
   leaves (cross-episode, fleet-global) pass through unblended, and the sticks
   firmware pair is restored whole through `FirmwareFleet.reset(mask)` — the fleet
   masks internally. Pre-reset obs goes to `info["final_obs"]`;
   `info["terminated"]/["truncated"]` are the pre-reset flags.
10. Episode bookkeeping EMAs for `metrics`: on the pre-reset done rows, update the
    SimState EMA leaves (§4) — outcome fractions (crash / success-at-end / pure
    truncation) and completed-episode return/length — with the done-row means, decayed
    `0.99` per completed episode (`alpha = 0.99^n_done`; no-op when nothing finished).
    EMAs start at 0 after `reset` and warm up from there.

### 7.1 DR coverage matrix (the completeness rule)

Every input that crosses into the control loop gets one row: randomized, fitted
constant, declared exact, or roadmap. "Declared exact" is an allowed answer; silence is
not. Adding a new input to `env.step` obliges a new row.

| Real-world variation | Knob | Status |
|---|---|---|
| mass, inertia, thrust/torque curves, motor τ, aero terms | `body_scale` × brackets (§6) | randomized (default on) |
| motor-to-motor asymmetry (worn/weak motor) | per-rotor ct*/cq* entries draw independently (§6) | randomized |
| per-motor response spread | — (`tau_m` is one scalar) | roadmap (spec change via Dynamics INTAKE) |
| steady wind | `wind_mean_mps` | randomized |
| gusts | `wind_gust_mps`, `wind_tau_s` | randomized |
| turbulence spectrum (Dryden / von Kármán) | — | roadmap |
| collisions, bumps, prop wash from others | `poke_prob/force/torque` | randomized (crude, by design) |
| command transport delay | `delay_steps` | randomized |
| dropped command packets (link) | `cmd_drop_prob` (hold-last, the RX's ZOH) | randomized |
| firmware code + tune | none | declared exact — the §10 point; inputs get randomized, never the controller |
| IMU white noise | `gyro_noise_rps`, `accel_noise_mps2` | randomized |
| IMU constant bias | `gyro_bias_rps`, `accel_bias_mps2` | randomized (firmware gyro-cal absorbs gyro bias at boot, like hardware) |
| IMU bias walk, staleness, scale/misalignment | — | roadmap |
| RPM-tracked vibration harmonics + RPM filter | — | roadmap (researched, parked in ROADMAP.md) |
| baro noise | `baro_noise_pa` | randomized (sticks) |
| battery voltage sag | `battery_sag` (per-episode rotor-speed-ceiling draw) | randomized |
| state-estimate error at the policy | `obs_error` (L5 model, ERRORS.md); `obs_noise` legacy | randomized (env-applied) |
| initial state spread | `spawn_scale` (+ task spawn kwargs) | randomized — task variety, outside `scale` |
| gravity, geometry, spin signs, thrust axes | NEVER_JITTER (§6) | declared exact |

Sensing knobs bite where sensors are consumed: always in sticks mode (the firmware
reads the rows every substep); in motors mode only for tasks that observe the IMU
(hover observes exact state).

## 8. Ground contact (harness bookkeeping, not physics)

At each substep, where `z ≤ 0`: clamp `z = 0`, clamp `v_z = max(v_z, 0)`, zero in-plane
velocity and body rates, hold quaternion. Documented as the registry's
`ground_contact_heuristic` (candidate, harness) — replaceable by a real contact model via
the spec later.

## 9. Tasks

**hover** — spawn on ground pad (motors near idle) with jittered XY; goal setpoint drawn
in a box, resampled every `goal_hold_s`; obs = `[rel_pos(3), vel(3), rot_matrix(9),
last_action(4)]` = 19 (observation corruption is env-side: §7 `DomainRand.obs_noise` —
tasks own semantics, the env owns corruption); reward per control step:
`w_pos·exp(-3·d) + w_hold·exp(-50·d) - w_vel·|v| - w_rate·|ω| + progress(d_prev² - d²)`;
success `d < 0.1 m` (does not terminate); task crash: leaving the safe box.

**figure_eight** — the registered name of the generic `GateCourseTask`; course = `GateSet`
from `vision/gates.py`, defaulting to the shipped `figure_eight(...)` builder: two
ellipse lobes (semi-axes `lobe_radius_m` x `lobe_half_width_m`) tangent at a gate-free
crossover, 2·k gates tangent-yawed at the ellipse angles φ = j·2π/(k+1) per lobe,
alternating crossing directions at the center. The 6-gate default is the nav-jax
FigureEight map EXACTLY (z-up from its NED rows, nav-jax tests/test_gate_spawn.py):
shoulders (±5, ∓3) m flown straight along ±x, apexes (±10, 0) flown along +y, 1.5 m
altitude, 20 x 6 m footprint; gate 1 is the right lobe's lower shoulder. Spawn: podium pad ON
THE GROUND (z = 0, resting, airborne latch cold — arm at idle, spool, lift; a raised pad
free-falls during spool-up and trips the §7 ground-impact terminal) facing gate 1 — the
shipped course pads the centre of the cluster OPPOSITE gate 1 (the canonical start;
`podium_pos_m` / `podium_height_m` override, custom courses default behind gate 1) — or
spread across gates (curriculum knob). Active-gate
progress: `r = w_prog·(d_prev - d)` toward the pre-gate point + pass credit
`w_gate·centering` on crossing (from `classify_crossings`; miss/frame-hit = task crash),
minus rate penalty. Obs (state mode): `[gate_rel(3), gate_normal(3), next_gate_rel(3),
vel_body(3), rot_matrix(9), last_action(4)]`; vision mode replaces the gate blocks with
the rendered mask `[H,W,1]` (+ same proprio tail). Success = last gate passed (terminates).

Reward constants live in task kwargs with the shipped defaults; no reward code in env.py.

## 10. firmware.py — control="sticks"

cudaflight facts (Python API additive through v0.3.4): package
`cudaflight`, no core deps; sensor input per 1 kHz tick = f32 [F,7] NED/FRD
`[gyro_FRD rad/s (3), specific force FRD m/s² (3), baro Pa (1)]` (level hover ⇒
az = -9.81); sticks f32 [F,4] AETR in [-1,1]; output motors f32 [F,4] in [0,1] QUADX
order + armed u8 [F]; GPU fleet needs n ≥ 3 and, since v0.3.3, the wheel ships its
in-jit FFI half — the `cudaflight.xla` module + prebuilt XLA handlers
(`_data/libcudaflight_xla.so`, source fallback compiled on demand). CPU SITL
(`libcpuflight.so`, ctypes + `io_callback(ordered=True)`) is self-contained, works for
any fleet size, jits but is not vmappable/replayable.

SkyFlow therefore ships:
- `types.FirmwareFleet` protocol: `act_dim`, `fresh_firmware_state() -> (blob, fwstate)`,
  `fw_step(blob, fwstate, sticks [F,4], sensors [F,7]) -> (blob, fwstate, motors [F,4],
  armed [F] u8)`, `reset(blob, fwstate, mask u8[F]) -> (blob, fwstate)`, `close()`.
- `firmware.CpuFirmwareFleet` — full implementation via ctypes on the cudaflight wheel
  (`cudaflight.lib.load_cpu`), `io_callback(ordered=True)`, zero-length blob placeholders.
- `firmware.GpuFirmwareFleet` — full implementation via `cudaflight.xla` (>= 0.3.3):
  in-jit pure step/reset custom calls, firmware state GENUINELY value-threaded as
  donated (blob, fwstate) uint8 buffers copied from the armed-on-ground snapshot.
  With cudaflight >= 0.3.4 the snapshot ALSO rides as read-only JAX buffer arguments
  into the reset call and the library-side snapshot copies are freed after
  construction — every firmware DATUM then lives in XLA buffers (checkpointable,
  swappable); the library keeps only kernels, context, and launch configuration.
  On a 0.3.3 wheel the reset falls back to the library-side snapshot pointers.
  `fleet >= 3`; one handle = one device = `fleet` worlds; requires
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` before jax touches the GPU — the LAUNCHER
  exports it; the library never mutates the environment. The constructor touches the
  target device first so XLA claims the primary CUDA context.
- Backend selection (`SimConfig.firmware`): `"cpu"` / `"gpu"` force a backend and fail
  loudly; `"auto"` (default) picks GPU when `fleet >= 3` and a CUDA device is visible,
  falling back to CPU with a `warnings.warn` when GPU construction fails.
- Injection: `SkyFlowEnv(cfg, firmware_fleet=...)` accepts any FirmwareFleet and
  overrides `cfg.firmware`.
- Boot config (`SimConfig.eeprom` / `eeprom_overrides`): `eeprom` is a path to a drone's
  Betaflight CLI `dump all` text — the config source of truth (examples/configs/README.md).
  `_build_fleet` renders it at construction via `cudaflight.render_eeprom` (version-gated
  strict round-trip on one throwaway CPU boot, BEFORE the fleet exists — the CPU library
  allows one live fleet per process) and hands the image to either backend.
  `eeprom_overrides` is an optional file of sim-only CLI lines appended after the dump
  (e.g. `blackbox_device = NONE`). A stale or foreign dump fails at construction — never a
  silent factory reset to stock defaults. The rendered temp-file path lands on
  `env.eeprom_image` for run logs. Fail-loud guards: overrides without a dump, eeprom in
  motors mode, eeprom plus an injected `firmware_fleet` (that fleet already booted its own
  config), missing files. `eeprom=None` boots the wheel's stock defaults. The dump header
  also SELECTS the firmware base (cudaflight >= 0.6.0): a dump built from another
  Betaflight base than the installed wheel picks its (libcpuflight.so, fw.fatbin) pair
  from the cudaflight bundle cache — fetched once with `python -m cudaflight.bases <rev>`;
  a cache miss fails at construction naming that command. A norevision dump or the
  installed base uses the wheel's embedded binaries. Provenance: `env.firmware_base`.

Sticks substep order (normative): synth sensors
(FLU→FRD flip; baro from z-up altitude, isothermal 101325 Pa / 8434 m — a harness-side
sensor model, documented as such) →
`fw_step` → motors [0,1] reordered by `motor_perm` → motor duties feed the throttle map
(no board-align rotation exists: a dump with `align_board_*` set, or with
`yaw_motors_reversed = ON`, is REJECTED at construction — the packaging cannot honor it)
as u (ZOH for that 1 ms substep). `control="sticks"` requires `physics_hz == 1000`: each
physics substep pairs one plant step with exactly one 1 kHz firmware tick (the firmware's
virtual clock advances 1 ms per tick, unconditionally), so any other rate would silently
skew firmware time against simulated time — rejected with ValueError at construction.
`control="sticks"` without a fleet instance and without
cudaflight importable → ImportError with install guidance at construction.
`differentiable=True` + sticks → raise. Motors mode never touches this module.

## 11. Required tests (all CPU, deterministic keys)

- `test_dynamics_adapter.py` — SkyFlow substep ≡ backend `rk4_step_fn`+`post_step` on
  random fleets (x64, tol 1e-9); throttle map endpoints; imu hover = (0,0,g).
- `test_env_contract.py` — shapes/dtypes/finiteness; determinism (same key → identical
  rollout); done semantics; final_obs correctness (pre-reset obs); auto-reset isolation
  (non-done worlds bit-identical); delay buffer (action takes effect exactly k steps
  late, k=delay); ZOH (constant action ⇒ constant Ω_c across substeps).
- `test_disturbances.py` — OU gust stationary std ≈ configured; poke rate ≈ poke_prob;
  wind actually enters aerodynamics (drag response differs with wind on a vehicle with
  c_D > 0), zeros when disabled.
- `test_domainrand.py` — off()/scale=0 are bit-exact nominal (params, traits, rollout);
  effective() folds the master scale into magnitudes only; bracket overrides pin keys
  and reject unknown/structural keys; body positivity guard; steady-wind trait is
  horizontal, bounded, reaches the aerodynamics, and is redrawn at respawn (with IMU
  bias); sensors.measure bias/noise hooks; obs noise is env-applied and bounded; sticks
  mode — corrupted rows reach the firmware boundary (recorder fleet): exact ⇒ zero gyro
  rows on the pad, noise varies per sample, bias holds within an episode, baro noise
  varies.
- `test_ground.py` — no penetration; resting vehicle stays; spawned-on-pad hover task
  takes off under full throttle.
- `test_task_hover.py` / `test_task_gate.py` — hover: spawn/obs/reward shapes, reward increases as distance falls;
  figure_eight (GateCourseTask): scripted straight-line fly-through registers pass exactly once, centering
  ∈ (0,1]; miss trajectory → crash; figure-eight builder: 2k gates, closed loop, gate
  normals alternate through the crossover.
- `test_vision.py` — camera ray invariants (center pixel = optical axis, FOV edges);
  gate dead-ahead → mask centroid ≈ image center; behind camera → zero mask;
  mask ∈ [0,1]; mask_noise: output ∈ [0,1], persistence across held frames,
  disabled ⇒ identity.
- `test_firmware.py` — skipped unless cudaflight importable; arm→spin-up→hover smoke.
  GPU-fleet twin marked `gpu` (skipped without a CUDA device): same smoke through
  `GpuFirmwareFleet`, plus snapshot restore determinism and the `firmware="auto"` pick.
- `test_eeprom.py` — own module (test_firmware.py keeps a module-scoped fleet open, and
  the render needs a fleet-free process): construction-time guards (overrides without a
  dump, eeprom in motors mode, eeprom + injected fleet, missing file); stock dump renders,
  boots, and climbs (`env.eeprom_image` set); version gate rejects a foreign-release dump;
  base resolution (installed base / norevision → embedded binaries, foreign base → bundle
  cache or a construction error naming the fetch command).
- `test_jit.py` — one jitted 50-step rollout, no NaN; second call does not retrace
  (jit cache check); vision task jitted rollout smoke.
- `test_viz.py` — pygame-free half: primitive serde round-trip (bind is live-only);
  projection invariants (z-up maps screen-up, AABB fits the rect); FPV composite colors
  and dtype; FlightLog npz round-trip; core-must-not-import-viz import scan.
- `test_viz_panes.py` — pygame half (skipped unless pygame importable; SDL dummy video
  driver): scene/HUD builders draw onto a plain surface; headless Viewer smoke over a real
  hover rollout with a screenshot; headless replay smoke from a saved flight.npz.
- `test_registry.py` — task/airframe registry contracts (unknown names, double register).
- `test_domainrand_factors.py` — correlated factor stage (§6): limits, guards, structure.
- `test_battery_sag.py` — sag trait draw, thrust-to-weight guard re-run, off ⇒ bit-exact.
- `test_delay_link.py` / `test_obs_error.py` — L3 link (drops) and L5 estimator error
  (ERRORS.md): off-knobs bit-exact, on-knobs shaped and bounded.
- `test_lifecycle.py` — close()/context manager; one-CPU-fleet guard; use-after-close
  raises; eeprom image reaping.
- `test_task_carry.py` — SimState.task_state raises on the sticks carry; the accessor
  unwraps in both modes; sticks step rejects a bare task pytree.
- `test_config_guards.py` — negative construction tests (num_envs, firmware value,
  negative DR magnitudes, act_dim mismatch, motor_perm).
- `test_sticks_axis.py` — the production axis: sticks + gate task on the real CPU SITL,
  multi-step jitted with auto-reset; FlightLog bind round-trip; snapshot through the
  accessor (and the raw-carry read raising); both-control-modes contract fixture.

## 12. Deferred (roadmap, do not build now)

The roadmap lives in ROADMAP.md: a backlog list plus one section per researched idea,
with the evidence kept. Nothing there is part of the design until it is promoted into
this document.

## 13. viz — the optional viewer (extra "viz")

Everything under `src/skyflow/viz/`, installed by the `viz` extra (pygame, nothing else).
Boundary: viz SHOWS what the vehicle did and sensed — it never decides it. Normative rules:

- **No core module imports `skyflow.viz`** (test-enforced). Display-only geometry — the
  floor composite behind the mask, the pilot camera, scene props — never enters an
  observation.
- **The policy FPV pane shows the observation verbatim**: the mask block sliced from the
  obs vector the policy received, corruption included. A fresh render there would flatter
  the policy.
- **Builders draw, hosts own windows.** Panel builders are windowless (surface/array in,
  pixels out); the live viewer, the replay host and any export share them, so all hosts
  look identical. The live host draws on a background RENDER THREAD that owns the window
  (latest-wins mailbox; feeding calls only snapshot watch rows, so the sim loop never
  pays draw time); replay and export run synchronous (`threaded=False`) for frame-exact
  draws.

Four layers, strict about what each may know:

1. **Vehicle truth — always drawn.** Pose, attitude, rotor speeds, trails, glyphs, the
   fixed HUD instruments (sticks/action, motors, attitude, heading, speed/altitude). All from
   `plant` and the step outputs. No task knowledge.
2. **World geometry — always data.** Scene primitives. Tasks and users contribute them;
   the pane only draws. Extension is public: `register_primitive(cls, draw_fn)` — the
   same registry idiom as `register_task`/`register_airframe`.
3. **Sensor truth — follows `vision/`.** The pilot cam renders what the sim camera
   defines; the policy pane shows the obs image block verbatim (`image_term` names the
   block, default "mask"). Viz never invents a sensor.
4. **Channels — named scalars, caller-selected.** Anything from step returns, `info` or
   `metrics` can be traced (`viewer.frame(..., channels={...})`); the HUD draws one graph
   per name. Reward is a channel, not a viz concept.

Conventions degrade to nothing: `viz_scene()`, `camera`, `gates` are optional duck-typed
hooks; a task without them still gets layers 1, 3 and 4.

Scene model: the scene pane draws a `Scene` — a flat list of primitive dataclasses
(shipped: `Grid`, `Path`, `Gate`, `Box`, `Marker`), each one dataclass plus one registered
draw function, JSON round-trippable. `bind` (a dotted ViewFrame attribute path, or any
callable of the ViewFrame) makes a primitive track live state; the PRIMITIVE interprets
the bound value (`_apply_bind`): Marker/Box move their centre, Path replaces its points,
Gate reads an active index and turns accent on matching its own `index` — so the gate
task declares `bind="task_state.active_gate"` itself and no task field names appear in
generic code. Callable binds are live-only and drop from serialization. Drone glyphs come
from `plant`, never from primitives. Tasks contribute defaults through the OPTIONAL
duck-typed hook `viz_scene() -> list[dict]` (serde-form dicts, so tasks stay
viz-import-free); `hover` and `figure_eight` ship hooks.

FlightLog: pose logs, never pixels — the camera is a pure function of pose, so poses
reproduce any view at any resolution forever. `flight.npz` is self-describing (JSON
header: serialized scene, camera, gate geometry, airframe, config fields; arrays: plant,
action, done, channel traces `ch:<name>`, and `bind:<path>` values for the scene's string
binds — replay resolves the same binds and plots the same channels with no task code, all
`[T, W, ...]`). Live capture slices watch rows per control step; training capture takes
whole `[T, W, ...]` chunks (`extend`) so the fused scan is never stalled. Replay
(`python -m skyflow.viz.replay`) re-renders FPV from poses; mp4 export through imageio
when importable (soft dep, like matplotlib in the examples).

Teleop is an example, not package code: `examples/fly_teleop.py` drives control="sticks"
through the CPU firmware fleet, paced to the wall clock, sticks from the keyboard, a local
pygame joystick, or UDP datagrams (20-byte little-endian `<ffffI` = roll, pitch, yaw,
throttle in [-1,1] + button bitmask, latest-wins, any sender).
