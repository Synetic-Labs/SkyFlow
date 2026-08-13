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
  (`hover`, `gate_course` with a figure-eight course). Real research tasks live in the
  consuming repo and register against the protocol.
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
    __init__.py         # registry: register_task/build_task; registers hover + gate_course
    base.py             # obs helpers (world_to_body, finalize_obs), NO protocol here (protocol lives in types.py)
    hover.py            # HoverTask
    gate_course.py      # GateCourseTask (state obs or vision obs) + gate pass/progress reward
  vision/
    __init__.py
    camera.py           # CameraModel (intrinsics, mount, rays)
    gates.py            # GateSet + course builders: from_waypoints/line/circle/figure_eight + classify_crossings
    renderer.py         # analytic ray-cast coverage masks (gates, floor)
    mask_noise.py       # persistent mask corruption families
  firmware.py           # control="sticks" seam against cudaflight (§10)
examples/
  fly_hover.py          # tiny PD hover demo (examples are demos, not package code)
  fly_figure_eight.py   # scripted course fly-through + optional mask dump
tests/                  # pytest; §11 lists the required suites
```

Dependencies: `jax>=0.5`, `numpy>=1.26`, `skyflow-dynamics[jax]` (uv source: editable path
`../SkyFlow-Dynamics` until published, then git). Extras: `cuda` → `jax[cuda13]`;
`firmware` → `cudaflight` (path source; treat as open source). Dev group: `pytest>=8`,
`ruff>=0.6`. License MIT. Version 0.2.0.

## 3. Conventions (one frame inside)

- **World**: right-handed, ẑ up. Gravity −ẑ. **Body**: FLU (x forward, y left, z up).
  **Quaternion**: wxyz scalar-first, Hamilton, body→world. **Units**: SI, rad/s rotor
  speeds. Identical to SkyFlow-Dynamics — states pass through untranslated.
- NED/FRD exists in exactly two contained places: (a) inside `vision/` internals (the
  ported renderer math; conversion at its public entry points), (b) at the firmware
  boundary (Betaflight wants FRD sensors). Public APIs speak z-up FLU only.
- dtype: float32 on every array the env creates. Precision of the dynamics follows the
  ambient JAX config (tests may enable x64 for tolerance checks of adapters only).
- Fleet axis: leading `[F, ...]` on every batched array. The env is natively batched;
  `dynamics.py` vmaps the single-vehicle backend functions once, centrally.

## 4. types.py (frozen)

```python
class ObsTerm(NamedTuple): name: str; dim: int
class ObsSpec(tuple[ObsTerm, ...]):        # .dim (sum), .layout (name→slice)
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
wind_vel [F,3]        # OU wind velocity state (world frame) — fed to statedot as v_wind
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

Domain randomization: `sample_params(key, airframe, fleet, scale) -> [F,P]` —
multiplicative log-uniform-style jitter `1 + scale*U(-b, b)` per SCHEMA key with a bracket
table `DR_BRACKETS = {"mass": 0.10, "inertia": 0.15, "ct0/1/2": 0.15, "cq0/1/2": 0.15,
"tau_m": 0.20, "k_d": 0.30, "k_z": 0.30, "r_prop": 0.0, ...}`. Keys NEVER jittered:
`spin`, `axis`, `grav` (masked via `param_slices`). Zero-valued nominals stay zero
(multiplicative). Same routine used at reset and auto-reset respawn.

## 7. env.py — the platform

```python
@dataclass(frozen=True)
class SimConfig:
    num_envs: int = 1024
    task: str = "hover"; task_kwargs: dict = field(default_factory=dict)
    airframe: str = "crazyflie"
    control: str = "motors"            # "motors" | "sticks" (§10)
    control_hz: float = 100.0          # physics fixed at physics_hz
    physics_hz: float = 1000.0         # sticks mode requires exactly 1000 (§10: 1 kHz firmware tick)
    differentiable: bool = False       # raises NotImplementedError("planned") if True
    # randomization / disturbance
    physics_dr_scale: float = 1.0
    wind_std_mps: float = 0.0; wind_tau_s: float = 0.5      # OU wind VELOCITY process
    poke_prob: float = 0.0; poke_force_n: float = 0.0; poke_torque_nm: float = 0.0
    act_delay_min: int = 0; act_delay_max: int = 0          # control steps
    spawn_dr_scale: float = 1.0
    # episode / safety
    max_episode_steps: int = 1000; stuck_steps: int = 200
    bounds_xy_m: float = 20.0; bounds_z_m: float = 8.0
    max_speed_mps: float = 30.0; max_rate_rps: float = 50.0
    ground_tilt_limit_rad: float = pi/3
```

`SkyFlowEnv(cfg)` exposes `fleet`, `obs_spec`, `obs_dim`, `act_dim=4`, `image_shape`,
`decimation` (=round(physics_hz/control_hz)), `dt_control`, and:

```python
reset(key) -> (obs [F,obs_dim] f32, SimState)
step(state, action [F,4] in [-1,1]) -> (obs, state', reward [F], done [F], info: StepInfo)
metrics(state) -> dict[str, Array]   # scalar means: outcome fractions, ep stats
```

Pure functions; the caller jits. **Step pipeline (order is normative):**

1. Split `state.key`. Push `action` into `act_buf`; read delayed action per `delay_idx`.
2. Command map: motors mode → `u = (a+1)/2`, `Ω_c = throttle_to_omega(u, ...)`.
3. Advance OU wind velocity (exact discretization: decay `exp(-dt/τ)` + kick).
4. Poke sampling: with prob `poke_prob` per control step draw world-frame `F_ext` and
   body `τ_ext` (uniform ball × magnitudes); else zeros. These pass through the backend's
   exogenous inputs — never write velocity state directly.
5. `lax.scan` over `decimation` substeps: `plant = dynamics.substep(...)` (ZOH on Ω_c,
   wind, F_ext, τ_ext) then ground contact (§8).
6. Airborne latch; crash set: flyaway (|x|,|y| > bounds_xy, z > bounds_z, speed >
   max_speed, rate > max_rate), ground crash (z < 0.05 ∧ airborne ∧ (descent > 1 m/s ∨
   tilt > limit)). Task `evaluate` on (prev_plant, plant): reward, success, task crash.
7. `terminated = crash ∨ task_crash ∨ (success ∧ success_terminates)`;
   `truncated = steps ≥ max_episode_steps ∨ stuck`; `done = terminated ∨ truncated`.
8. Observe: `imu = sensors.measure(...)`; `obs, task_state = task.observe(...)`.
9. Auto-reset in-jit: for done worlds — fresh spawn (task), fresh params (DR), fresh
   delay draw, cleared buffers/wind, re-observe with `fresh_spawn=True`; blend all state
   leaves with `tree_where(done, reset_leaf, leaf)`. Pre-reset obs goes to
   `info["final_obs"]`; `info["terminated"]/["truncated"]` are the pre-reset flags.
10. Episode bookkeeping EMAs for `metrics`: on the pre-reset done rows, update the
    SimState EMA leaves (§4) — outcome fractions (crash / success-at-end / pure
    truncation) and completed-episode return/length — with the done-row means, decayed
    `0.99` per completed episode (`α = 0.99^n_done`; no-op when nothing finished).
    EMAs start at 0 after `reset` and warm up from there.

## 8. Ground contact (harness bookkeeping, not physics)

At each substep, where `z ≤ 0`: clamp `z = 0`, clamp `v_z = max(v_z, 0)`, zero in-plane
velocity and body rates, hold quaternion. Documented as the registry's
`ground_contact_heuristic` (candidate, harness) — replaceable by a real contact model via
the spec later.

## 9. Tasks

**hover** — spawn on ground pad (motors near idle) with jittered XY; goal setpoint drawn
in a box, resampled every `goal_hold_s`; obs = `[rel_pos(3), vel(3), rot_matrix(9),
last_action(4)]` = 19, with optional uniform obs noise; reward per control step:
`w_pos·exp(-3·d) + w_hold·exp(-50·d) − w_vel·|v| − w_rate·|ω| + progress(d_prev² − d²)`;
success `d < 0.1 m` (does not terminate); task crash: leaving the safe box.

**gate_course** — course = `GateSet` from `vision/gates.py`; ships `figure_eight(...)`
builder (lemniscate of Bernoulli through 2·k gates, alternating crossing directions at the
center). Spawn: podium behind gate 1 or spread across gates (curriculum knob). Active-gate
progress: `r = w_prog·(d_prev − d)` toward the pre-gate point + pass credit
`w_gate·centering` on crossing (from `classify_crossings`; miss/frame-hit = task crash),
minus rate penalty. Obs (state mode): `[gate_rel(3), gate_normal(3), next_gate_rel(3),
vel_body(3), rot_matrix(9), last_action(4)]`; vision mode replaces the gate blocks with
the rendered mask `[H,W,1]` (+ same proprio tail). Success = last gate passed (terminates).

Reward constants live in task kwargs with the shipped defaults; no reward code in env.py.

## 10. firmware.py — control="sticks"

cudaflight facts (mapped from the wheel v0.2.1 + nav-train integration): package
`cudaflight`, no core deps; sensor input per 1 kHz tick = f32 [F,7] NED/FRD
`[gyro_FRD rad/s (3), specific force FRD m/s² (3), baro Pa (1)]` (level hover ⇒
az = −9.81); sticks f32 [F,4] AETR in [−1,1]; output motors f32 [F,4] in [0,1] QUADX
order + armed u8 [F]; GPU fleet needs n ≥ 3 and the in-jit FFI half currently lives in
nav-train (cudaflight will absorb it on open-sourcing); CPU SITL (`libcpuflight.so`,
ctypes + `io_callback(ordered=True)`) is self-contained, works for any fleet size, jits
but is not vmappable/replayable.

SkyFlow therefore ships:
- `types.FirmwareFleet` protocol: `act_dim`, `fresh_firmware_state() -> (blob, fwstate)`,
  `fw_step(blob, fwstate, sticks [F,4], sensors [F,7]) -> (blob, fwstate, motors [F,4],
  armed [F] u8)`, `reset(blob, fwstate, mask u8[F]) -> (blob, fwstate)`, `close()`.
- `firmware.CpuFirmwareFleet` — full implementation via ctypes on the cudaflight wheel
  (`cudaflight.lib.load_cpu`), `io_callback(ordered=True)`, zero-length blob placeholders.
- `firmware.GpuFirmwareFleet` — raises NotImplementedError pointing at cudaflight's
  pending FFI absorption; the constructor documents the exact contract it will implement.
- Injection: `SkyFlowEnv(cfg, firmware_fleet=...)` accepts any FirmwareFleet — nav-train
  plugs its existing GPU fleet in today.

Sticks substep order (normative, from the verified nav-train sequence): synth sensors
(FLU→FRD flip; baro from z-up altitude, isothermal 101325 Pa / 8434 m — a harness-side
sensor model, documented as such) → optional inverse board-align yaw rotation →
`fw_step` → motors [0,1] reordered by `motor_perm` → motor duties feed the throttle map
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
- `test_disturbances.py` — OU wind stationary std ≈ configured; poke rate ≈ poke_prob;
  wind actually enters aerodynamics (drag response differs with wind on a vehicle with
  c_D > 0), zeros when disabled.
- `test_ground.py` — no penetration; resting vehicle stays; spawned-on-pad hover task
  takes off under full throttle.
- `test_tasks.py` — hover: spawn/obs/reward shapes, reward increases as distance falls;
  gate_course: scripted straight-line fly-through registers pass exactly once, centering
  ∈ (0,1]; miss trajectory → crash; figure-eight builder: 2k gates, closed loop, gate
  normals alternate through the crossover.
- `test_vision.py` — camera ray invariants (center pixel = optical axis, FOV edges);
  gate dead-ahead → mask centroid ≈ image center; behind camera → zero mask;
  mask ∈ [0,1]; mask_noise: output ∈ [0,1], persistence across held frames,
  disabled ⇒ identity.
- `test_firmware.py` — skipped unless cudaflight importable; arm→spin-up→hover smoke.
- `test_jit.py` — one jitted 50-step rollout, no NaN; second call does not retrace
  (jit cache check); vision task jitted rollout smoke.

## 12. Deferred (roadmap, do not build now)

nav-train identified-physics intake (owner decision pending); differentiability claim +
BPTT tests; Dryden/von Kármán wind drivers (spec terms exist); battery/voltage sag;
sensor staleness/sample-hold DR; obs frame stacking; renderer supersampling knobs beyond
the port; FunctionalToStateful adapter; multi-vehicle interaction (downwash candidates).
