# SkyFlow technical-debt register — 2026-08-23

Provenance: five parallel subsystem audits (core env/types, firmware seam, viz,
tasks+vision, tests/examples/docs), triggered by the shipped sticks-mode
task-state bug. Every finding is pinned to a file:line that was read and, where
marked "verified", executed. Fix sizes: S = under an hour, M = a session,
L = multi-session.

Legend: [C]=core audit, [F]=firmware audit, [V]=viz audit, [T]=tasks/vision
audit, [Q]=tests/docs audit.

---

## 0. The six systemic roots

Most findings are instances of six patterns. Fixing a pattern kills its class.

- **R1 — "missing" and "hidden" share one sentinel.** `getattr(path, None)` →
  `None` → hide, everywhere binds and duck-typed hooks are read. A typo, a
  renamed field, or the wrong pytree renders as a clean-looking scene.
- **R2 — the render thread has no boundary.** It runs user callables, jax
  compiles, and device transfers, with no try/except around the loop body.
  Failures present as a frozen or black window.
- **R3 — `SimState.task_state` changes shape by mode.** The firmware carry
  wraps the task pytree in sticks mode; an accessor exists but every raw read
  compiles fine. Shipped one bug already; three more raw-read paths remain.
- **R4 — the sticks axis is untested.** Real-firmware tests are 1-step smokes
  at fleet ≤ 4 with hover; production is sticks+gpu+gate task+fleet 1024+DR.
  No CI exists at all.
- **R5 — duplicate truths.** Plant [*,17] layout ×3, step-pipeline contract ×3,
  sensor [F,7] layout ×3, NED flip ×4 hand-rolled, cudaflight minimum ×4
  (mutually contradictory), crossing math ×2, `fleet >= 3` ×2.
- **R6 — no lifecycle.** `SkyFlowEnv` has no `close()`; the CPU SITL allows one
  fleet per process; use-after-close is a segfault; eeprom temp files leak.

---

## 1. Fix-first list (production impact for sticks GPU training, in order)

1. **`armed` is read from the firmware and thrown away** — env.py:785 discards
   it every substep; it never reaches `info`, so the HUD ARMED light is dead
   and a mid-episode disarm (failsafe, runaway-takeoff) looks like a policy
   failure. [F4] Fix S. NOTE: this is the exact observable the fig8
   ground-start debugging needed.
2. **Delay-ring neutral is 50% throttle in sticks mode** — env.py:649,882 fill
   the action ring with 0.0; in AETR that is mid-throttle, not idle, for the
   first `delay` steps of every episode and respawn. Breaks arming. Any
   sticks run with `dr.delay_steps > 0` is wrong. [F7] Fix S.
3. **The carry can unpack a 3-field task state as (ts, blob, fwstate)** —
   env.py:707-711; a bare HoverTaskState destructures without error and the
   firmware steps on a goal tensor. Silent garbage physics. [C1] Fix S
   (isinstance guard). Structural fix for R3: make the carry unconditional or
   rename the wrapped field so raw reads fail loudly. Fix M.
4. **Bind/hook misses are silent** — primitives.py:235-256 (`_lookup`),
   record.py:166-216, plus 5 duck-typed task/env hooks read via getattr
   (viewer.py:247-258, record.py:134-160). One warn-once helper converts the
   whole class from invisible to diagnosable. [V1,C4,T1,T16] Fix S.
5. **Render thread: no exception boundary; `_snap` swallows everything** —
   viewer.py:202-230 (thread dies, `open` stays True), viewer.py:343 (bare
   except drops frames forever). [V2,V3,C5] Fix S.
6. **The vision policy pane still stalls on GPU jax** — viewer.py:592 →
   fpv.py:98-114: `_policy_floor` is a second PilotCam, jitted on the training
   device, unreachable from outside (nav's CPU-pin covers only `_pilot`). Every
   vision-task live viewer will black-screen exactly like the shipped bug.
   [V4] Fix S (constructor injection or device= arg).
7. **`firmware="auto"` silently falls back to the CPU SITL** — env.py:487-501:
   fleet < 3, no CUDA, or ANY GPU construction failure (bare except → one
   RuntimeWarning) → sequential CPU firmware, orders slower, different
   numbers. `XLA_PYTHON_CLIENT_PREALLOCATE` is never checked. [F1] Fix S.
8. **No `SkyFlowEnv.close()`** — env.py has no teardown; tests reach `env._fw`;
   the benchmark constructs sticks envs in a loop and breaks on the second;
   use-after-close segfaults (firmware.py:201-206, closure over freed GPU
   handle firmware.py:298). Eeprom temp files never unlink (env.py:537-542).
   [C16,F2,F10,F15] Fix S/M.
9. **`battery_sag` with default `factors=None` disables the thrust-to-weight
   guard entirely** — env.py:421, params.py:294-296: worlds that cannot lift
   off, truncated as `stuck`, reads as a policy problem. [C6] Fix S.
10. **`dr.obs_noise` sprays metre-scale noise over vision mask pixels** —
    env.py:614-621 applies one half-width to the whole obs vector; a mask is
    destroyed by values sane for positions. [C8,T7] Fix M (per-ObsTerm scale;
    the units field already exists).

---

## 2. Correctness traps (silent wrong behavior)

- OOB `active_gate` is clamped by JAX gathers and STICKY — an index of 99
  never self-heals, success can never fire, reward looks plausible.
  gate_course.py:214-350. [T2] Fix S (clip in observe/evaluate).
- Reward geometry vs collision geometry: `_centers/_normals/...` cached at
  construction, `classify_crossings` reads live `self.gates`; a subclass that
  reassigns `gates` gets a chimera. gate_course.py:149-153 vs :319. [T3] Fix M.
- `corrupt_mask(scale>=4.17)` silently blanks the whole mask (hole_thr*scale
  unclamped, mask_noise.py:283-286; verified coverage 0.911→0.044). [T4] Fix S.
- `outer_half < inner_half` builds an invisible, non-solid, passable gate —
  validated nowhere (gates.py:127-196; verified pass=True through empty sky).
  [T8] Fix S.
- Non-divisor `control_hz` silently desyncs env clock from task clock —
  env.py:405,427,566 (decimation rounds; task gets the requested hz). [C7] Fix S.
- `_build_task` forwards config by parameter-NAME matching; a renamed builder
  param or an opaque callable (functools.partial) silently gets defaults.
  env.py:559-570. [C10] Fix S.
- Info/metrics key collisions resolve in OPPOSITE directions, both silent —
  env.py:928-939 (env wins) vs :975-976 (task wins). [C11] Fix S.
- `motor_perm` default (3,1,0,2) is validated only as a permutation; a custom
  airframe or `yaw_motors_reversed` dump transposes the mixer with no check.
  env.py:337-465, spin/rotor order live in 3 unlinked places. [C13,F9] DECIDED
  2026-08-25: no check — airframe `spin` + the vehicle's dump are the truth.
- Firmware protocol text says fresh state is "disarmed"; both implementations
  return ARMED (types.py:146 vs firmware.py:170,322). External implementers
  ship dead fleets. [F5] Fix S. The real arming/re-arm rule exists only in an
  example comment (fly_figure_eight.py:78-79) and contradicts the snapshot
  text; no test pins post-auto-reset arming. [F6] Fix M.
- DESIGN §10 names a board-align rotation step the code does not implement; a
  real dump with `align_board_yaw` set swaps roll/pitch inside the firmware.
  DESIGN.md:407 vs env.py:771-789. [F8] Fix M (implement or reject).
- Injected fleet with wrong size = heap over-read before any Python error
  (env.py:461 checks act_dim only; firmware.py:130-148 raw pointers). [F12] Fix S.
- `settle_ms` and `device_index` unreachable from SimConfig: snapshots taken
  with unsettled filters; multi-GPU boxes bake cuctx on device 0.
  env.py:483-501 vs firmware.py:107-114,249-258. [F16] Fix S.
- Sticks-only config silently inert in motors mode (`firmware`, `baro_noise_pa`,
  `motor_perm` ignored, unvalidated). env.py:355-371,476-479. [F17] Fix S.
- SimState advertised checkpointable; CPU-sticks firmware state is host-side
  and NOT in the pytree; GPU carry leaves break the [F,...] leaf contract that
  public `tree_where` assumes. env.py:316, firmware.py:88-92. [F18] Fix M (doc).
- `grow_from_keys` misindexes family 6 of a 6-family key array (JAX clamps);
  zero callers, zero tests, broken by default. mask_noise.py:141. [T5] Fix S.
- Vision ObsTerm units string says "{0,1}" but default supersample=2 emits
  {0,.25,.5,.75,1} — the units-hash contract certifies a lie. gate_course.py:181,
  camera.py:51. [T6] Fix S.
- Podium spawn keys on `gates is None` (argument identity, not value): passing
  the identical default course explicitly moves the spawn 10 m. gate_course.py:162.
  [T9] Fix S.
- EMA decay 0.99**n_done degenerates to the instantaneous mean at fleet 1024 —
  the smoothing the name implies does not exist at production scale. env.py:843-855.
  [C26] Fix S (doc or per-fleet decay).
- Mid-run firmware failure leaves the CPU fleet desynced with no recovery path
  and no sticky failure flag. firmware.py:144-148. [F14] Fix M.
- No fork/thread guards on either firmware handle; two GpuFleets on one device
  construct without complaint. firmware.py:107-127,239-317. [F13] Fix M.

## 3. Viz-specific (beyond the fix-first items)

- `snapshot()` device_get runs ON the render thread (frame() path); fleet
  scatter pulls the WHOLE fleet positions with no device-side stride; GPU
  sticks pulls the entire [F·stride] firmware blob per frame. frame.py:112-140.
  [V5,F3] Fix M (snapshot in the caller's thread; stride on device).
- record/replay silent data loss: never-resolving binds simply absent then
  hidden at replay; save() validates only at the end (a bind that misses step 0
  loses the whole flight); one missing `action=` drops the whole action array;
  binds resolve against ViewFrame live but SimState in record (different
  namespaces); replay surfaces only the `task_state.` subtree. record.py:166-281,
  replay.py:60-89. [V7,V8,V9] Fix S each.
- flight.npz: no version/width gate on load (plant 17 unchecked; header
  "skyflow" never read); Scene.from_dicts is strict at replay (a user primitive
  or added field makes old logs unreplayable); header mixes NED (gateset) and
  z-up (scene) with no frame marker; `every` means three different things
  across capture/extend/replay. record.py, replay.py. [V10,V11,V13,T11] Fix S.
- extend()'s watch-vs-fleet shape heuristic mislabels worlds when
  len(watch)==fleet or watch is a permutation. record.py:236-240. [V12] Fix S.
- push() ignores display_hz; force=True can report "drawn" for a dropped frame
  and times out silently; screenshot mailbox races; the 20fps reuse throttle is
  keyed on OBJECT IDENTITY and skips the event pump; idle() dead before first
  frame on sync viewers; mailbox pins the last fleet-sized state forever;
  frames/shot budget burned by idle redraws; ready-wait timeout unchecked.
  viewer.py. [V15-V21,V38] Fix S each.
- SDL_VIDEODRIVER set process-wide and never restored (one headless viewer
  makes all later windows invisible); one pygame window per process, unstated;
  sync close() tears down a threaded viewer's display. viewer.py:185-198,278-288.
  [V14] Fix M.
- Duplicate truths: plant slices transcribed in frame.py/record.py/hud.py
  (canonical: types.py:172, no import link, no ViewFrame validation);
  _PILOT_RES + _camera/_gates now dead after the V-key removal; the splash
  docstring still names an H key that no longer exists; _bars return value dead
  and geometry re-derived by hand. [V22,V23,V24,V26] Fix S.
- viewer._pilot is a private attr two external callers assign; _policy_floor
  has no equivalent hook. [V25] Fix S (constructor kwargs + tiny Protocol).

## 4. Tasks/vision-specific

- Subclass contract is implicit: binds by string, viz hooks by getattr, cached
  geometry attrs by name, task_state leaves must lead [F], class-vs-instance
  obs_spec split. One `check_task_hooks()` + a contract note fixes discovery.
  [T16, T-Rank4] Fix S/M.
- NED flip hand-rolled at camera.py:69,103, renderer.py:82, firmware.py:64
  despite a canonical helper (_ned.py) and three "single site" claims. [T10] Fix S.
- mask_noise pixel constants hard-coded to 64px frames; erasure_at returns int
  dtype when scale<=0; corrupt_mask/erasure_at duplicate 9 kwargs verbatim;
  405-line module has NO production consumer (wiring deferred per
  gate_course.py:263). [T13,T14,T-R5] Fix S/M.
- Crossing/centering math implemented twice in two frames (gates.py:297-307 vs
  gate_course.py:326-342) — one epsilon change desyncs crash from pass. [T15] Fix M.
- render_masks Python-unrolls per gate: compile time linear in G. [T17] Fix M.
- Dead: gates.line, gates.circle (exported, uncalled, untested);
  spawn_alt_jitter_m inert on the podium path; SimState.last_action written
  never read while an obs term of the same name means something else. [T-R5,C17] Fix S.

## 5. Tests, examples, docs, CI

- Coverage axis holes (top 5): sticks+real-firmware auto-reset of done worlds;
  sticks+gpu multi-step at fleet ≥ 256; sticks+full DR through real firmware;
  sticks+gate task (the shipped-bug axis); sticks+viz/record on a real env.
  [Q] Fix M total — one parametrized control-mode fixture is the real fix.
- conftest claims "all CPU"; nothing pins JAX_PLATFORMS, so on this box the
  bit-exactness tests validate GPU fusion. [Q1] Fix S.
- mp4 export untested wherever imageio is absent (no extra declares it). [Q2] Fix S.
- Firmware ImportError guidance paths self-skip exactly where cudaflight IS
  installed. [Q3] Fix S. 8 construction guards have no negative tests. [Q4] Fix S.
- Screenshot smokes assert only file size; the bug that shipped would pass
  them. [Q5] Fix S (assert accent pixels, as test_viz_panes.py:51 already does).
- Private-attr coupling in tests (env._fw, log._plant, viewer._thread, ...).
  [Q9] Fix M. Eeprom test depends on alphabetical file order. [Q13] Fix S.
- Examples: teleop's S (throttle down) collides with the viewer's S
  (screenshot) — every descent writes a PNG; fly_hover teaches the raw
  task_state anti-pattern; stale cudaflight >= 0.5.0 claims; no example is
  smoke-tested. [Q-3] Fix S each.
- DESIGN.md: battery_sag still "roadmap"; DomainRand block omits factors +
  battery_sag; SimConfig block omits eeprom fields; §11 required-tests list
  omits three suites and encodes the sticks-viz gap; step-pipeline text drifted
  from code (EMA/firmware leaves excluded from the blend). [Q-4,C12] Fix S.
- No CI, no coverage tooling, ruff/pyright configured but never invoked, no
  --strict-markers, no timeout on GPU tests, pytest-queue undocumented in-repo.
  [Q-5] Fix M (one workflow: ruff + pyright + pytest -m "not gpu"; a gpu job).

---

## 6. Suggested attack order

1. **Loud-failure wave (all S): DONE 2026-08-24** — armed→info [F4]; delay-ring
   throttle fill [F7]; carry isinstance guard [C1]; bind warn-once [V1] (also in
   record._resolve_path); render-thread boundary + snap_drops counter [V2/V3];
   pilot/policy_floor constructor injection [V4]; auto-fallback ALWAYS warns +
   PREALLOCATE check in GpuFirmwareFleet [F1]; battery_sag guard gated on sag
   alone [C6]; OOB active_gate clip in observe+evaluate [T2]; gate-geometry
   validation in _gateset_from_world [T8]; hole-threshold clamp in corrupt_mask
   + erasure_at [T4]. Verified: full suite 178 passed, ruff + pyright clean.
   Uncommitted; the working tree also carries UNRELATED in-flight work
   (estimator-error / cmd-drop SimState leaves) — separate the hunks at commit.
   Deferred from this wave: armed_frac metric (needs a SimState leaf or a
   metrics-side channel); nav's live-viz tap does not carry info["armed"] yet.
2. **Lifecycle [C16/F2/F10/F15]: DONE 2026-08-24** — SkyFlowEnv.close() (+ context
   manager, owns-fw rule: injected fleets belong to the caller); one-CPU-fleet
   guard (weakref, frees on close); closed-handle raises at the public entry
   points AND the io_callback host halves (a host-half raise poisons jax's
   ordered token — public-entry guards keep normal misuse clean); eeprom image
   reaped at close/GC/exit (weakref.finalize) + on failed construction;
   benchmark closes envs. tests/test_lifecycle.py pins it all.
3. **Structural R3 [C2/F3/V-]: DONE 2026-08-24** — field renamed
   `SimState.task_state` → `task_carry`; `FirmwareCarry` moved to types.py
   (public export); `SimState.task_state` is now a PROPERTY that passes bare
   pytrees through (motors unchanged) and RAISES TypeError on the carry, so
   every raw sticks read — snapshot fallback, binds, user code — fails loudly
   with the accessor named. env.task_state() unwraps by isinstance, not mode.
   Pytree shapes unchanged in both modes. tests/test_task_carry.py pins it.
   nav's adapter already reads through the accessor — repin-compatible.
4. **Test spine [Q]: DONE 2026-08-24** — conftest pins jax default device to CPU
   session-wide (gpu-marked tests place explicitly); parametrized control_mode
   fixture (sticks skips without the SITL); tests/test_sticks_axis.py walks the
   production axis (sticks+gate+jit+auto-reset through the REAL firmware,
   FlightLog bind round-trip, snapshot-through-accessor + raw-read raise);
   tests/test_config_guards.py adds the missing negative tests; --strict-markers;
   .github/workflows/ci.yml (ruff + pyright + CPU pytest; firmware extra installed
   so the sticks axis runs IN CI; gpu-marked excluded — no hosted CUDA runner).
5. **Docs/duplicate truths [R5]: DONE 2026-08-24** — arming protocol: the
   normative statement now lives on types.FirmwareFleet.fresh_firmware_state
   (armed-on-ground snapshot; re-arm only on LOW throttle) — the "disarmed" lie
   is gone. cudaflight floor: pyproject 0.6.0 is THE floor; firmware.py marks
   per-feature versions as history; stale 0.5.0 claims in examples fixed.
   fleet>=3: one constant, firmware.GPU_FLEET_MIN, read by both sites. NED flip:
   firmware.flu_to_frd delegates to vision._ned.flip_xyz; false "exactly once"
   claims in gates.py/renderer.py corrected. DESIGN sync: battery_sag +
   cmd_drop_prob shipped rows; DomainRand block gains factors/battery_sag/
   cmd_drop_prob/obs_error; SimConfig block gains eeprom fields; §7 step 9 names
   the two blend exclusions (EMA leaves, firmware pair — also fixed in the
   tree_where docstring); §10 board-align step replaced by the reject rule; §11
   lists all 22 suites. Viewer: screenshot moved S→P (teleop's throttle-down
   collided), dead _PILOT_RES cycle/`_camera`/`_gates` removed, splash H-key
   mention gone, hud advances from _bars' return. NOT deduped (canonical home
   declared, transcriptions left): plant-layout restatements in comments/tests.
6. **M-sized correctness: DONE 2026-08-24 (two deliberate partials)** —
   [F8] board-align — REVISED 2026-08-25: the wave-6 rejection was WRONG. The
   Air75 factory CLI dump carries align_board_yaw = -135 and yaw_motors_reversed
   = ON; a real config must never be refused. Now: the effective align_board_*
   (overrides win) WARNS once when nonzero; the inverse pre-rotation stays
   planned (implement + verify against a real SITL hover). yaw_motors_reversed
   is accepted with no check; [C13/F9] the airframe-spin consistency check and
   the motor_perm derivation stay OPEN. [C8] obs_noise now skips
   image terms (ObsTerm.image=True) via a per-dim scale; numeric terms bit-exact
   legacy. [T3] gates is a property whose setter re-derives ALL cached geometry —
   the reward/collision chimera is impossible. [T15] one crossing solve:
   gates._crossing_point shared by classify_crossings and the new
   crossing_offsets; gate_course.evaluate consumes it (centering/miss can drift
   from the pass predicate by nothing; reward numerics move by last-ulp only —
   dot-then-interpolate vs interpolate-then-dot). [T6] DONE 2026-08-25: the mask
   ObsTerm units read "[0,1] coverage HxW row-major" (supersample emits
   fractions); nav's task declares its own units strings and never consumed
   this one, so no downstream hash moves. [V5] partial: the fleet
   scatter is strided ON DEVICE (snapshot fleet_stride; viewer caps ~2048 dots);
   snapshot itself stays on the render thread by design (callers must not pay
   draw time) — the carry no longer rides into it after wave 3, so the GPU
   sticks blob pull is gone.

Still open after wave 6: motor_perm derivation + airframe-spin consistency
[C13/F9]; the inverse board-align pre-rotation [F8]; Q9 private-attr coupling in
the older tests; example smoke tests; the §2/§3/§4 trap-list items not named in
any wave (C7 control_hz desync, C10 name-matched task kwargs, C11 info-key
collision direction, F12 injected-fleet size, F13 fork/thread guards, F14
sticky CPU-fleet failure, F16 settle_ms/device_index, F17 sticks-only knobs
inert in motors, F18 checkpoint doc, C26 EMA decay at fleet 1024, T5, T9,
T13/T14, T17, V7-V21, V38, T16 check_task_hooks); §7 items D1-D4, D6-D9,
D12-D24. D3 is closed as user responsibility.

Cross-author note 2026-08-24: ruff/pyright are now enforced repo-wide (CI), which
required mechanical fixes in the OTHER session's files: en-dashes in errors.py
comments, Optional-narrowing asserts + one deliberate-invalid-call ignore in
test_obs_error.py/test_delay_link.py, and one possibly-unbound `done` init.
Behavior unchanged; separate these hunks at commit if authorship matters.

---

## 7. DR-area review — 2026-08-25 (the five DR commits e03b597..531a477)

Requested by James after waves 1-6 landed. Read-only review of the factor
stage, battery_sag, cmd_drop_prob and the L5 estimator model against the six
roots. Nothing here is fixed yet; James decides what to take.

Silent-wrong-behavior traps
- **D1** DONE 2026-08-26 (image tasks receive `true_plant=`; the mask renders from the true pose). Was: env.py: with `obs_error` on, the vision task renders the gate MASK
  from the ESTIMATED pose (`corrupt_plant` output feeds `task.observe`, which
  calls `render_masks`). A real camera images from the true pose; only derived
  state is wrong. Violates ERRORS.md "never corrupt what the agent truly
  knows". No test runs vision + obs_error. Fix: render from `plant`.
- **D2** DONE 2026-08-26 (zero attitude widths skip the compose). Was: errors.py:225 quaternion renormalization is NOT bit-exact at zero
  widths (1.19e-7 measured on non-identity quats); contradicts the "none
  profile leaves values bit-identical" claim. The pinning test hovers at
  identity attitude, so it cannot see it. Fix: skip the rotation compose when
  attitude widths are 0.
- **D3** WILL NOT FIX (James 2026-08-26: the user owns thrust-to-weight sanity). Was: params.py:293 + env.py `_tw_reguard`: the shipped default
  (factors=None, battery_sag=0, body DR on) still runs NO thrust-to-weight
  guard — [C6] is half-closed. A low-T/W airframe or a `brackets={"mass":..}`
  override yields worlds that truncate as `stuck`. Fix: always re-guard.
- **D4** "scale=0 with delay (0,0) is bit-exact nominal" (env.py docstring,
  DESIGN §7) is now false: `effective()` never scales `cmd_drop_prob` or
  obs_error `p_drop`. Restate as an `off()`-only invariant.
- **D5** DONE 2026-08-25 (ObsTerm.image flag replaces the units-prefix key). Was: env.py `_is_image_units`: the obs_noise image exclusion keys on a
  free-form units prefix. `ObsTerm.units` defaults to ""; a vision task that
  omits units gets metre-scale noise on every pixel again. Fix: an explicit
  image marker (ObsTerm field, or the task's image term name), not a string.
- **D6** params.py:176 `base.get(name, 0.0)`: a SCHEMA key missing from
  DR_BRACKETS/RESIDUAL_BRACKETS is silently never jittered (an override for it
  raises). Fix: assert table keys == schema keys minus NEVER_JITTER.
- **D7** params.py:307 factor groups combine additively; disjointness is a
  comment, not a check. Fix: validate in `_factor_tables`.
- **D8** params.py:242 `factor_floor` bounds only the low side;
  `factors={"mass": (0, 50)}` constructs. Fix: bound `hi`.
- **D9** env.py negativity loop omits battery_sag/cmd_drop_prob; a negative
  battery_sag with scale=0 folds to -0.0 and passes. Validate raw fields.

Untested axes
- **D10** DONE 2026-08-25 (tests/test_sticks_dr.py). Was: no test runs sticks mode with battery_sag, cmd_drop_prob, obs_error
  or factors — the knobs justified by sticks-mode evidence (R4).
- **D11** DONE 2026-08-25 (test_sticks_dr.py). Was: estimator leaves (est_ou/est_hold/est_bias) untested through the
  auto-reset path.
- **D12** `test_scale_zero_disables_every_continuous_knob` omits the four new
  knobs (and would fail for two, see D4).
- **D13** obs_error never tested x delay_steps (the measured lethal pair), x
  the gate task, x obs_noise on top.

Duplicate truths / disagreements
- **D14** DESIGN §4 SimState block still names `task_state`, lists no
  cmd_prev/est_* leaves; DRState block omits w_max/est_bias.
- **D15** DESIGN §4 "new traits go HERE, never as new SimState leaves" vs
  ERRORS.md "drift state in SimState" — charter conflict.
- **D16** types.py:201 dr_state comment stale (missing w_max/est_bias) while
  the DRState docstring above it is correct.
- **D17** DESIGN §6 omits `factors` from sample_params and the whole factor
  stage (RESIDUAL_BRACKETS, FACTOR_GROUPS, FACTOR_LIMITS, TW_FLOOR, guards).
- **D18** Three draw classes (DESIGN, env.py docstring) vs four in ERRORS.md
  (drift); the new knobs are classified nowhere in env.py.
- **D19** env.py normative pipeline (steps 1, 8) and DESIGN §7 never mention
  the drop hold or the L5 plant corruption/dropout hold.

Dead / self-contradicting
- **D20** errors.py:124 "must be a dict or None" is unreachable through
  SkyFlowEnv — `effective()` does `dict(obs_error)` first and raises a bare
  ValueError.
- **D21** errors.py:95 comment says "beyond profile" above a tuple starting
  with "profile".
- **D22** errors.py:227 rotor error is a uniform half-width but is documented
  and scaled as a white std.
- **D23** env.py draws `draw_hold` and runs `advance_ou` every step even when
  inert (dead hot-step work; numerically harmless).
- **D24** env.py hold logic: a hold cannot extend or restart on its exit step,
  so realized dropout law is exactly draw_hold's, not the documented
  restart-capable process. Document or allow restarts.

Wave-1 deferrals closed 2026-08-25: `SimState.armed` leaf ([F] bool; motors all
True) + `metrics()["armed_frac"]`; nav's live-viz tap now carries info["armed"]
so the HUD ARMED caption works during training.
