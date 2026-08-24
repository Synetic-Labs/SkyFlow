# ERRORS.md — the error-dynamics charter

Companion to DESIGN.md. DESIGN.md governs the deterministic simulator; this file
governs everything stochastic that corrupts the interface between the true plant
and its two consumers (the firmware and the policy). Scope: the DomainRand block,
`errors.py`, and the corruption hooks in `sensors.py` and `env.py`.

## Why errors get their own charter

Most physical error sources are stochastic dynamical systems — they have state,
time constants, and structure. The two zero-memory extremes (white noise, a
per-episode constant) are the easy endpoints; most real error lives between them.
Structure decides training outcomes as strongly as magnitude does; both measured
examples are in the audit report (nav-train, 2026-08-23):

- Body DR at equal ±20% width: correlated factor draws fly 0.88, independent
  per-rotor draws fly 0.12.
- White observation noise x variable transport delay is lethal to a memoryless
  policy (0.17 airborne) while either alone learns; a 3-row history stack
  restores 0.90 — but only because white noise averages out, which real
  correlated error does not.

## The boundary (unchanged, load-bearing)

SkyFlow-Dynamics stays deterministic: golden vectors pin it, and no SkyFlow
module writes a force, torque, or sensor equation. Randomness enters only at
loop boundaries, through explicitly passed keys. Deterministic modeling gaps
(IMU mount pose, ground effect, vibration modeled physically) go through the
SkyFlow-Dynamics INTAKE protocol; stochastic error models live here.

## The five layers

| layer | what | lives in |
|---|---|---|
| L1 parametric | the vehicle is not the nominal vehicle | `params.py` |
| L2 exogenous | wind, gusts, pokes | `env.py` step |
| L3 transport/actuation | delay, jitter, drops, battery sag | `env.py` step / traits |
| L4 sensing | what the firmware measures (IMU, baro) | `sensors.py` |
| L5 estimation | what the policy observes | `errors.py` |

L4 and L5 are different consumers and never share knobs: the firmware eats the
sensor model, the policy eats the estimator model.

## Draw classes

- **trait** — drawn once per episode, constant within it, redrawn at respawn
  (body params, steady wind, IMU bias, delay draw, battery sag, estimator bias).
- **drift** — a process WITH memory: OU or a random walk advanced every step
  (gust deviation, estimator OU error). New error models default here unless
  the physics says otherwise.
- **process** — memoryless, fresh every sample (sensor white noise, estimator
  white floor).
- **event** — Bernoulli occurrences with consequences (pokes, command drops,
  estimator dropouts). Event RATES are never scaled by the master dial.

## Rules for every error model

1. **Name, units, equations.** Every knob is a physical quantity in SI units,
   documented where it is declared.
2. **Provenance per value.** Every default carries its source class in a
   comment: `datasheet`, `measured` (name the rig and date), or `literature`
   (replace when a measurement exists). The sit-still Allan bench
   (`nav.deploy.control.sysid_sit_still`) is the standard instrument for IMU
   and estimator-rate values: white-noise density -> white, bias instability ->
   bias width, flat-region knee -> OU tau.
3. **Validation test.** Each model ships a test in `tests/`: distribution or
   stationarity check, structural invariants (a corrupted quaternion is a unit
   quaternion), and the off-is-bit-exact guard.
4. **Off is bit-exact.** Zero widths / `None` configs leave values bit-identical
   and the legacy RNG stream untouched (new draws come from `fold_in`, never
   from re-splitting an existing key).
5. **State lives in the pytree.** Traits in `DRState`, drift state in
   `SimState`, both redrawn/cleared at respawn — never module globals.
6. **Physical structure over blanket scalars.** Correlate what physics
   correlates (one rotation error, not nine i.i.d. matrix entries; one shared
   factor, not four independent coefficient draws). Never corrupt what the
   agent truly knows (map constants, its own commanded action, validity flags).
7. **Falsify before you trust.** Deterministic corner probes first, then fleet
   ladders, then training probes; never judge a run whose airborne fraction
   is 0; decision gates get 3 seeds (single-seed outcomes are bimodal).

## Current models

- L5 estimator error: `errors.py` — profiles `mocap` (the current rig) and
  `vio` (planned), per-group bias + OU + white, attitude as one small rotation,
  relative rotor-telemetry error, dropout holds. Config: `DomainRand.obs_error`.
- L3 link: per-episode delay draw + command drops (`cmd_drop_prob` — a dropped
  or LATE frame holds the last applied command, the next success applies the
  newest frame; drops subsume link jitter with NO reordering), battery-sag
  ceiling trait. An i.i.d. per-step jitter index was measured and REMOVED
  2026-08-24: it reorders commands, which no real link does, and the reordered
  stick stream through the firmware's RC feedforward kills takeoff (0.003
  airborne vs 0.91 for drops, whoop 10M probes) — a live example of rule 6.
- L4 IMU: white noise + per-episode bias (`sensors.measure`); baro white noise.
- L1/L2: factor-stage body DR (`params.py`), OU gusts, steady wind, pokes.

`DomainRand.obs_noise` (unit-blind white on the finalized obs vector) is a
LEGACY stress knob kept for compatibility; realism arms use `obs_error`.

## Queued (approved audit backlog, 2026-08-23)

P1: gyro saturation clip + scale factor (sensors.py); IMU mount pose trait
(INTAKE); CoG offset as a rotor_pos common-mode factor group (params.py);
weight-relative pokes. P2: vibration injection, anisotropic gusts, poke
duration, within-episode sag ramp.
