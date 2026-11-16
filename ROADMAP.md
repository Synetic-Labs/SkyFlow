# SkyFlow — roadmap

Deferred work, parked deliberately. DESIGN.md stays the design of record: nothing here
is part of the design until it is promoted into DESIGN.md. A researched idea keeps its
evidence and design sketch here, so the research is not repeated when the idea is
picked up.

## Backlog (not yet researched)

- identified-physics intake for measured airframes (through the SkyFlow-Dynamics INTAKE protocol)
- differentiability claim + BPTT tests
- Dryden/von Kármán wind drivers (spec terms exist)
- battery/voltage sag
- sensor staleness/sample-hold DR
- obs frame stacking
- renderer supersampling knobs beyond the port
- FunctionalToStateful adapter
- multi-vehicle interaction (downwash candidates)
- viz: a rerun sink over the same builders
- viz: analytic camera primitives beyond gates
- per-rotor wind for gust torques (SkyFlow-Dynamics `rotor_gust_gradient`, proposed:
  under 1 % of roll authority on the racer up to severe turbulence at 2 m; needs a
  gust driver correlated across the hub positions)
- propwash thrust fluctuation in descent (SkyFlow-Dynamics
  `propwash_thrust_unsteadiness`, proposed: measured σ(T)/T̄ 0.14–0.20 at 1.2–1.3 v_h
  for 4–6 in rotors; no spectrum or rotor-to-rotor correlation published yet)

## Researched: gyro vibration model + RPM filter in the loop

**Status: parked 2026-08-20 — interesting, not needed yet.** Researched 2026-08.

### The gap

A real quad's gyro carries narrow noise peaks at each motor's rotation frequency and
its harmonics (2x, 3x). With bidirectional DShot the ESC reports true eRPM back to the
flight controller, and the firmware's RPM filter places tracking notch filters exactly
on those peaks — 3 harmonics x 4 motors x 3 axes = 36 notches on a typical tune. Tunes
are light on static filtering *because* the RPM filter carries the load (gyro LPF1 off,
LPF2 high).

SITL builds compile the whole chain out: SITL never defines `USE_DSHOT`, so
`USE_DSHOT_TELEMETRY` is absent and `common_post.h` drops `USE_RPM_FILTER` and
`USE_DYN_IDLE`. Two consequences for training:

- The sim PID loop sees less filter delay than the real loop. Each notch adds phase
  lag near its frequency; a policy trained on the faster sim loop can over-command the
  real quad.
- The sim gyro is clean. The dynamic notch and lowpass filters act on a signal that
  never occurs on hardware.

### Prior art (searched 2026-08)

- The standard simulator IMU model is white noise + bias random walk (the
  [Kalibr model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)), as
  implemented by RotorS/Gazebo, [Flightmare](https://arxiv.org/abs/2009.00563), and
  PX4 SITL. No harmonics.
- Closest paper: a quadrotor deep-RL infrastructure
  ([arXiv 2504.15129](https://arxiv.org/abs/2504.15129)) adds banded vibration
  harmonics at 10–15 Hz with random phases — aerodynamic body shake, not RPM-tracked
  motor noise (hundreds of Hz).
- RL sim-to-real practice injects broadband observation noise plus domain
  randomization, with levels fitted from real flights
  ([Molchanov et al., arXiv 2109.07735](https://arxiv.org/abs/2109.07735)). Robust,
  but not spectral.
- MEMS vibration rectification error — vibration aliasing into a false rate/bias — is
  documented sensor physics
  ([Analog Devices](https://www.analog.com/en/resources/technical-articles/vibration-rectification-in-mems-accelerometers.html),
  [tuning-fork gyro study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6022183/)),
  handled by isolation and compensation, never simulated for training.

Why the combination is absent from the literature:

1. Almost no pipeline runs the flight-controller firmware in the loop; policies see
   state estimates, so there is no filter chain for harmonic noise to exercise.
2. Most sims step physics at 100 Hz–1 kHz and cannot represent 300–800 Hz peaks.
3. Harmonic amplitudes are airframe-specific (prop balance, frame resonance,
   mounting), so no general model exists to publish.
4. The fitting data — synchronized pre-filter gyro plus per-motor eRPM — comes from
   Betaflight blackbox logs, not from the datasets academia uses.

SkyFlow holds all four missing pieces: firmware in the loop, an 8 kHz virtual gyro,
true rotor speed in the physics state, and blackbox logs from measured airframes.
"Measured RPM-harmonic gyro noise through the real firmware filter chain" is a claim
no current simulator makes.

### Design sketch

1. **Firmware (cudaflight).** Compile the RPM filter into SITL_LOCKSTEP behind an
   opt-in define, the same pattern as `SIMULATOR_DYN_NOTCH`. The filter needs one
   input, `getMotorFrequencyHz()` (`flight/rpm_filter.c`); feed it from a virtual eRPM
   source set each step through the lockstep API from the sim's rotor speeds. This
   also unlocks `USE_DYN_IDLE` for tunes that use it.
2. **Plant (SkyFlow-Dynamics).** An IMU vibration model: white-noise floor plus
   per-motor harmonics with amplitude as a function of rotor speed, injected at the
   virtual gyro rate. Fit per airframe from one blackbox flight with
   `debug_mode = GYRO_SCALED` (pre-filter gyro) and bidirectional-DShot eRPM: per
   axis, a PSD floor plus amplitude-vs-RPM for harmonics 1x–3x. Fitted parameters are
   per-drone data and live with that drone's config records, not in this repo.
3. **Validation.** Replay a logged real flight's stick input in sim; compare
   filtered-gyro spectra and step response. This measures the remaining gap instead of
   estimating it.

Order matters: step 1 alone restores the notch cascade's phase lag (the closed-loop
effect); step 2 makes the notches do real work and trains the policy on realistic
residual noise. Step 3 bounds what is left.

### Update 2026-11-15 — the plant side now has a spec term

SkyFlow-Dynamics 0.2.0 carries step 2's physics as the candidate term
`rotor_imbalance_vibration` (`spec.wrench.rotor_imbalance_wrench`):

- Each rotor pulls on its hub with `F_i = U_i·Ω_i²·(cos θ_i, sin θ_i, 0)`, with
  `U_i = m_u·r_u` the imbalance in kg·m and `θ_i` the rotor phase. Source: Ghalamchi,
  Jia, Mueller, IEEE/ASME T-Mech 2020, eq. (4)–(5), measured in flight on 39 g to
  1170 g vehicles.
- The accelerometer line is `U·Ω²/m`. On the racer, 10 mg at one blade tip gives
  1.0 m/s² at hover and 4.5 m/s² at full thrust.
- The gyro line is the rigid-body consequence `|r|·U·Ω/J_zz` about yaw (0.021 rad/s at
  hover for the same imbalance). Roll and pitch lines need hub height or frame flex.
  The paper models no gyro; the only gyro magnitude found in the literature is an
  ArduPilot source comment (0.2–0.4 rad/s on a 180 mm copter).
- Amplitude laws: accelerometer ∝ Ω², gyro ∝ Ω. This replaces "amplitude as a function
  of rotor speed" in step 2 with a one-parameter fit per rotor (`U_i`), plus free
  weights for the 2x and 3x lines and for frame transmissibility. The paper's own
  estimate of one damaged prop doubled between two airframes, so the fit from a
  blackbox log stays necessary.
- The term is not in the backend. It needs the rotor phase as harness state and a
  sample step well below 1/Ω: the 1 kHz physics step cannot carry a 550 Hz line. The
  8 kHz virtual gyro of step 2 is the right place to inject it.

The 2026-11 literature search found no sim-to-real paper that models rotor-harmonic
IMU noise. The claim in "Prior art" still holds.

## Researched: vehicle-family parameter sampling (RAPTOR / L2F)

**Status: parked 2026-11-15 — a different goal from today's DomainRand.** Read in
rl-tools L2F at the commits pinned by rl-tools/raptor (e43ae4b) and rl-tools/hyperdrone
(ad2e56e); recorded in SkyFlow-Dynamics REFERENCES.md, "rl-tools L2F".

### What it is

RAPTOR trains one policy for many quadrotors. Its sampler draws a whole vehicle, not a
perturbation of one:

1. Thrust-to-weight ratio, uniform in 1.5–5.
2. Size: uniform in the cube root of mass, mass in 0.02–5 kg. Sampling mass uniformly
   would favour large vehicles.
3. Thrust curve: scaled so the drawn vehicle has the drawn thrust-to-weight.
4. Arm length: ∝ mass^(1/3), times a deviation factor (parameter 0.1).
5. Inertia: set from a drawn torque-to-inertia ratio, uniform in 40–1200 rad/s².
6. Rotor torque constant, uniform in 0.005–0.05 m. Motor time constants, uniform per
   direction: rising 0.03–0.10 s, falling 0.03–0.30 s.
7. Disturbance force: std up to 0.3 × the surplus thrust-to-weight, so a weak vehicle
   is not pushed harder than it can answer.

Source: `include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h:50-216`
and `src/foundation_policy/pre_training/sample_dynamics_parameters.cpp:49-63`.

### How it compares with §6

- Same idea as the factor stage: draw physical causes, not independent entries. §6
  measured why (0.88 against 0.12 flyable). RAPTOR reaches the same place from the
  other side: every draw is a capability ratio, so every drawn vehicle can fly.
- Different range. §6 moves one measured airframe by tens of percent. RAPTOR spans a
  factor of 250 in mass.
- Their thrust-to-weight floor is 1.5 ("to maintain maneuverability"). `TW_FLOOR` is
  1.3, the measured liftoff edge. The two answer different questions.
- Their disturbance rule is close to `poke_force_frac`: both scale the push to the
  vehicle. SkyFlow scales to the weight; RAPTOR scales to the surplus thrust.
- Their model has no drag and no rotor gyroscopics, so the scheme says nothing about
  how `k_d`, `k_z`, `c_D`, `I_rot` or the rotor-speed limits scale with size.

### Design sketch (if a multi-vehicle policy becomes a goal)

1. A `size` factor with exponents instead of a 0/1 mask: one draw λ scales `mass` by
   λ³, `rotor_pos` and `r_prop` by λ, `inertia` and `I_rot` by λ⁵, `ct*` by λ⁴,
   `cq*` by λ⁵, `c_D` by λ², `k_d` and `k_z` by λ³ (from `k_d = κ·√(ρ·A·c_T/2)`,
   SkyFlow-Dynamics `rotor_drag_hforce`). The current tables are multiplicative masks;
   this needs a power-law table.
2. The rotor-speed limits must scale too (constant tip speed gives Ω ∝ 1/λ). They are
   airframe scalars today; only the ceiling is per-world (the battery-sag trait).
3. Replace the `air_prop` and `inertia` limits by targets when this mode is on:
   thrust-to-weight and torque-to-inertia ranges, solved for the factors.
4. Sticks mode cannot use it as it stands: one Betaflight tune does not fly a 30 g and
   a 2 kg vehicle. Motors mode only, or one tune per size band.
5. Validate the scaling laws against the two measured airframes before use: scale the
   crazyflie row to the racer's mass and compare with the racer row.

### Notes for whoever picks this up

- L2F parameters are in normalized command units, and its files record no rotor-speed
  scale. They cannot seed `k_d·Ω` or `I_rot·Ω` terms.
- The size-deviation draw passes `(-range, range)` to a `(mean, std)` normal sampler.
  As written it is centred on −0.1, not zero. Read, not executed.
- L2F's `CLOSED_FORM` motor option takes the raw action as the rotor setpoint
  (SkyFlow-Dynamics finding F-31). Do not copy that mapping.
