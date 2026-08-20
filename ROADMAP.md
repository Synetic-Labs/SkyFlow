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
