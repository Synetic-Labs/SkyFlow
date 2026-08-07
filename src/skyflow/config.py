"""Hydra structured configs for the functional SkyFlow env + its tasks.

``SkyFlowConfig`` (``env`` group) is the *platform*: firmware, plant/airframe,
domain randomization, disturbances, transport latency, and the camera/vision
sensor pipeline — everything shared across objectives. It carries NO objective
params.

The *objective* is a separate top-level ``task`` group (sibling of ``env``/``algo``):
``SkyFlowTaskConfig`` subclasses, one per task, holding only that task's params.
Select and tune them at the root:

    python -m <your_trainer> env=skyflow task=hover task.target_alt_hi_m=2.0

``make_skyflow`` reads the ``env`` config for the platform and the chosen ``task``
config for the objective; ``task.name`` picks the implementation in
:func:`~skyflow.tasks.base.build_task`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class SkyFlowConfig:
    name: str = "skyflow"
    paradigm: str = "functional"     # must match the algo's `declares`
    # APG/BPTT: supported ONLY with control="motors" (pure-JAX plant, gradients flow
    # action → plant RK4 → reward); control="sticks" (real Betaflight, no gradient
    # path) rejects it. Vision mode = visual BPTT (mask is input-only for gradients,
    # stop_gradient'ed in the vision task's observe). PPO ignores it.
    differentiable: bool = False

    fleet: int = 1024                # parallel worlds (= firmware instances)
    # action interface: "sticks" = policy AETR through real Betaflight, the firmware
    # a whoop-class quad flies (needs the firmware extra + GPU); "motors" = DIRECT
    # per-motor commands into the plant, no firmware — the SkyDreamer-paper control
    # axis, and the matching target for a simulator twin with a raw-motor RL
    # interface. Pure JAX, so "motors" also needs no cudaflight.
    control: str = "sticks"
    control_hz: float = 90.0         # policy rate (SkyDreamer); firmware runs at 1 kHz
    max_ep: int = 900
    stuck_after: int = 150
    # randomize each world's initial episode phase (steps ~ U[0, max_ep)) so max_ep
    # truncations decorrelate across the fleet — brax APG's scramble_times analog
    # (without it the whole fleet truncates in the SAME BPTT window). PPO-safe too.
    scramble_ep_phase: bool = False

    # airframe coefficients — one entry per fleet drone (params.AIRFRAME_PARAMS):
    # "air75_ii_racer" (BETAFPV Air75 II Racer, Vicon-sysid-fitted)
    airframe: str = "air75_ii_racer"

    # reward / bounds (platform terminal handling)
    crash_penalty: float = 0.0       # paper zeroes reward on crash; extra penalty optional
    # -- reward-side action-calmness bonuses (algo-INDEPENDENT replacement for PPO's
    #    loss-side algo.action_magnitude_weight / action_smoothness_weight, which only
    #    the PPO trainers implement — see the trainer's guard). Positive-Gaussian
    #    form (never net-negative: with zero-on-crash, a persistent per-step penalty
    #    makes dying beat surviving — the observed disarm-suicide failure).
    #    Both default OFF so existing runs are bit-identical.
    #
    #    CALIBRATED 2026-07-27 (Monte Carlo over the sampled-action distribution at
    #    PPO's σ=0.368, grounded in a measured thrust curve: hover trim u=0.266 →
    #    a_hover=−0.468/motor). Two findings baked in:
    #    1. The magnitude term needs the DEADZONE: a bare exp(−‖a‖²/c) is centred on
    #       a=0 = mid-throttle ≈ 2.5× weight thrust, so it paid a hard 2.5g maneuver
    #       (0.62) TWICE what it paid hover (0.31) — backwards for "calm". With
    #       d=0.6 the bonus is flat across all normal flight (hover 0.68, racing
    #       0.86, 2.5g 0.95) and collapses only toward the rails (full punch 0.08,
    #       one railed motor 0.40): a pure anti-saturation guard, motion-neutral.
    #    2. The smoothness scale must exceed the exploration noise floor
    #       E‖Δa‖² = 8σ² ≈ 1.08: at the old 0.25 a perfectly smooth policy could
    #       transmit only 10% of the bonus (noise ate it); at 1.0 it transmits 42%
    #       while thrash gets 3% — 13:1 discrimination that survives sampling.
    #
    #    THE BAND IS PHASE-SPECIFIC (crawl first, racing later):
    #    - CRAWL: centre the band on HOVER actuation (hover trim u=0.266 →
    #      a=−0.468/motor) so SUSTAINED thrust above the crawl envelope decays the
    #      bonus. center=−0.468, deadzone=0.25, scale=0.5 transmits hover 0.77 /
    #      gentle descent 0.74 / 1.5× weight 0.70, discounts 2.5g to 0.35 and kills
    #      a full punch (0.00) — monotone against thrust, sharpening as σ anneals.
    #    - RACING (later): center=0.0, deadzone=0.6, scale=0.25 = the motion-neutral
    #      rail guard (hover 0.68 / racing 0.86 / 2.5g 0.95, punch 0.08).
    #
    #    RECOMMENDED CRAWL enable block:
    #        act_calm_weight: 0.02        act_smooth_weight: 0.05
    #        act_calm_center: -0.468      # hover trim of the flown airframe (params.py)
    #        act_calm_deadzone: 0.25      act_calm_scale: 0.5
    #    → calm-loiter income 0.037/step = 44% of the dense progress rate
    #    (0.083/step at 1.5 m/s — approaching strictly dominates hovering), ≤8% of a
    #    a gate-pass credit of ~110 per approach; sustained 2.5g forfeits 0.008/step, punch
    #    0.015/step. ⚠ Both terms are computed on SAMPLED actions, so they mildly pay
    #    for shrinking σ (measured +0.021/step for σ 0.368→0.2): watch policy entropy
    #    when enabling under PPO (consider ent_coef 0.003→0.005); SAC auto-α resists.
    act_calm_weight: float = 0.0     # bonus/step: w·exp(−‖max(|a−center|−d,0)‖²/scale)
    act_calm_center: float = 0.0     # actuation the band is centred on (crawl: −0.468 = hover)
    act_calm_deadzone: float = 0.6   # half-width of the free band (crawl: 0.25)
    act_calm_scale: float = 0.25     # roll-off sharpness beyond the band (crawl: 0.5)
    act_smooth_weight: float = 0.0   # bonus/step for a held action: w·exp(−‖Δa‖²/scale)
    act_smooth_scale: float = 1.0    # ‖Δa‖² scale; must clear the 8σ²≈1.1 noise floor
    bounds_xy_m: float = 20.0
    bounds_z_m: float = 8.0
    # near-ground tip-over tilt limit (rad): below GROUND_ALT, roll/pitch beyond this is
    # a ground collision. Raise (e.g. ~90°) so a hard-tilted take-off is not killed
    # before it clears the floor.
    ground_tilt_limit_rad: float = 1.0471975511965976   # math.pi / 3 (60°)

    # domain randomization + sim2real
    physics_rando_scale: float = 1.0
    disturbance_scale: float = 1.0
    # Individual disturbance channels (all scaled by disturbance_scale above). Declared here
    # (2026-07-28) because ``make.py`` has always forwarded them but the schema never carried
    # them, so in struct mode they could not be SET from a config at all —
    # ``env.disturb_poke_prob=...`` failed to compose. That matters most for the poke, which is
    # the one silently rate-coupled channel: a per-CONTROL-STEP probability, so the same value
    # delivers 3x fewer pokes per second of flight at 30 Hz than at 90 Hz and a rate change
    # MUST be able to rescale it. The others are already rate-correct (wind_tau_s is converted
    # through the step period in env.py, and the OU kick is derived from the decay), but a
    # config that can set one and not the rest is a trap of its own.
    disturb_wind_acc: float = 0.6        # m/s² OU wind acceleration std (world frame)
    disturb_wind_tau_s: float = 0.5      # OU correlation time (s)
    disturb_poke_prob: float = 3.0e-4    # ⚠ PER CONTROL STEP — scale with control_hz
    disturb_poke_vel_mps: float = 0.5    # poke magnitude — linear velocity kick (m/s)
    disturb_poke_rate_rps: float = 3.0   # poke magnitude — body-rate kick (rad/s)
    act_delay_min_steps: int = 1
    act_delay_steps: int = 4          # sets action_hist width = (act_delay_steps+1)*4, fixed
    act_delay_nominal_steps: int | None = None   # deterministic delay for the clean sim-to-sim
                                                 # eval; None → round mid of [min, max] (env.py)
    warmstart_steps: int = 0          # frozen-podium launch warm-up steps prepended per episode
                                      # (GRU warms on the countdown obs); 0 = off. See env.py.
    obs_frame_stack: int = 2          # mask frames stacked (vision only)
    vision: bool = True               # True = mask+gyro+motors+action_hist (CNN); False = state MLP smoke
    obs_rgb: bool = False             # True = RGB image obs (black sky, white floor, #ff4000 gates, C=stack*3); False = 1-ch coverage
    obs_retina: bool = False          # pool the coverage frames onto a coarse 8x12 retina grid
                                      # (occ + in-cell centroid per cell) — image block becomes a flat
                                      # (8,12,stack*ch*3) MLP vector; requires vision_encoder=false, aug 0
    asymmetric_critic: bool = False   # actor=vision obs, critic=privileged state (vision only); no-op otherwise

    # camera / vision sensor (SkyDreamer 64x64 mask) — the drone's ONE camera,
    # shared by every vision task; the task renders its own geometry through it.
    cam_height: int = 64
    cam_width: int = 64
    # BETAFPV C03 (the Air75 II's camera): a 160° 4:3 fisheye, modeled in the pinhole
    # renderer as its rectilinear equivalent ~99°H/79.8°V.
    cam_fov_x_deg: float = 99.0
    cam_fov_y_deg: float = 79.8
    # canopy tilts the camera UP 25° (indoor lab); down-positive convention → negative.
    cam_mount_pitch_deg: float = -25.0
    # camera position on the body, FRD metres (C03: 2 cm forward, 2 cm up; for a
    # spec camera that sits AT the body origin — set [0, 0, 0]).
    cam_offset_body: list[float] = field(default_factory=lambda: [0.02, 0.0, -0.02])
    cam_supersample: int = 2
    # camera EXTRINSIC domain randomization (per-episode mounting variation): each
    # world draws a fixed camera TILT (pitch, about the lens right axis) and ROTATION
    # (roll, about the optical axis) offset from U[-this, +this] degrees at reset, so
    # the policy is robust to a canopy/camera that isn't mounted at the nominal angle.
    # 0 = exact nominal mount. Small values (a few degrees) model real build tolerance.
    cam_pitch_jitter_deg: float = 0.0
    cam_roll_jitter_deg: float = 0.0
    # camera refresh rate (Hz). None → a fresh mask every control step; a float
    # (e.g. 30.0) delivers the mask at that rate over the faster control loop —
    # a typical deployment camera is 30 Hz — holding each frame on the bus between
    # refreshes. cam_hz_jitter = per-world fractional jitter on the inter-frame
    # interval (sensor-timing domain randomization; 0 = exact rate).
    cam_hz: float | None = None
    cam_hz_jitter: float = 0.0
    # proprioception (gyro + motor) telemetry staleness: fleet-max per-world
    # probability a control-step read repeats the previous sample — models a deploy
    # IMU (~86 Hz) / actuator-status (~75 Hz) duplicate/stale wire samples read at
    # the faster control rate. Each world draws its gyro & motor hold-prob
    # independently from U[0, this]; 0 = every read fresh. Vision (gate) obs only.
    obs_stale_prob: float = 0.0
    # ⚠ ``obs_imu_substeps`` (a fixed K=4 sub-step IMU stack on a decimation/K grid) was
    # REMOVED. It and ``task.imu_hz``
    # solved the same problem — "a 30 Hz control step throws away most of an 86 Hz IMU wire"
    # — and cannot both be applied. ``task.imu_hz`` won: the wire is free-running, so it
    # reproduces the measured 2/3/4-packet histogram (the K-grid + Bernoulli-hold model puts
    # 12.6% of steps at <=1 unique sample, which the wire never does) and, unlike the
    # obs-only stack, its arrival TIMESTAMPS feed the filter block's propagate.
    # IMU-RESPONSE DR (a measured sim-to-sim transfer gap):
    # per-world U[1±imu_scale_dr] gain on the accel/gyro obs models the deploy sim's
    # dynamics answering the same commands stronger/weaker than the twin; the noise
    # stds are per-sample Gaussian sensor noise in PHYSICAL units (m/s², rad/s),
    # applied to fresh samples before the staleness hold. Vision (gate) obs only.
    imu_scale_dr: float = 0.0
    accel_noise_std: float = 0.0
    gyro_noise_std: float = 0.0
    mask_noise_scale: float = 0.0
    # max CAMERA FRAMES a mask-noise artifact persists before it is redrawn (each
    # artifact lives ~U{1..hold} frames per world; 1 = i.i.d. per frame).
    mask_noise_hold: int = 16
    mask_outer_grow_m: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # frozen gate-segmentation U-Net (checkpoint dir) applied to the mask in the obs path,
    # after mask_noise — the "same perception function on both sims" bench arm.
    mask_unet_ckpt: str | None = None

    # Betaflight motor index → SkyDreamer W1..W4. Derived from the plant torque
    # signs (W1=FR, W2=RR, W3=FL, W4=RL) vs Betaflight QUADX corners (0=RR,1=FR,
    # 2=RL,3=FL): crazyflow's (1,0,2,3) with the left rotors swapped (SkyDreamer
    # orders them FL-then-RL). Verified against the Racer sysid data (per-motor
    # sign patterns + props-out yaw sign — physics doc §Sysid).
    motor_perm: list[int] = field(default_factory=lambda: [1, 0, 3, 2])

    # firmware
    eeprom: str | None = None        # Betaflight config blob; None = stock
    device_index: int = 0
    settle_ms: int = 0
    # firmware SITL backend: "gpu" (cudaflight, fleets >= 3 — training) | "cpu"
    # (libcpuflight, small interactive fleets — teleop/eval). No auto-fallback;
    # a 1-world eval MUST use "cpu" (cudaflight can't create < 3 instances).
    firmware_backend: str = "gpu"


# -- task group (top-level `task`): one schema per SkyFlow objective ----------

@dataclass
class SkyFlowTaskConfig:
    """Base for a SkyFlow objective. ``name`` selects the task implementation."""

    name: str = MISSING


@dataclass
class HoverTaskConfig(SkyFlowTaskConfig):
    """THE canonical hover benchmark (see tasks/hover.py for the design rationale).

    Canonical ground spawn + RANDOM, in-episode-RESAMPLED goal (repositioning);
    Vicon-only obs (rel_pos, lin_vel, rotation matrix, last action — no gyro/motors);
    bounded-kernel reward + tight at-target bonus; lab safe-area terminal. Requires
    ``env.vision=false`` (state-based, Vicon deployable). Action smoothness is CAPS at
    the trainer (``algo.action_smoothness_weight``), not a reward term."""

    name: str = "hover"
    # spawn: CANONICAL ground pad by default (no xy scatter — the random goal supplies
    # the position variety, since the obs is target-relative). Set spawn_ground_frac < 1
    # to add an AIRBORNE recovery mode; spawn_rando_scale shrinks recovery spreads toward
    # the pad (0 = deterministic ground start). Pad centre = (spawn_north, spawn_east).
    spawn_north_m: float = 0.0
    spawn_east_m: float = 0.0
    spawn_radius_m: float = 0.0                # ±x/y pad scatter (0 = canonical, no scatter)
    spawn_alt_m: list[float] = field(default_factory=lambda: [0.6, 1.4])  # AIRBORNE alt range
    spawn_vel_mps: float = 0.3                 # airborne random velocity magnitude
    spawn_tilt_rad: float = 0.2                # airborne random tilt from level
    spawn_yaw_rad: float = 3.141592653589793   # random heading (both modes)
    spawn_ground_frac: float = 1.0             # fraction on the ground (1 = pure take-off; <1 adds recovery)
    spawn_air_motor_norm: float = -0.186       # airborne pre-spin motor state (≈ racer hover W)
    spawn_rando_scale: float = 1.0             # master: shrink recovery spreads toward the pad (0 = deterministic)
    # goal box about the origin (NED), sampled per episode AND resampled every
    # resampling_time_s (repositioning). Degenerate (half_xy=0, alt_lo==alt_hi) = a
    # FIXED goal with no resample (deploy / deterministic eval).
    target_half_xy_m: float = 1.0
    target_alt_lo_m: float = 0.5
    target_alt_hi_m: float = 1.5
    resampling_time_s: float = 3.0   # in-episode goal resample period (0 = fixed goal)
    # lab SAFE AREA — 12 ft cube ≈ ±1.8288 m around origin; leaving it = task_crash.
    # Altitude is floor-referenced (0 = floor); map to your Vicon origin at deploy.
    safe_half_xy_m: float = 1.8288
    safe_alt_lo_m: float = 0.0
    safe_alt_hi_m: float = 3.0
    # viz-only OUTER reference zone (20 ft × 20 ft × 4 m room margin, drawn gray in
    # the training plot). Cosmetic — leaving the inner safe zone already ends the episode.
    viz_outer_half_xy_m: float = 3.048     # 20 ft / 2
    viz_outer_alt_lo_m: float = 0.0
    viz_outer_alt_hi_m: float = 4.0
    hold_radius_m: float = 0.1       # "at target" hold metric
    # reward shaping (rate weights are pre-dt, scaled by 1/control_freq at runtime;
    # prog_weight scales a per-step Δdistance and is NOT dt-scaled). hold_bonus is the
    # tight at-target sharpening term on top of the broad pos kernel.
    pos_weight: float = 2.0          # broad hold bonus w_p·exp(λ_p·dist)
    pos_lambda: float = -3.0
    hold_bonus_weight: float = 1.0   # tight at-target bonus w_h·exp(λ_h·dist)
    hold_bonus_lambda: float = -50.0
    prog_weight: float = 0.5         # dense approach pull w_g·(dist₋₁−dist) (telescoping Δdistance)
    yaw_weight: float = 0.01         # heading hold w_y·exp(λ_y·|yaw|)
    yaw_lambda: float = -10.0
    vel_weight: float = 0.05         # -w_v·speed (settle)
    rate_weight: float = 0.0002      # -w_ω·‖ω/π‖ (damp body rates)
    yaw_rate_weight: float = 0.0     # -w_yr·(r/π)² dedicated yaw-spin penalty (0 = off; hover enables)
    # observation scales (vertical axis treated apart)
    rel_pos_xy_scale: float = 0.3333333333333333
    rel_pos_z_scale: float = 1.0
    lin_vel_xy_scale: float = 0.1
    lin_vel_z_scale: float = 0.3
    # additive Vicon measurement noise (raw sensor units; 0 disables)
    obs_noise_pos_m: float = 0.02
    obs_noise_vel_mps: float = 0.05
    obs_noise_rot: float = 0.02
