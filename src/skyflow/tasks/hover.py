"""Hover — THE canonical hover benchmark (Vicon state-based station-keeping).

Take-off-and-reposition. Worlds spawn ON THE GROUND at a canonical pad (idle rotors —
the policy must lift off), then fly to and hold a goal that is RANDOM per episode AND
**resampled every ``resampling_time_s``** within the episode (repositioning). Because
the obs is target-RELATIVE (``rel_pos = goal − pos``), randomising the goal is equivalent
to randomising the spawn *position*, so the randomisation is applied on the goal side
only and the spawn stays canonical (no xy scatter). Repositioning is strictly richer
than a random spawn:
it trains the approach→decelerate→settle transient repeatedly, from an already-hovering
state, which a one-shot random spawn never exercises.

An optional AIRBORNE recovery mode (``spawn_ground_frac`` < 1) starts some worlds in
the air with random altitude/velocity/tilt (rotors pre-spun to hover) for tumble
recovery; it is OFF by default (repositioning already generates varied in-flight states,
so it is not required for a stable hover). The inner rate loop is closed by Betaflight
(acro), so the policy commands rate setpoints and never observes the rates it produces.

Reward design (a bounded-kernel shaping standard for quadrotor station-keeping):

    pos    ``w_p·exp(λ_p·dist)``   broad bounded positive hold bonus. A bounded kernel
                                   (not a raw −‖err‖ cost) keeps the value scale tame
                                   for PPO and removes the early-termination incentive
                                   of all-negative rewards.
  + hold   ``w_h·exp(λ_h·dist)``   tight at-target bonus (large |λ|, ~0 beyond a few cm):
                                   sharpens the last centimetres with a dedicated bonus
                                   rather than by steepening the broad kernel (which
                                   would flatten its far-field gradient).
  + prog   ``w_g·(dist₋₁² − dist²)`` dense approach pull (a telescoping Δdistance²
                                   progress term): episode sum telescopes to
                                   ``w_g·(spawn_dist² − final_dist²)`` — a control-rate-
                                   invariant Δ, so unlike the shaped rate terms it is NOT
                                   ×dt. Lifts a grounded drone off the floor and pulls it
                                   to each goal; the square strengthens the far pull and
                                   flattens the near-goal gradient (no fight at the point).
  + yaw    ``w_y·exp(λ_y·|yaw|)``  pick the reference heading (with a fixed yaw=0 spawn this
                                   anchors north). Does NOT stop a spin — see yawrate.
  − vel    ``w_v·‖vel‖``           settle instead of orbiting the point.
  − rate   ``w_ω·‖ω/π‖``           damp body-rate jitter (Molchanov 2019; Eschmann 2024).
  − yawrate``w_yr·(r/π)²``         penalise YAW SPIN directly (r = body yaw rate). A spinning
                                   drone sweeps every heading for the same average yaw bonus,
                                   so only a rate cost holds heading when yaw authority is
                                   uncapped (~8.7 rad/s through Betaflight). Quadratic: cheap
                                   for small heading corrections, steep for a fast spin.
  The pos/hold/yaw/vel/rate/yawrate terms are × dt (per-second shaping); the progress
  term is a per-step Δ and stays un-scaled.

Action smoothness is deliberately NOT a reward term — it is CAPS regularisation at the
trainer (``algo.action_smoothness_weight``, Mysore 2021). The crash penalty is an
env-level terminal (``env.crash_penalty``), replacing the step reward on any terminal.

Observation (19, WORLD NED — exactly what a Vicon rig gives, nothing it doesn't):
    rel_pos(3, scaled, clip ±1) · lin_vel(3, scaled, clip ±1)
    · rot_matrix(9, body→world) · last_action(4)
with additive uniform measurement noise on pos/vel/rotation (Vicon jitter). No gyro,
no motor speeds — deployable against mocap alone.

Terminal: leaving the absolute lab safe box (``±safe_half_xy``, altitude
``[safe_alt_lo, safe_alt_hi]``) is a task crash; the env's generic ground/tumble crash
set guards the floor. "At target" (``dist < hold_radius``) is a metric, never a terminal.

NOTE on the frame: altitude is floor-referenced (0 = floor). If the Vicon origin sits
mid-air, shift ``spawn_alt_m``/``target_alt_*``/``safe_alt_*`` at deploy — ``rel_pos``
is origin-agnostic so the policy transfers regardless.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import plant
from .base import ObsSpec, TaskEval, finalize_obs


def _qmul_wxyz(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product of two batched wxyz quaternions [..,4] (a∘b = a then b)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


class HoverTaskState(NamedTuple):
    """Per-world carried sub-state: the current hover target (NED; resampled on the
    ``resampling_time_s`` timer), the last commanded action (obs tail), and a per-world
    step counter that drives the goal-resample timing."""

    setpoint: jax.Array     # [F, 3] NED hover target (random per episode + resampled)
    last_action: jax.Array  # [F, 4] most-recent commanded action (obs tail)
    step: jax.Array         # [F] int32 step counter (goal-resample clock)


class HoverTask:
    """Canonical take-off-and-reposition benchmark (see module docstring)."""

    act_dim = 4
    term_labels = ("term_oob", "hold_rate")
    success_terminates = False   # holding is continuous; only crash/timeout ends it

    def __init__(
        self, *, control_freq: float, act_hist: int, stack: int, vision: bool,
        # -- spawn: CANONICAL ground pad by default (no xy scatter — the random goal
        #    supplies the position variety, since the obs is target-relative). Set
        #    spawn_ground_frac < 1 to add an AIRBORNE recovery mode (random
        #    alt/vel/tilt, rotors pre-spun to hover); spawn_rando_scale shrinks every
        #    recovery spread toward the pad (0 → deterministic ground start). Pad
        #    centre = (spawn_north, spawn_east). --
        spawn_north_m: float = 0.0, spawn_east_m: float = 0.0,
        spawn_radius_m: float = 0.0,                       # ±x/y pad scatter (0 = canonical, no scatter)
        spawn_alt_m: tuple[float, float] = (0.6, 1.4),     # AIRBORNE start-altitude range (m)
        spawn_vel_mps: float = 0.3,                        # airborne random velocity magnitude
        spawn_tilt_rad: float = 0.2,                       # airborne random tilt from level
        spawn_yaw_rad: float = math.pi,                    # random heading (both modes)
        spawn_ground_frac: float = 1.0,                    # fraction on the ground (1 = pure take-off)
        spawn_air_motor_norm: float = -0.186,              # airborne pre-spin motor state (≈ racer hover W)
        spawn_rando_scale: float = 1.0,                    # master: shrink recovery spreads toward the pad
        # -- goal (NED), sampled per episode AND resampled every resampling_time_s. A
        #    degenerate box (half_xy=0, alt_lo==alt_hi) = a FIXED goal (no resample):
        #    that is the deploy / deterministic-eval mode. --
        target_half_xy_m: float = 1.0,
        target_alt_lo_m: float = 0.5, target_alt_hi_m: float = 1.5,
        resampling_time_s: float = 3.0,                    # in-episode goal resample period (0 = fixed goal)
        # -- lab SAFETY ZONE: 12 ft × 12 ft floor (½·12 ft = 1.8288 m half-width),
        #    3.0 m usable ceiling; leaving it → task_crash. The physical room extends
        #    beyond the zone. --
        safe_half_xy_m: float = 1.8288,
        safe_alt_lo_m: float = 0.0, safe_alt_hi_m: float = 3.0,
        # -- viz-only OUTER reference zone (drawn in the training plot, not a boundary):
        #    the 20 ft × 20 ft × 4 m room margin. Cosmetic. --
        viz_outer_half_xy_m: float = 3.048,   # 20 ft / 2
        viz_outer_alt_lo_m: float = 0.0, viz_outer_alt_hi_m: float = 4.0,
        # -- "at target" hold metric --
        hold_radius_m: float = 0.1,
        # -- reward shaping (rate weights are pre-dt, scaled by dt at runtime;
        #    prog_weight multiplies a per-step Δdistance and is NOT dt-scaled). --
        pos_weight: float = 2.0, pos_lambda: float = -3.0,
        hold_bonus_weight: float = 1.0, hold_bonus_lambda: float = -50.0,
        prog_weight: float = 0.5,
        yaw_weight: float = 0.01, yaw_lambda: float = -10.0,
        vel_weight: float = 0.05, rate_weight: float = 0.0002,
        yaw_rate_weight: float = 0.0,        # dedicated yaw-spin penalty -w_yr·(r/π)² (0 = off)
        # -- obs scales (per-axis; the vertical axis is treated apart) --
        rel_pos_xy_scale: float = 1.0 / 3.0, rel_pos_z_scale: float = 1.0,
        lin_vel_xy_scale: float = 0.1, lin_vel_z_scale: float = 0.3,
        # -- additive measurement noise (Vicon jitter; raw sensor units) --
        obs_noise_pos_m: float = 0.02, obs_noise_vel_mps: float = 0.05,
        obs_noise_rot: float = 0.02,
        # -- camera knobs the platform always forwards; unused (state-only task).
        #    Kept in sync with the real BETAFPV C03 (Air75 II Racer) for consistency. --
        cam_height: int = 64, cam_width: int = 64,
        cam_fov_x_deg: float = 99.0, cam_fov_y_deg: float = 79.8,
        cam_mount_pitch_deg: float = -25.0,
        cam_offset_body: tuple[float, float, float] = (0.02, 0.0, -0.02),  # FRD: 2cm fwd, 2cm up
        cam_supersample: int = 2,
        cam_pitch_jitter_deg: float = 0.0, cam_roll_jitter_deg: float = 0.0,  # parity (unused; state-only)
        marker_ahead_m: float = 2.0,
        marker_inner_half: tuple[float, float] = (0.30, 0.30),
        marker_frame_width: float = 0.10,
        mask_noise_scale: float = 0.0,
        mask_noise_hold: int = 16,
        mask_outer_grow_m: tuple[float, float] = (0.0, 0.0),
        # accepted for platform parity (the env forwards camera + sensor-rate knobs
        # to every task); unused — hover is state-only (Vicon), no mask/IMU obs.
        cam_hz: float | None = None,
        cam_hz_jitter: float = 0.0,
        obs_stale_prob: float = 0.0,
        # IMU-response DR (per-world accel/gyro gain + additive noise); platform
        # parity — accepted and unused, hover reads Vicon state, not the IMU obs.
        imu_scale_dr: float = 0.0,
        accel_noise_std: float = 0.0,
        gyro_noise_std: float = 0.0,
        obs_rgb: bool = False,        # platform parity (RGB vision knob); unused — hover is state-only
        obs_retina: bool = False,     # platform parity (gen-6 retina knob); unused — hover is state-only
        mask_unet_ckpt: str | None = None,   # platform parity (gate-seg U-Net obs arm); unused — no mask obs
        motor_perm: tuple[int, int, int, int] = (0, 1, 2, 3),   # platform parity (env forwards it); unused — hover has no motor-echo obs
    ) -> None:
        if bool(vision):
            raise ValueError(
                "hover is a Vicon state-based task (no vision); set env.vision=false")
        self.vision = False
        self.image_shape: tuple[int, int, int] | None = None
        self.control_freq = float(control_freq)
        self._dt = 1.0 / float(control_freq)     # reward integrates over dt

        # spawn pad centre (NED north/east; ground starts sit on the floor z=0). Every
        # recovery spread is pre-scaled by spawn_rando_scale toward the pad here, so
        # spawn() just samples the (already-shrunk) box. ground_frac blends toward 1.0
        # as the scale → 0, so scale=0 is a clean deterministic ground start.
        self._spawn_pad = jnp.array([float(spawn_north_m), float(spawn_east_m)], jnp.float32)
        ss = float(spawn_rando_scale)
        self._sp_radius = float(spawn_radius_m) * ss
        _ac = 0.5 * (float(spawn_alt_m[0]) + float(spawn_alt_m[1]))
        self._sp_alt_lo = _ac + ss * (float(spawn_alt_m[0]) - _ac)
        self._sp_alt_hi = _ac + ss * (float(spawn_alt_m[1]) - _ac)
        self._sp_vel = float(spawn_vel_mps) * ss
        self._sp_tilt = float(spawn_tilt_rad) * ss
        self._sp_yaw = float(spawn_yaw_rad) * ss
        self._sp_ground_frac = 1.0 - ss * (1.0 - float(spawn_ground_frac))
        self._sp_air_motor = float(spawn_air_motor_norm)

        self._tgt_half_xy = float(target_half_xy_m)
        self._tgt_alt_lo = float(target_alt_lo_m)
        self._tgt_alt_hi = float(target_alt_hi_m)
        # goal-resample clock: period in steps. Disabled for a FIXED (degenerate) goal
        # — that is deploy / deterministic-eval, where the setpoint is externally fixed.
        self._resample_period = int(round(float(resampling_time_s) * self.control_freq))
        _box_random = (self._tgt_half_xy > 0.0) or (self._tgt_alt_lo != self._tgt_alt_hi)
        self._resample = bool(self._resample_period > 0 and _box_random)

        # A FIXED goal (degenerate box) gets a marker in the training viz; a
        # random/resampled goal can't be drawn as one point, so viz_goal stays None.
        self.viz_goal = ((0.0, 0.0, self._tgt_alt_lo) if not _box_random else None)

        self._safe_half_xy = float(safe_half_xy_m)
        self._safe_alt_lo = float(safe_alt_lo_m)
        self._safe_alt_hi = float(safe_alt_hi_m)
        self._hold_radius = float(hold_radius_m)

        # Two nested zones drawn in the training viz (display frame: north, east,
        # ALTITUDE m). INNER = the crash boundary (from the safe params; 12 ft cube,
        # 0→3 m). OUTER = a cosmetic room-margin reference (20 ft, 0→4 m) and, being
        # the outermost geometry, sets a steady view extent for the uirevision camera.
        self.viz_inner_zone = (-self._safe_half_xy, self._safe_half_xy,
                               -self._safe_half_xy, self._safe_half_xy,
                               self._safe_alt_lo, self._safe_alt_hi)
        _oh = float(viz_outer_half_xy_m)
        self.viz_outer_zone = (-_oh, _oh, -_oh, _oh,
                               float(viz_outer_alt_lo_m), float(viz_outer_alt_hi_m))

        self._pos_w, self._pos_l = float(pos_weight), float(pos_lambda)
        self._hold_w, self._hold_l = float(hold_bonus_weight), float(hold_bonus_lambda)
        self._prog_w = float(prog_weight)
        self._yaw_w, self._yaw_l = float(yaw_weight), float(yaw_lambda)
        self._vel_w = float(vel_weight)
        self._rate_w = float(rate_weight)
        self._yaw_rate_w = float(yaw_rate_weight)

        self._rp_xy, self._rp_z = float(rel_pos_xy_scale), float(rel_pos_z_scale)
        self._lv_xy, self._lv_z = float(lin_vel_xy_scale), float(lin_vel_z_scale)
        self._obs_scale = jnp.array(
            [self._rp_xy, self._rp_xy, self._rp_z,
             self._lv_xy, self._lv_xy, self._lv_z], jnp.float32)     # [6] pos+vel scales

        # per-element additive-noise std on the SCALED obs (raw × obs_scale), zero on
        # the rotation-matrix tail and the action (known exactly).
        n_pos, n_vel, n_rot = float(obs_noise_pos_m), float(obs_noise_vel_mps), float(obs_noise_rot)
        noise = jnp.concatenate([
            jnp.array([n_pos * self._rp_xy, n_pos * self._rp_xy, n_pos * self._rp_z,
                       n_vel * self._lv_xy, n_vel * self._lv_xy, n_vel * self._lv_z], jnp.float32),
            jnp.full((9,), n_rot, jnp.float32),      # rotation-matrix entries
            jnp.zeros((4,), jnp.float32),            # last action: policy knows it exactly
        ])
        self._noise_vec = noise
        self._noise_any = bool(jnp.any(noise > 0.0))

        # obs / privileged layout: rel_pos(3) lin_vel(3) rot_matrix(9) action(4) = 19.
        # State-mode task, so the actor obs IS the privileged state (asymmetric critic
        # is a no-op without vision); one layout is the single source of truth.
        self.OBS_LAYOUT = ObsSpec(
            [("rel_pos", 3), ("lin_vel", 3), ("rot_matrix", 9), ("action", 4)])
        self.priv_layout = self.OBS_LAYOUT
        self.obs_dim = self.OBS_LAYOUT.dim              # 19
        self.priv_dim = self.priv_layout.dim            # 19

    # -- goal sampling -------------------------------------------------------

    def _sample_setpoint(self, key: jax.Array, n: int) -> jax.Array:
        """Sample [n,3] NED goals: x,y ~ U(±target_half_xy) about the origin,
        z ~ U(alt_lo, alt_hi). A degenerate box returns the fixed goal."""
        kx, ky, kz = jax.random.split(key, 3)
        tx = jax.random.uniform(kx, (n,), minval=-1.0, maxval=1.0) * self._tgt_half_xy
        ty = jax.random.uniform(ky, (n,), minval=-1.0, maxval=1.0) * self._tgt_half_xy
        alt = jax.random.uniform(kz, (n,), minval=self._tgt_alt_lo, maxval=self._tgt_alt_hi)
        return jnp.stack([tx, ty, -alt], axis=1)      # NED (down = -alt)

    # -- spawn ---------------------------------------------------------------

    def spawn(self, key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
        """Per-world spawn. By default (spawn_ground_frac=1, spawn_radius=0) every world
        starts on the GROUND at the pad, idle rotors — a canonical take-off. With
        spawn_ground_frac < 1 the remainder start AIRBORNE (random alt/vel/tilt, rotors
        pre-spun to ~hover) for recovery. Idle rotors are motor-state −1 (W=0), NOT 0
        (that is half-throttle W=1500). Returns (plant_state [n,17], pos_ned [n,3])."""
        kg, kxy, kyaw, kz, kv, kphi, kang = jax.random.split(key, 7)
        on_ground = jax.random.uniform(kg, (n, 1)) < self._sp_ground_frac
        xy = self._spawn_pad + jax.random.uniform(kxy, (n, 2), minval=-1.0, maxval=1.0) * self._sp_radius

        # random heading about the NED down-axis (wxyz), shared by both modes
        hh = 0.5 * jax.random.uniform(kyaw, (n, 1), minval=-1.0, maxval=1.0) * self._sp_yaw
        z1 = jnp.zeros_like(hh)
        q_yaw = jnp.concatenate([jnp.cos(hh), z1, z1, jnp.sin(hh)], axis=-1)

        # airborne: random alt / velocity / tilt about a random horizontal axis
        alt = jax.random.uniform(kz, (n, 1), minval=self._sp_alt_lo, maxval=self._sp_alt_hi)
        vel_air = jax.random.uniform(kv, (n, 3), minval=-1.0, maxval=1.0) * self._sp_vel
        phi = jax.random.uniform(kphi, (n, 1), minval=0.0, maxval=2.0 * jnp.pi)
        ang = jax.random.uniform(kang, (n, 1), minval=0.0, maxval=self._sp_tilt)
        s2 = jnp.sin(0.5 * ang)
        q_tilt = jnp.concatenate([jnp.cos(0.5 * ang), s2 * jnp.cos(phi), s2 * jnp.sin(phi),
                                  jnp.zeros_like(ang)], axis=-1)
        q_air = _qmul_wxyz(q_yaw, q_tilt)                     # yaw then tilt

        gnd_pos = jnp.concatenate([xy, jnp.zeros((n, 1), jnp.float32)], axis=-1)   # z_ned=0 (floor)
        air_pos = jnp.concatenate([xy, -alt], axis=-1)                            # z_ned=-alt
        pos_ned = jnp.where(on_ground, gnd_pos, air_pos)
        vel_ned = jnp.where(on_ground, 0.0, vel_air)
        quat_ned = jnp.where(on_ground, q_yaw, q_air)
        rates_ned = jnp.zeros((n, 3), jnp.float32)
        motors = jnp.broadcast_to(
            jnp.where(on_ground, -1.0, self._sp_air_motor), (n, 4))   # −1 = idle (W=0)

        pos_p, vel_p, quat_p, rates_p = plant.pose_from_ned(pos_ned, vel_ned, quat_ned, rates_ned)
        state = plant.make_state(pos_p, vel_p, quat_p, rates_p, motors)
        return state, pos_ned

    def init(self, key: jax.Array, n: int, plant_state: jax.Array,
             pos_ned: jax.Array) -> HoverTaskState:
        """Fresh per-episode state: sample the first goal, zero the action tail and the
        resample clock. Called on every (auto-)reset."""
        return HoverTaskState(
            setpoint=self._sample_setpoint(key, n),
            last_action=jnp.zeros((n, 4), jnp.float32),
            step=jnp.zeros((n,), jnp.int32))

    # -- observation ---------------------------------------------------------

    def _obs_core(self, plant_state: jax.Array, setpoint: jax.Array, last_action: jax.Array
                  ) -> jax.Array:
        """The noise-free 19-vector (world NED): scaled+clipped rel_pos & lin_vel,
        the body→world rotation matrix, and the last commanded action."""
        pos_ned, vel_ned, quat_ned, _ = plant.pose_ned(plant_state)
        f = plant_state.shape[0]
        rel = setpoint - pos_ned                              # world-NED target error
        posvel = jnp.concatenate([rel, vel_ned], axis=1) * self._obs_scale
        posvel = jnp.clip(posvel, -1.0, 1.0)
        rot = plant.rot_matrix(quat_ned).reshape(f, 9)        # body→world
        return jnp.concatenate([posvel, rot, last_action], axis=1)

    def observe(self, plant_state: jax.Array, task_state: HoverTaskState,
                act_buf: jax.Array, key: jax.Array,
                params: jax.Array | None = None, *,
                fresh_spawn: bool = False,
                substeps: jax.Array | None = None) -> tuple[jax.Array, HoverTaskState]:
        """Advance the resample clock, resample the goal when it is due (so the obs
        reflects the goal the next reward will use), then build the obs. ``params``
        is unused (Vicon-state task — no synthesised IMU); ``fresh_spawn`` and ``substeps``
        are ignored (no per-step integrator worth skipping, and no IMU wire to model at a
        rate above the control loop). All three are accepted for the platform
        Task.observe contract."""
        n = plant_state.shape[0]
        step = task_state.step + 1
        setpoint = task_state.setpoint
        if key is not None and self._resample:
            key, ksp = jax.random.split(key)
            due = (jnp.mod(step, self._resample_period) == 0)[:, None]   # [n,1]
            setpoint = jnp.where(due, self._sample_setpoint(ksp, n), setpoint)

        last_action = act_buf[:, 0]                           # most-recent commanded stick frame
        obs = self._obs_core(plant_state, setpoint, last_action)
        if key is not None and self._noise_any:               # Vicon measurement jitter
            obs = obs + jax.random.uniform(key, obs.shape, minval=-1.0, maxval=1.0) * self._noise_vec
        task_state = task_state._replace(
            setpoint=setpoint, last_action=last_action, step=step)
        return finalize_obs(obs), task_state

    def privileged_state(self, plant_state: jax.Array, task_state: HoverTaskState
                         ) -> jax.Array:
        """Noise-free obs (= actor obs in state mode). Only consulted by the asymmetric
        critic, a no-op here; provided for the :class:`Task` contract."""
        return self._obs_core(plant_state, task_state.setpoint, task_state.last_action)

    # -- reward / events -----------------------------------------------------

    def evaluate(self, prev_pos_ned: jax.Array, plant_state: jax.Array,
                 task_state: HoverTaskState) -> TaskEval:
        pos_ned, vel_ned, quat_ned, rates_ned = plant.pose_ned(plant_state)
        # plant.safe_norm, not jnp.linalg.norm: the ground spawn has EXACTLY zero
        # velocity/rates, where norm's gradient is NaN — the first BPTT (APG) update
        # would NaN before the drone ever moves. Forward values are identical.
        dist = plant.safe_norm(pos_ned - task_state.setpoint, axis=1)
        prev_dist = plant.safe_norm(prev_pos_ned - task_state.setpoint, axis=1)
        speed = plant.safe_norm(vel_ned, axis=1)

        # NED heading (yaw about down): reward holding the reference heading (0 = spawn)
        w, x, y, z = quat_ned[:, 0], quat_ned[:, 1], quat_ned[:, 2], quat_ned[:, 3]
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        r_pos = self._pos_w * jnp.exp(self._pos_l * dist)               # broad hold bonus
        r_hold = self._hold_w * jnp.exp(self._hold_l * dist)            # tight at-target bonus
        r_yaw = self._yaw_w * jnp.exp(self._yaw_l * jnp.abs(yaw))       # heading pick (north)
        r_vel = -self._vel_w * speed                                    # settle (don't orbit)
        r_rate = -self._rate_w * plant.safe_norm(rates_ned / math.pi, axis=1)  # damp body rates
        # dedicated YAW-SPIN penalty (quadratic on body yaw rate r = rates_ned[:,2]). The
        # heading bonus can't stop a spin — a spinning drone sweeps every heading for the
        # same average bonus — so this rate cost is what holds heading with yaw authority
        # left uncapped. Quadratic: cheap for small heading corrections, steep for a spin.
        r_yaw_rate = -self._yaw_rate_w * (rates_ned[:, 2] / math.pi) ** 2
        # dense approach pull (telescoping Δdistance² progress term); the env
        # resets prev_pos_ned to the fresh spawn, so no teleport artefact at reset.
        r_prog = self._prog_w * (prev_dist ** 2 - dist ** 2)
        reward = (r_pos + r_hold + r_yaw + r_vel + r_rate + r_yaw_rate) * self._dt + r_prog

        # out of the lab safe area (absolute 12 ft cube around origin) → task_crash;
        # the env replaces the step reward with -crash_penalty on any terminal.
        alt = -pos_ned[:, 2]
        oob = ((jnp.abs(pos_ned[:, 0]) > self._safe_half_xy)
               | (jnp.abs(pos_ned[:, 1]) > self._safe_half_xy)
               | (alt < self._safe_alt_lo) | (alt > self._safe_alt_hi))
        settled = dist < self._hold_radius
        return TaskEval(
            reward=reward.astype(jnp.float32), success=settled, task_crash=oob,
            info={"settled": settled, "oob": oob})

    def scalar_metrics(self, plant_state: jax.Array, task_state: HoverTaskState,
                       ep_reach: jax.Array | None = None) -> dict[str, jax.Array]:
        pos_ned, vel_ned, _, _ = plant.pose_ned(plant_state)
        dist = jnp.linalg.norm(pos_ned - task_state.setpoint, axis=1)
        return {"hover_err_m": dist.mean(),
                "speed_mps": jnp.linalg.norm(vel_ned, axis=1).mean()}

    def fpv(self, plant_state: jax.Array, task_state: HoverTaskState, n: int) -> jax.Array:
        return jnp.zeros((n, 1, 1), jnp.float32)   # state-only task; no FPV
