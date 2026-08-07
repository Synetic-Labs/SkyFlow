"""SkyFlow env — SkyDreamer analytic plant behind real Betaflight, one platform,
many tasks.

End-to-end-JAX, PPO-trainable functional env (``fleet`` / ``jax_reset`` /
``jax_step``) that flies the SkyDreamer analytic quadrotor model (``plant.py``)
under a real Betaflight flight controller (``firmware_fleet.py``). This class is
the *platform*: it owns the firmware fleet, the plant, domain randomization,
in-flight disturbances, transport latency, the fused ``lax.scan`` rollout, the
generic crash set (disarm / flyaway / ground-collision) and the in-jit
auto-reset. The *objective* — spawn, observation, reward, task terminals — is a
:class:`~skyflow.tasks.base.Task` injected at construction
and selected by ``env.task`` (``gate`` | ``hover``); see ``tasks/`` and
``make.py``.

Control loop, one policy step (``control_hz``, default 90 Hz) = ``decimation``
firmware+plant substeps at 1 kHz, fused in one ``lax.scan`` on the GPU:

    policy(sticks AETR) ─► Betaflight PID+mixer (1 ms) ─► motors[0,1]
        ─► SkyDreamer plant RK4 (1 ms) ─► body state ─► synth IMU ─► next substep

Two action interfaces, selected by ``control``:

* ``"sticks"`` (default) — the loop above: AETR through real Betaflight, the
  same firmware a whoop-class quad flies. Requires the ``cudaflight`` wheel + a
  CUDA device; NOT faithful to SkyDreamer on the control axis (the paper uses
  direct motor commands). See README.md for the calibration that makes the
  closed loop match a real Air75 II.
* ``"motors"`` — DIRECT per-motor commands: the 4 policy actions in [-1, 1]
  map to the plant's ``U ∈ [0, 1]`` (zero-order hold across the substeps), no
  firmware in the loop. This is the SkyDreamer-paper control axis, and the
  matching target for any simulator twin whose RL interface takes raw per-motor
  commands. Pure JAX: needs no cudaflight and runs anywhere. ``motor_perm``
  then maps the ACTION slot order to plant W1..W4 (e.g. a ``[BR,BL,FR,FL]``
  slot order → ``[2, 0, 3, 1]``), and the disarm crash term never fires (there
  is no firmware to disarm).

Frames: the plant integrates in SkyDreamer's Z-up/FLU frame; tasks work in
NED/FRD, so the platform converts the plant pose per step (``plant.pose_ned``)
and hands NED pose to the task.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

from . import plant
from .params import airframe_params, randomization_scale
from .tasks.base import build_task, finalize_obs

if TYPE_CHECKING:  # firmware is imported lazily — see _require_firmware below
    from .firmware_fleet import CpuFirmwareFleet, FirmwareFleet  # type: ignore[import-not-found]


def _require_firmware() -> Any:
    """Import the Betaflight firmware fleet, or explain why it is not here.

    ``control="motors"`` (direct per-motor thrust) is pure JAX and is what this package
    ships. The stick/rate control modes close the loop through real Betaflight firmware
    compiled to an XLA custom call, which needs a prebuilt ``cudaflight`` wheel that is
    not part of this distribution. The plant, tasks and rendering are unaffected — only
    the firmware-in-the-loop control modes are.
    """
    try:
        from . import firmware_fleet  # type: ignore[attr-defined]
    except ImportError as exc:
        raise NotImplementedError(
            "control='sticks'/'rate' runs Betaflight firmware in the loop, which needs "
            "the `cudaflight` firmware wheel that ships separately from open-source "
            "SkyFlow. Use control='motors' (direct per-motor, pure JAX) instead.") from exc
    return firmware_fleet


def _align_board_yaw_from_eeprom(eeprom_path: str | None) -> float:
    """Read ``align_board_yaw`` (deg) from the eeprom's sibling Betaflight CLI dump.

    A drone's directory ships a ``*.txt`` CLI dump beside its
    ``.eeprom``; the firmware yaws the gyro/accel by ``align_board_yaw`` (whoop FCs
    mount rotated to the arms), so the seam must undo that. Returns 0.0 when there is
    no eeprom or no such line (stock firmware defaults)."""
    if not eeprom_path:
        return 0.0
    import re
    from pathlib import Path
    pat = re.compile(r"\s*set\s+align_board_yaw\s*=\s*(-?\d+(?:\.\d+)?)")
    for txt in sorted(Path(eeprom_path).parent.glob("*.txt")):
        try:
            for line in txt.read_text(errors="ignore").splitlines():
                m = pat.match(line)
                if m:
                    return float(m.group(1))
        except OSError:
            continue
    return 0.0


class SkyFlowState(NamedTuple):
    """Carried state (a pytree threaded through the PPO loop). Everything here is
    task-agnostic except ``task`` — the injected task's opaque sub-pytree (mask
    history, hover setpoint, …), blended on auto-reset by :func:`_tree_where`."""
    fw_blob: jax.Array       # [F*stride] u8 firmware instance blob (donated); empty in control="motors"
    fw_state: jax.Array      # [F*stateSize] u8 bfFlight_t state (donated); empty in control="motors"
    plant: jax.Array         # [F, 17] SkyDreamer state (Z-up/FLU)
    params: jax.Array        # [F, 35] per-world DR'd plant coefficients
    key: jax.Array           # [2] PRNG key (spawn / DR / mask noise)
    prev_pos_ned: jax.Array  # [F, 3] last step's NED position (crossing + progress)
    steps: jax.Array         # [F] int32 control steps since reset
    airborne: jax.Array      # [F] bool per-agent airborne latch
    act_buf: jax.Array       # [F, Dmax+1, 4] recent commanded sticks (newest@0)
    delay_idx: jax.Array     # [F] int32 per-agent transport delay
    wind: jax.Array          # [F, 3] OU wind acceleration (world NED)
    term_ema: jax.Array      # [7] EMA of per-step OUTCOME fractions (metric only):
                             # [disarm, flyaway, ground, task_crash, success, timeout, lost].
                             # MUTUALLY EXCLUSIVE and masked exactly like ``done``, so for
                             # a task whose success ends the episode (gate) these PARTITION
                             # the episode-end rate: their sum == train/done_frac. That is
                             # what makes ``timeout`` (the step-budget truncation) readable
                             # directly instead of as done_frac minus the terminal reasons.
    task: Any                # task-owned sub-pytree (opaque to the platform)
    ep_reach: jax.Array      # [F] gates cleared in each world's LAST COMPLETED episode
                             # (persists across the auto-reset, like term_ema) — the
                             # per-episode gate metrics read this, not a live snapshot.
    # -- per-episode aggregates, same latch-on-done pattern as ``ep_reach`` ----------
    # A per-STEP reward mean cannot answer "was the episode worth more than a shorter
    # one?" — the question any speed/time-budget shaping turns on. These carry the real
    # per-episode numbers instead of the geometric 1/done_frac estimate, which is biased
    # upward under a hard step cap (it can report a mean length ABOVE max_ep).
    ep_ret_acc: jax.Array    # [F] running undiscounted return of the IN-PROGRESS episode
    ep_ret: jax.Array        # [F] return of each world's LAST COMPLETED episode (latched)
    ep_len: jax.Array        # [F] that episode's length in control steps (latched)
    miss_ema: jax.Array      # [2] slow EMA of (Σ Chebyshev offset at gate-plane crossings,
                             # # crossings) per step; the RATIO is the mean miss margin in
                             # metres. Two accumulators because the mean is conditional on a
                             # rare event — a plain EMA of the offset would be diluted by
                             # every step with no crossing. Stays [0,0] for tasks that never
                             # report a crossing (hover), see ``jax_step``.


def _tree_where(mask: jax.Array, new_tree: Any, old_tree: Any) -> Any:
    """Per-agent select across a task sub-pytree: ``new`` where ``mask`` (a [F]
    bool), else ``old``. Reshapes the mask to each leaf's rank so it broadcasts."""
    def sel(a: jax.Array, b: jax.Array) -> jax.Array:
        m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
        return jnp.where(m, a, b)
    return jax.tree_util.tree_map(sel, new_tree, old_tree)


class SkyFlowEnv:
    """num_envs independent worlds; one drone each. Objective set by ``task``."""

    AIRBORNE_ALT = 0.3
    MAX_SPEED = 30.0
    MAX_RATE = 50.0
    GROUND_ALT = 0.5
    GROUND_VZ = 1.0            # NED descent (down positive) threshold, m/s
    TILT_LIMIT = math.pi / 3   # 60° ground-collision tilt

    def __init__(
        self, num_envs: int, *,
        # objective, resolved through the task registry (skyflow.tasks.register_task).
        # "hover" ships with the package; downstream projects register their own.
        task: str = "hover",
        # action interface: "motors" = direct per-motor plant commands, no firmware —
        # pure JAX, differentiable, and the SkyDreamer-paper control axis. "sticks" =
        # AETR through real Betaflight, which needs the `cudaflight` wheel that ships
        # separately from this distribution (see _require_firmware).
        control: str = "motors",
        # differentiable step for APG/BPTT: only control="motors" qualifies (the
        # whole substep rollout is pure JAX); "sticks" runs real Betaflight with
        # no gradient path and rejects the flag. Vision mode = "visual BPTT": the
        # mask is INPUT-only for gradients (the coverage render is piecewise-
        # constant in pose and explicitly stop_gradient'ed in gate.observe), so
        # credit assignment flows through the dynamics + proprio/action tail while
        # the CNN still trains through the direct policy pathway.
        differentiable: bool = False,
        # randomize each world's initial episode phase (steps ~ U[0, max_ep)) so
        # truncation resets stay decorrelated across the fleet — brax APG's
        # scramble_times: without it every world truncates at max_ep in the SAME
        # BPTT window (a periodic fleet-wide gradient cut). Ground-spawned worlds
        # scrambled past stuck_after truncate on step 1 (one wave, then staggered
        # forever) — harmless. Useful for PPO too; off by default.
        scramble_ep_phase: bool = False,
        control_hz: float = 90.0,
        max_ep: int = 900, stuck_after: int = 150,
        airframe: str = "air75_ii_racer",  # fleet drone (a key of params.AIRFRAME_PARAMS)
        control_freq_override: float | None = None,
        crash_penalty: float = 0.0,      # paper zeroes reward on crash; extra penalty optional
        # Reward-side action-calmness bonuses (see config.py for semantics + the
        # 2026-07-27 calibration: deadzone so hover isn't penalized relative to
        # mid-throttle, smoothness scale above the exploration-noise floor).
        act_calm_weight: float = 0.0, act_calm_center: float = 0.0,
        act_calm_deadzone: float = 0.6, act_calm_scale: float = 0.25,
        act_smooth_weight: float = 0.0, act_smooth_scale: float = 1.0,
        bounds_xy_m: float = 20.0, bounds_z_m: float = 8.0,
        # near-GROUND tip-over tilt limit (rad): below GROUND_ALT, roll/pitch beyond
        # this = a ground collision. Default 60°; raise (e.g. ~90°) so a hard-tilted
        # TAKEOFF (hover spawns on the floor) isn't killed before it climbs past GROUND_ALT.
        ground_tilt_limit_rad: float = math.pi / 3,
        disable_crash: bool = False,     # testing escape hatch: never terminate on ANY crash
                                         # (disarm / ground / flyaway / task_crash) — drone can't crash

        # -- domain randomization (plant coefficients, Table III brackets)
        physics_rando_scale: float = 1.0,
        # -- in-flight disturbances
        disturbance_scale: float = 1.0,
        disturb_wind_acc: float = 0.6,   # m/s² OU wind std (world)
        disturb_wind_tau_s: float = 0.5,
        disturb_poke_prob: float = 3e-4,
        disturb_poke_vel_mps: float = 0.5,
        disturb_poke_rate_rps: float = 3.0,
        # -- transport latency + mask frame stack. The delay draw is DR, gated by
        # physics_rando_scale like the plant params: at scale 0 every world pins to
        # act_delay_nominal_steps (deterministic — what the clean sim-to-sim
        # instruments assume); at scale 1 it samples uniform{min..max} per episode.
        # (Pre-2026-07-23 the draw was UNconditional — "DR-off" runs still carried a
        # hidden random delay, and a fixed PRNGKey pinned it to one edge of the band.)
        act_delay_min_steps: int = 1,
        act_delay_steps: int = 4,
        act_delay_nominal_steps: int | None = None,  # None → round mid of [min, max]
        # -- 3s launch WARM-START (0 = off). Prepends `warmstart_steps` FROZEN podium steps to
        #    every episode: the drone is held on the pad (plant state frozen), sensors+vision run
        #    live with DR (varying obs), act-buf held IDLE (NOT the policy's computed action — that
        #    feedback saturates the launch, measured a0->+1.0), reward 0 / done False. The recurrent
        #    GRU warms on the realistic countdown obs, so the policy LAUNCHES from a warmed state
        #    (matching a deploy that runs the policy through the countdown). Gated internally by
        #    steps<0 (no new state field). Overrides scramble_ep_phase when >0.
        warmstart_steps: int = 0,
        obs_frame_stack: int = 2,
        # -- observation mode (meaning is task-specific; see the task docstrings):
        #  vision=True  → mask(k=obs_frame_stack)+gyro+motors+action_hist (CNN);
        #  vision=False → fully-observed state (MLP).
        # Defaults False to match the default ``hover`` task, which is state-based; a
        # vision task (a gate racer) sets it True. A task that cannot serve the mode it
        # is handed raises at construction rather than silently observing something else.
        vision: bool = False,
        # -- asymmetric critic: actor sees the vision obs, critic sees the task's
        #  fully-observed privileged state (``privileged_obs``). No-op without vision
        #  (state mode is already fully observed). The trainer threads two obs streams.
        asymmetric_critic: bool = False,
        # -- RGB analytic vision: render an RGB image (BLACK sky, WHITE floor, RED-ORANGE
        #    #ff4000 gates) instead of 1-channel coverage → image_shape C = stack*3.
        #    A cheap analytic ray-cast (gate + ground plane); vision only.
        obs_rgb: bool = False,
        # -- retina obs: image block pooled to the 8x12 cell grid (see the gate
        #    task's obs_retina). Forwarded to the task; exposed as `self.obs_retina`
        #    so a trainer can refuse encoder/aug combos that make no sense.
        obs_retina: bool = False,
        # -- vision camera (SkyDreamer: 64x64 mask) — forwarded to the task.
        #    Defaults = the BETAFPV C03 (Air75 II): 160° 4:3 fisheye → ~99°H/79.8°V
        #    rectilinear, canopy tilted UP 25° (down-positive → negative).
        cam_height: int = 64, cam_width: int = 64,
        cam_fov_x_deg: float = 99.0, cam_fov_y_deg: float = 79.8,
        cam_mount_pitch_deg: float = -25.0,
        cam_offset_body: tuple[float, float, float] = (0.02, 0.0, -0.02),  # FRD: 2cm fwd, 2cm up
        cam_supersample: int = 2,
        # per-episode camera extrinsic DR (deg): mount tilt (pitch) + rotation (roll) jitter.
        cam_pitch_jitter_deg: float = 0.0,
        cam_roll_jitter_deg: float = 0.0,
        mask_noise_scale: float = 0.0,
        mask_noise_hold: int = 16,
        mask_outer_grow_m: tuple[float, float] = (0.0, 0.0),
        # frozen gate-seg U-Net applied to the mask INSIDE the obs path (checkpoint dir).
        # Mirrors the deploy seam frame->filter->U-Net->policy, so training and the
        # deployment sim see the same function's output (the "U-Net on both sims" bench
        # arm). Applied after mask_noise; None = raw mask.
        mask_unet_ckpt: str | None = None,
        # camera refresh rate: None = a fresh mask every control step; a float
        # (e.g. 30.0) downsamples the mask stream to that rate over the faster
        # control loop (a typical deployment camera is 30 Hz). cam_hz_jitter =
        # per-world fractional jitter on the inter-frame interval (sensor-timing DR).
        cam_hz: float | None = None,
        cam_hz_jitter: float = 0.0,
        # proprioception (gyro + motor) telemetry staleness: fleet-max per-world
        # probability a control-step read repeats the last sample (a deploy IMU/actuator
        # wire can be ~75-86 Hz unique). 0 = every read fresh at control_hz. Vision only.
        obs_stale_prob: float = 0.0,
        # IMU-response DR: per-world U[1±imu_scale_dr] gain on the accel/gyro obs
        # (the deploy sim's dynamics answering the same commands stronger/weaker
        # than the twin) + per-sample Gaussian sensor noise in physical units
        # (m/s², rad/s). Vision only; all 0 = off. See tasks/gate.py.
        imu_scale_dr: float = 0.0,
        accel_noise_std: float = 0.0,
        gyro_noise_std: float = 0.0,
        # -- motor mapping command slot -> SkyDreamer W1..W4. sticks: Betaflight
        #    corners 0=RR,1=FR,2=RL,3=FL -> (1,0,3,2) (= crazyflow (1,0,2,3) w/
        #    L rotors swapped). motors: the ACTION slot order; a twin whose raw-motor
        #    slots are [BR,BL,FR,FL] uses (2,0,3,1) — see params.py.
        motor_perm: tuple[int, int, int, int] = (1, 0, 3, 2),
        # -- firmware
        eeprom: str | None = None,
        device_index: int = 0, settle_ms: int = 0,
        # "gpu" (cudaflight, fleets >= 3) | "cpu" (libcpuflight — small
        # interactive fleets; jits but is not vmappable). No auto-fallback.
        firmware_backend: str = "gpu",
        # Compensation for the eeprom's `align_board_yaw`: whoop FCs are mounted rotated
        # to the arms (Air75 = -45), so the firmware yaws the gyro/accel by that before
        # the PID. The plant synthesises sensors in the clean arm frame, so we apply the
        # INVERSE rotation to line the firmware's control axes back up with the plant's
        # motor axes — else roll/pitch come out coupled ~45° (roll stick tilts the drone
        # corner-first), which also breaks sim2real for a trained policy. Default None =
        # AUTO-READ align_board_yaw from the eeprom's sibling Betaflight CLI dump (the
        # per-drone directory convention); pass a float to override, 0.0 to disable.
        eeprom_board_align_yaw_deg: float | None = None,
        # -- task-specific knobs (gate geometry / hover setpoint / spawn …)
        **task_kwargs: Any,
    ) -> None:
        self.num_envs = int(num_envs)
        self.n_drones = 1
        self.control_hz = float(control_hz)
        self.decimation = max(1, round(1000.0 / self.control_hz))
        self.dt = 1.0 / 1000.0
        self.control_freq = (control_freq_override
                             if control_freq_override is not None
                             else 1000.0 / self.decimation)
        self.max_ep = int(max_ep)
        self.stuck_after = int(stuck_after)
        self.crash_penalty = float(crash_penalty)
        self._act_calm_w = float(act_calm_weight)
        self._act_calm_center = float(act_calm_center)
        self._act_calm_dead = float(act_calm_deadzone)
        self._act_calm_scale = float(act_calm_scale)
        self._act_smooth_w = float(act_smooth_weight)
        self._act_smooth_scale = float(act_smooth_scale)
        self.disable_crash = bool(disable_crash)
        self.bounds_xy = float(bounds_xy_m)
        self.bounds_z = float(bounds_z_m)
        self.TILT_LIMIT = float(ground_tilt_limit_rad)   # instance overrides the class default

        # -- action interface + firmware fleet ------------------------------
        self.control = str(control)
        if self.control not in ("sticks", "motors"):
            raise ValueError(f"unknown control {control!r} (use 'sticks' or 'motors')")
        # differentiable validation lives HERE (the single authority, before the
        # firmware import) so sticks+differentiable errors cleanly even on a
        # machine without cudaflight.
        self.differentiable = bool(differentiable)
        self.scramble_ep_phase = bool(scramble_ep_phase)
        if self.differentiable and self.control != "motors":
            raise NotImplementedError(
                "skyflow differentiable=True needs control='motors' (pure-JAX plant, "
                "BPTT-able); control='sticks' runs real Betaflight firmware with no "
                "gradient path — use algo=ppo, or switch to env.control=motors for APG.")
        # vision + differentiable is ALLOWED (visual BPTT): the mask block is
        # stop_gradient'ed in gate.observe (input-only — zero pose-gradient render),
        # so the window gradient flows through dynamics + the proprio/action tail
        # and the CNN trains through the direct policy pathway.
        self.firmware: FirmwareFleet | CpuFirmwareFleet | None = None
        if self.control == "motors":
            # direct per-motor: pure JAX, no firmware in the loop at all.
            if eeprom:
                raise ValueError(
                    "control='motors' bypasses Betaflight entirely — an eeprom would be "
                    "silently ignored; unset env.eeprom (or use control='sticks').")
            self.fleet = self.num_envs
        else:
            _fw = _require_firmware()
            if firmware_backend == "gpu":
                self.firmware = _fw.FirmwareFleet(
                    self.num_envs, device_index=device_index, settle_ms=settle_ms,
                    eeprom=eeprom)
            elif firmware_backend == "cpu":
                self.firmware = _fw.CpuFirmwareFleet(
                    self.num_envs, settle_ms=settle_ms, eeprom=eeprom)
            else:
                raise ValueError(
                    f"unknown firmware_backend {firmware_backend!r} (use 'gpu' or 'cpu')")
            self.fleet = self.firmware.fleet
            if self.firmware.act_dim != 4:
                raise ValueError(f"expected 4 stick channels, got {self.firmware.act_dim}")

        # -- plant params + motor mapping ----------------------------------
        base = airframe_params(airframe)
        self._params_nominal = base.to_array()            # [36]
        self.motor_perm = jnp.asarray(motor_perm, jnp.int32)

        # Board-align compensation: rotate synth gyro+accel by -(eeprom align_board_yaw)
        # about yaw so the firmware's post-align control axes match the plant motor axes.
        board_yaw = (_align_board_yaw_from_eeprom(eeprom)
                     if eeprom_board_align_yaw_deg is None else eeprom_board_align_yaw_deg)
        self.eeprom_board_align_yaw_deg = float(board_yaw)
        self._sensor_rot = None
        if abs(board_yaw) > 1e-6:
            th = -math.radians(board_yaw)                    # inverse of the eeprom rotation
            c, s = math.cos(th), math.sin(th)
            self._sensor_rot = jnp.asarray(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], jnp.float32)
        self.physics_rando_scale = float(physics_rando_scale)

        # -- disturbances ---------------------------------------------------
        d = float(disturbance_scale)
        self._wind_acc = float(disturb_wind_acc) * d
        step_dt = 1.0 / self.control_hz
        self._wind_decay = float(math.exp(-step_dt / max(disturb_wind_tau_s, 1e-3)))
        self._wind_kick = float((1.0 - self._wind_decay ** 2) ** 0.5)
        self._poke_prob = float(disturb_poke_prob) * d
        self._poke_vel = float(disturb_poke_vel_mps)
        self._poke_rate = float(disturb_poke_rate_rps)

        # -- transport latency + frame stack --------------------------------
        self._act_delay_min = int(act_delay_min_steps)
        self._act_delay_max = int(act_delay_steps)
        self._act_delay_nom = (int(round((self._act_delay_min + self._act_delay_max) / 2))
                               if act_delay_nominal_steps is None
                               else int(act_delay_nominal_steps))
        if not self._act_delay_min <= self._act_delay_nom <= self._act_delay_max:
            raise ValueError(f"act_delay_nominal_steps {self._act_delay_nom} outside "
                             f"[{self._act_delay_min}, {self._act_delay_max}]")
        self._warmstart = max(0, int(warmstart_steps))         # frozen-podium launch warm-up steps
        self._act_hist = self._act_delay_max + 1
        self.vision = bool(vision)
        # only vision tasks stack frames; state mode is fully observed.
        self._stack = max(1, int(obs_frame_stack)) if self.vision else 1
        # asymmetric critic only makes sense with vision (state mode is already
        # fully observed, so actor obs == privileged obs).
        self.asymmetric_critic = bool(asymmetric_critic) and self.vision

        # -- objective (spawn / obs / reward / task terminals) --------------
        self.task = build_task(
            task,
            control_freq=self.control_freq, act_hist=self._act_hist,
            stack=self._stack, vision=self.vision, obs_rgb=obs_rgb,
            obs_retina=obs_retina,
            cam_height=cam_height, cam_width=cam_width,
            cam_fov_x_deg=cam_fov_x_deg, cam_fov_y_deg=cam_fov_y_deg,
            cam_mount_pitch_deg=cam_mount_pitch_deg, cam_offset_body=cam_offset_body,
            cam_supersample=cam_supersample,
            cam_pitch_jitter_deg=cam_pitch_jitter_deg, cam_roll_jitter_deg=cam_roll_jitter_deg,
            mask_noise_scale=mask_noise_scale,
            mask_noise_hold=mask_noise_hold, mask_outer_grow_m=mask_outer_grow_m,
            mask_unet_ckpt=mask_unet_ckpt,
            cam_hz=cam_hz, cam_hz_jitter=cam_hz_jitter,
            obs_stale_prob=obs_stale_prob, imu_scale_dr=imu_scale_dr,
            accel_noise_std=accel_noise_std, gyro_noise_std=gyro_noise_std,
            motor_perm=tuple(int(x) for x in motor_perm),
            **task_kwargs)

        # A task that models a sensor FASTER than the control loop (a deploy IMU wire
        # can deliver 2-4 packets per 30 Hz control step) asks for the plant's
        # intermediate 1 kHz states by declaring how many it wants. Off by default: the
        # rollout scan then discards them exactly as before, so nothing pays for this.
        self._substep_states = int(getattr(self.task, "substep_imu_slots", 0)) > 0

        # -- obs contract exposed to the trainer (from the task) --
        self.OBS_LAYOUT = self.task.OBS_LAYOUT
        self.obs_dim = self.OBS_LAYOUT.dim
        self.act_dim = 4
        self.priv_dim = int(self.task.priv_dim)
        if self.vision and self.task.image_shape is not None:
            # image shape the CNN policy reshapes the mask block into (channels-last).
            self.image_shape = self.task.image_shape
        # retina runs are MLP-only — the trainer reads this to refuse encoder/aug.
        self.obs_retina = bool(getattr(self.task, "_retina", False))
        # gate geometry (if any) for the trainer's 3D-trajectory viz overlay.
        for a in ("gate_center", "gate_right", "gate_up", "gate_half"):
            if hasattr(self.task, a):
                setattr(self, a, getattr(self.task, a))
        # The gate task folds TWO unrelated failures into one terminal:
        # ``task_crash = gate_miss | lost`` (a frame hit, vs the per-gate step budget /
        # range backstop expiring). They call for opposite fixes — aim vs time budget —
        # so under the racing labels we split them into their own outcome slots. Any
        # other task (incl. the gate task in hover mode, which relabels to term_lost)
        # keeps the single undivided ``task_crash`` slot.
        self._split_lost = getattr(self.task, "term_labels", ("", ""))[0] == "term_gate_miss"

    # -- domain randomization -------------------------------------------------

    def _sample_delay(self, key: jax.Array, n: int) -> jax.Array:
        """Per-world transport delay, gated by the master DR dial: scale 0 → the
        nominal (deterministic), scale 1 → uniform{min..max}; between, the draw
        shrinks toward the nominal (rounded interpolation, integer steps)."""
        u = jax.random.randint(key, (n,), self._act_delay_min,
                               self._act_delay_max + 1).astype(jnp.int32)
        s = jnp.clip(jnp.float32(self.physics_rando_scale), 0.0, 1.0)
        nom = jnp.int32(self._act_delay_nom)
        return (nom + jnp.round(s * (u - nom).astype(jnp.float32)).astype(jnp.int32))

    def _sample_params(self, key: jax.Array, n: int) -> jax.Array:
        factors = randomization_scale(key, n, self.physics_rando_scale)
        p = self._params_nominal[None, :] * factors            # [n, 35]
        # k (blend, idx 28) must stay in [0, 1]
        return p.at[:, 28].set(jnp.clip(p[:, 28], 0.0, 1.0))

    # -- firmware + plant rollout ---------------------------------------------

    @staticmethod
    def _ground_contact(st):
        """The world has a floor: never integrate through z=0 (plant frame is
        Z-up). Contact = clamp to the plane, kill the downward velocity, and
        damp horizontal velocity (kinetic friction, ~50 ms time constant —
        a sub-hover whoop grips the floor rather than skating; without this
        a tilted drone slides metres sideways while its takeoff throttle
        winds up). Without the clamp itself, a descent with crash-resets
        disabled passes straight through the ground and "falls forever";
        the ground-crash terms still fire off alt == 0 when enabled."""
        below = st[:, 2] < 0.0
        st = st.at[:, 2].set(jnp.maximum(st[:, 2], 0.0))
        st = st.at[:, 5].set(jnp.where(below, jnp.maximum(st[:, 5], 0.0), st[:, 5]))
        fric = jnp.where(below[:, None], 0.98, 1.0)          # per-1 ms substep
        return st.at[:, 3:5].set(st[:, 3:5] * fric)

    def _rollout(self, fw_blob, fw_state, plant_state, params, sticks):
        """decimation substeps of (synth IMU → firmware 1 ms → motors → plant RK4
        1 ms), one lax.scan. Returns (fw_blob, fw_state, plant_state, motors, armed)."""
        def sub(carry, _):
            blob, fwst, st = carry
            sensors = plant.synth_sensors(st, params)
            if self._sensor_rot is not None:                 # cancel eeprom board-align yaw
                R = self._sensor_rot
                sensors = (sensors.at[:, 0:3].set(sensors[:, 0:3] @ R.T)
                                  .at[:, 3:6].set(sensors[:, 3:6] @ R.T))
            blob, fwst, motors, armed = self.firmware.fw_step(blob, fwst, sticks, sensors)
            st = plant.step(st, motors, params, self.dt, self.motor_perm)
            st = self._ground_contact(st)
            return (blob, fwst, st), (motors, armed, st if self._substep_states else None)

        (fw_blob, fw_state, plant_state), (motors_seq, armed_seq, sub_seq) = jax.lax.scan(
            sub, (fw_blob, fw_state, plant_state), None, length=self.decimation)
        return fw_blob, fw_state, plant_state, motors_seq[-1], armed_seq[-1], sub_seq

    def _rollout_direct(self, plant_state, params, cmd):
        """decimation plant RK4 substeps under a zero-order-held DIRECT motor
        command ``cmd`` [F, 4] ∈ [0, 1] (the policy action, latched for the whole
        control period — raw-motor latch semantics). No firmware; the
        plant's own sqrt thrust curve + first-order motor lag are the only
        actuator dynamics.

        Returns ``(plant_state, sub_seq)``. ``sub_seq`` is ``[decimation, F, S]`` — EVERY
        1 ms plant state inside the control step, oldest first, the last being the end of
        the step (so ``sub_seq[-1] is`` the returned state). It is ``None`` unless the task
        asked for sub-step IMU (``task.substep_imu_slots > 0``), which keeps the
        single-sample path exactly as it was.

        Why: a control step spans ``decimation`` ms, and the policy used to see ONE IMU
        sample from the end of it. That is fine at 90 Hz (11 ms) but throws away most of
        the motion at 30 Hz (33 ms). A deploy IMU wire can deliver ~86 Hz of unique samples
        regardless of the control rate, so a 30 Hz step really carries 2-4 fresh ones.

        Why the WHOLE 1 ms sequence rather than K evenly spaced samples: the wire is
        free-running, so its packets land wherever its phase puts them, not on a
        ``decimation/K`` grid (see ``GateTask._imu_step``). The task picks the instants it
        needs out of this sequence at 1 ms resolution; handing it a pre-decimated grid would
        force the even spacing back in — and that spacing is the thing the accumulator
        exists to get rid of.

        This path is fully differentiable (the basis of ``differentiable=True``):
        gradients flow action → clip → U → plant RK4 → pose → reward, and the
        auto-reset spawns are key-derived (action-independent) so BPTT truncates
        cleanly at terminals. Benign zero-grad regions: the action clip, the
        inflow rail (plant), ground contact, and the terminal reward replacement
        in ``jax_step``."""
        def sub(st, _):
            st = self._ground_contact(plant.step(st, cmd, params, self.dt, self.motor_perm))
            return st, (st if self._substep_states else None)

        plant_state, sub_seq = jax.lax.scan(sub, plant_state, None, length=self.decimation)
        return plant_state, sub_seq

    def _disturb(self, plant_state, wind, key):
        """OU wind (world accel over one control step) + rare pokes on the plant."""
        kf, kp, kpv, kpw = jax.random.split(key, 4)
        n = plant_state.shape[0]
        wind = (self._wind_decay * wind
                + self._wind_kick * self._wind_acc * jax.random.normal(kf, (n, 3)))
        dt = 1.0 / self.control_hz
        vel = plant_state[:, 3:6] + wind * dt
        poke = jax.random.uniform(kp, (n, 1)) < self._poke_prob
        vel = vel + jnp.where(poke, jax.random.uniform(kpv, (n, 3), minval=-1.0, maxval=1.0)
                              * self._poke_vel, 0.0)
        rates = plant_state[:, 10:13] + jnp.where(
            poke, jax.random.uniform(kpw, (n, 3), minval=-1.0, maxval=1.0) * self._poke_rate, 0.0)
        plant_state = plant_state.at[:, 3:6].set(vel).at[:, 10:13].set(rates)
        return plant_state, wind

    # -- functional interface -------------------------------------------------

    def jax_reset(self, key: jax.Array) -> tuple[jax.Array, SkyFlowState]:
        n = self.fleet
        if self.firmware is None:                            # control="motors"
            fw_blob = fw_state = jnp.zeros((0,), jnp.uint8)
        else:
            fw_blob, fw_state = self.firmware.fresh_firmware_state()
        key, ksp, kpar, kdelay, kinit, kobs, kph = jax.random.split(key, 7)
        params = self._sample_params(kpar, n)
        plant_state, pos_ned = self.task.spawn(ksp, n)
        delay_idx = self._sample_delay(kdelay, n)
        act_buf = jnp.zeros((n, self._act_hist, 4), jnp.float32)
        wind = jnp.zeros((n, 3), jnp.float32)
        airborne = (-pos_ned[:, 2]) > self.AIRBORNE_ALT
        # scramble_ep_phase: stagger the initial episode phase so max_ep truncations
        # decorrelate across the fleet (auto-reset zeroes steps, so once staggered the
        # phases stay staggered). See the __init__ kwarg comment.
        if self._warmstart > 0:                                # launch warm-start: steps<0 = frozen podium
            steps = jnp.full((n,), -self._warmstart, jnp.int32)
        elif self.scramble_ep_phase:
            steps = jax.random.randint(kph, (n,), 0, self.max_ep).astype(jnp.int32)
        else:
            steps = jnp.zeros((n,), jnp.int32)

        task_state = self.task.init(kinit, n, plant_state, pos_ned)
        # Every world here IS a fresh spawn, so it takes the same path as the auto-reset
        # re-observe below — otherwise a world's first obs would depend on whether it arrived
        # via jax_reset or via an in-episode reset, which is exactly the kind of difference
        # nothing downstream would ever notice.
        obs, task_state = self.task.observe(
            plant_state, task_state, act_buf, kobs, params, fresh_spawn=True)
        zf = jnp.zeros((n,), jnp.float32)
        state = SkyFlowState(
            fw_blob, fw_state, plant_state, params, key, pos_ned,
            steps, airborne, act_buf, delay_idx, wind,
            jnp.zeros((7,), jnp.float32), task_state, zf,
            ep_ret_acc=zf, ep_ret=zf, ep_len=zf, miss_ema=jnp.zeros((2,), jnp.float32))
        return obs, state

    def jax_step(self, state: SkyFlowState, action: jax.Array
                 ) -> tuple[jax.Array, SkyFlowState, jax.Array, jax.Array, dict[str, Any]]:
        n = self.fleet
        (fw_blob, fw_state, plant_state, params, key, prev_pos_ned, steps,
         airborne, act_buf, delay_idx, wind, term_ema, task_state, ep_reach,
         ep_ret_acc, ep_ret, ep_len, miss_ema) = state

        # launch WARM-START: steps<0 → the frozen-podium countdown. Hold the plant, keep act-buf
        # IDLE (feed the APPLIED idle, not the policy's computed action — that feedback saturates
        # the launch), and later zero reward + block done. The GRU still warms via the recurrent
        # forward on the (DR'd) resting obs built below.
        warming = steps < 0                                    # [F] bool
        plant_before = plant_state

        # transport latency: push newest action, apply the delayed one
        prev_action = act_buf[:, 0]     # last step's policy action (pre-push newest slot)
        act_buf = jnp.concatenate([action[:, None, :], act_buf[:, :-1, :]], axis=1)
        act_buf = jnp.where(warming[:, None, None], jnp.zeros_like(act_buf), act_buf)   # idle while held
        cmd = act_buf[jnp.arange(n), delay_idx]

        if self.firmware is None:
            # direct per-motor: action [-1,1] -> plant command U [0,1]; nothing
            # can disarm (no firmware), so the disarm crash term never fires.
            plant_state, substeps = self._rollout_direct(
                plant_state, params, jnp.clip((cmd + 1.0) * 0.5, 0.0, 1.0))
            armed = jnp.ones((n,), bool)
        else:
            fw_blob, fw_state, plant_state, _motors, armed_u8, substeps = self._rollout(
                fw_blob, fw_state, plant_state, params, cmd)
            armed = armed_u8 != 0
        # [decimation, F, 17] when the task asked for sub-step IMU, else None. The plant
        # already integrates at 1 kHz; this is the only place those intermediate states are
        # visible, and a task that models a sensor faster than the control loop needs them.
        if substeps is not None:
            substeps = jnp.swapaxes(substeps, 0, 1)          # -> [F, decimation, 17]
        plant_state = jnp.where(warming[:, None], plant_before, plant_state)   # freeze on the pad
        if substeps is not None:        # the frozen podium must read frozen sub-samples too
            substeps = jnp.where(warming[:, None, None], plant_before[:, None], substeps)

        key, kdist = jax.random.split(key)
        if self._wind_acc > 0.0 or self._poke_prob > 0.0:
            plant_state, wind = self._disturb(plant_state, wind, kdist)

        pos_ned, vel_ned, quat_ned, rates_ned = plant.pose_ned(plant_state)
        alt = -pos_ned[:, 2]
        airborne = airborne | (alt > self.AIRBORNE_ALT)

        # task reward + events (NED frame; gyro = FRD body rates)
        ev = self.task.evaluate(prev_pos_ned, plant_state, task_state)
        # a task may advance its carried state inside evaluate (the only method
        # given prev_pos) — e.g. the gate task bumping the active-gate index on a
        # pass. Adopt it before the post-step observe so the next obs reflects it.
        if ev.task_state is not None:
            task_state = ev.task_state

        # generic (platform) crash set
        speed = jnp.linalg.norm(vel_ned, axis=1)
        rate_mag = jnp.linalg.norm(rates_ned, axis=1)
        grounded = alt <= 1e-4
        flyaway = ((alt > self.bounds_z) | (jnp.abs(pos_ned[:, 0]) > self.bounds_xy)
                   | (jnp.abs(pos_ned[:, 1]) > self.bounds_xy)
                   | (speed > self.MAX_SPEED) | (rate_mag > self.MAX_RATE))
        # ground collision (NED): low & (descending fast or over-tilted)
        w, x, y, z = quat_ned[:, 0], quat_ned[:, 1], quat_ned[:, 2], quat_ned[:, 3]
        roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
        over_tilt = (jnp.abs(roll) > self.TILT_LIMIT) | (jnp.abs(pitch) > self.TILT_LIMIT)
        ground = (alt < self.GROUND_ALT) & ((vel_ned[:, 2] > self.GROUND_VZ) | over_tilt)
        ground_any = ground | (airborne & grounded)

        crashed = (~armed) | (airborne & grounded) | flyaway | ground | ev.task_crash
        if self.disable_crash:                  # testing: drone can't crash (no crash terminal)
            crashed = jnp.zeros_like(crashed)
        crashed = crashed & (~warming)          # can't crash while held on the pad (warm-start)
        finished = ev.success & (~crashed)
        task_reward = ev.reward
        # Reward-side calm bonuses (config.act_calm/act_smooth, default OFF). Computed on
        # the RAW policy action stream (a=0 is mid-throttle in motors mode), positive-
        # Gaussian so a step is never net-negative (zero-on-crash would otherwise make
        # dying beat paying — the observed disarm-suicide failure). Part of the MDP, so
        # the behaviour survives an algo swap — unlike PPO's loss-side regularizers.
        if self._act_calm_w > 0.0:
            # Free band centred on `act_calm_center` (crawl: hover trim −0.468/motor,
            # so SUSTAINED thrust above the crawl envelope decays the bonus; racing
            # later: center 0 / wide band = a motion-neutral rail guard). A bare ‖a‖²
            # would centre the bonus on mid-throttle ≈ 2.5× weight thrust and pay
            # aggression over hover — backwards for crawl. Only the excess decays it.
            excess = jnp.maximum(
                jnp.abs(action - self._act_calm_center) - self._act_calm_dead, 0.0)
            task_reward = task_reward + self._act_calm_w * jnp.exp(
                -jnp.sum(jnp.square(excess), axis=-1) / self._act_calm_scale)
        if self._act_smooth_w > 0.0:
            task_reward = task_reward + self._act_smooth_w * jnp.exp(
                -jnp.sum(jnp.square(action - prev_action), axis=-1) / self._act_smooth_scale)
        reward = jnp.where(crashed, -self.crash_penalty, task_reward)
        reward = jnp.where(warming, 0.0, reward)   # no reward signal during the frozen warm-up

        steps = steps + 1
        # a met objective ends the episode only if the task says so (gate: yes; hover: no)
        end_success = finished & jnp.asarray(self.task.success_terminates)
        done_terminal = crashed | end_success
        trunc = (~done_terminal) & (
            (steps >= self.max_ep) | ((steps >= self.stuck_after) & ~airborne))
        done = (done_terminal | trunc) & (~warming)    # never end during the frozen warm-up
        # SPLIT flags for bootstrapped off-policy targets. An on-policy GAE loop only ever
        # needs the merged `done`, but a TD target must bootstrap through a TRUNCATION (the
        # episode was cut by the step budget, the value beyond it is real) and must NOT
        # bootstrap through a TERMINATION (crash / course complete — there is no beyond).
        # Merging them teaches the critic that running out of clock is worth -crash_penalty.
        # Mutually exclusive by construction: `trunc` already excludes `done_terminal`.
        terminated = done_terminal & (~warming)
        truncated = trunc & (~warming)

        # EMA of per-step OUTCOME fractions (diagnostic metric only):
        # [disarm, flyaway, ground, task_crash, success, timeout, lost] — slots 3,4 named
        # by the task. MUTUALLY EXCLUSIVE (a cascade in the same priority order as the
        # logger's outcome taxonomy) and masked by ~warming exactly like ``done``.
        # Both properties matter: the raw flags OVERLAP (a flyaway is usually also over-
        # tilted, so it fires `ground` too) and the unmasked ones fire during the frozen
        # warm-up when no episode can actually end — so the old un-cascaded sum could
        # exceed done_frac AND leave truncation unmeasurable. With the cascade, for a task
        # whose success terminates (gate) these sum EXACTLY to done_frac, which is what
        # makes `timeout` a real reading rather than a residual. (For hover, success does
        # NOT end the episode, so slot 4 is a state rate — its label says so, `hold_rate` —
        # and only slots 0-3 + 5 partition the ends.)
        live = ~warming
        o_disarm = (~armed) & live
        o_flyaway = flyaway & live & ~o_disarm
        o_ground = ground_any & live & ~o_disarm & ~o_flyaway
        prior = o_disarm | o_flyaway | o_ground
        if self._split_lost:      # gate racing: frame hit vs per-gate time budget (see __init__)
            o_crash = ev.info["gate_miss"] & live & ~prior
            o_lost = ev.info["lost"] & live & ~prior & ~o_crash
        else:
            o_crash = ev.task_crash & live & ~prior
            o_lost = jnp.zeros_like(o_crash)
        if self.disable_crash:    # testing: no crash terminal, so no crash outcome
            o_disarm = o_flyaway = o_ground = o_crash = o_lost = jnp.zeros_like(o_disarm)
        frac = jnp.stack([
            o_disarm.mean(), o_flyaway.mean(), o_ground.mean(), o_crash.mean(),
            (finished & live).mean(), truncated.mean(), o_lost.mean(),
        ]).astype(jnp.float32)
        term_ema = 0.99 * term_ema + 0.01 * frac

        # per-episode return + length, latched on done like ``ep_reach`` below. ``reward``
        # is the post-terminal-zeroing value the trainer actually optimises, so the latch
        # matches the objective rather than a reconstruction of it. ``steps`` is already
        # incremented and a warm-start seeds it at -warmstart, so at done it holds exactly
        # the number of LIVE control steps the episode ran.
        ep_ret_acc = ep_ret_acc + reward
        ep_ret = jnp.where(done, ep_ret_acc, ep_ret)
        ep_len = jnp.where(done, steps.astype(jnp.float32), ep_len)
        ep_ret_acc = jnp.where(done, 0.0, ep_ret_acc)

        # mean gate-plane miss margin, as a ratio of two slow EMAs (numerator = summed
        # Chebyshev offset over crossings, denominator = crossings). Decay 0.999 not 0.99:
        # a crossing is a rare per-step event, so the 100-step window the terminal EMA uses
        # would leave the ratio dominated by sampling noise. Keyed on the task REPORTING a
        # crossing — hover's info omits these, so its miss_ema stays [0,0].
        if "cross_cheby" in ev.info:
            cx = (ev.info["crossed"] & live).astype(jnp.float32)
            miss_ema = 0.999 * miss_ema + 0.001 * jnp.stack(
                [(ev.info["cross_cheby"] * cx).mean(), cx.mean()]).astype(jnp.float32)

        # per-episode gate progress: on done, latch gates CLEARED this episode =
        # (active_gate - start_gate) + 1 on a full-course success (active clamps at the
        # last gate). Full count — so a completion latches ``gates_remaining`` (the win),
        # which the from_start metric keeps and gates_cleared excludes (a completion is
        # exactly ep_reach == remaining). Latched BEFORE the auto-reset, so it survives.
        if hasattr(self.task, "ep_progress"):
            reach = self.task.ep_progress(task_state) + finished.astype(jnp.float32)
            ep_reach = jnp.where(done, reach, ep_reach)

        # build the next obs (advances any frame history) from the pre-reset state
        key, kobs = jax.random.split(key)
        obs, task_state = self.task.observe(
            plant_state, task_state, act_buf, kobs, params, substeps=substeps)
        # The TRUE s' of this transition, before the auto-reset below overwrites the done
        # agents' rows with their fresh spawn. On-policy PPO never needs it (it bootstraps
        # from the value function and masks at `done`), but an off-policy replay buffer
        # stores (s, a, r, s') literally — storing the post-reset obs would train the critic
        # on a teleport from the crash site to the podium. Surfaced in `info` so this stays
        # additive: `obs` (the return value) keeps its auto-reset semantics unchanged.
        final_obs = obs

        # in-jit auto-reset of done agents
        fmask = done.astype(jnp.uint8)
        m = fmask != 0
        if self.firmware is not None:
            fw_blob, fw_state = self.firmware.reset(fw_blob, fw_state, fmask)
        key, ksp, kpar, kdelay, kinit, kobs2 = jax.random.split(key, 6)
        rstate, rpos = self.task.spawn(ksp, n)
        rparams = self._sample_params(kpar, n)
        plant_state = jnp.where(m[:, None], rstate, plant_state)
        params = jnp.where(m[:, None], rparams, params)
        steps = jnp.where(m, -self._warmstart, steps)         # re-spawned agents restart the warm-start
        act_buf = jnp.where(m[:, None, None], jnp.zeros_like(act_buf), act_buf)
        wind = jnp.where(m[:, None], jnp.zeros_like(wind), wind)
        new_delay = self._sample_delay(kdelay, n)
        delay_idx = jnp.where(m, new_delay, delay_idx)
        airborne = jnp.where(m, (-rpos[:, 2]) > self.AIRBORNE_ALT, airborne)
        prev_pos_ned = jnp.where(m[:, None], rpos, pos_ned)

        # task carried state: fresh for reset agents, then re-observe them so obs
        # matches the fresh spawn (non-reset agents keep the pre-reset obs/state).
        rtask = self.task.init(kinit, n, plant_state, rpos)
        task_state = _tree_where(m, rtask, task_state)
        # ``fresh_spawn`` tells the task that ONLY the reset worlds' half of this result is
        # kept, so per-step integrators with nothing to integrate at t=0 can be skipped. This
        # call is otherwise a full second observe over the WHOLE fleet — on a 4090 at fleet
        # 1024 that is most of the env step — and everything it computes for a non-reset world
        # is discarded three lines below. See GateTask.observe's docstring.
        # No ``substeps``: they belong to the PRE-reset trajectory while the pose below is
        # the fresh spawn, so feeding them here would hand a respawned world another world's
        # IMU. ``fresh_spawn`` already tells the task to fall back to the endpoint sample.
        robs, rtask_state = self.task.observe(
            plant_state, task_state, act_buf, kobs2, params, fresh_spawn=True)
        obs = jnp.where(m[:, None], robs, obs)
        task_state = _tree_where(m, rtask_state, task_state)

        new_state = SkyFlowState(
            fw_blob, fw_state, plant_state, params, key, prev_pos_ned, steps,
            airborne, act_buf, delay_idx, wind, term_ema, task_state, ep_reach,
            ep_ret_acc=ep_ret_acc, ep_ret=ep_ret, ep_len=ep_len, miss_ema=miss_ema)
        # per-agent flags for a step-by-step driver (teleop / diagnostic); the PPO
        # loop drops info in its rollout, so training-time visibility rides term_ema.
        info = {"pos": pos_ned, "crashed": crashed, "disarm": ~armed,
                "flyaway": flyaway, "ground": ground_any,
                # off-policy contract (see the `terminated` / `final_obs` comments above)
                "terminated": terminated, "truncated": truncated, "final_obs": final_obs}
        info.update(ev.info)
        return obs, new_state, reward, done, info

    def privileged_obs(self, state: SkyFlowState) -> jax.Array:
        """Fully-observed privileged state for the asymmetric critic (same clip as
        the actor obs). The trainer derives the critic's obs stream from the
        env_state via this method — the deployed actor never sees it."""
        return finalize_obs(self.task.privileged_state(state.plant, state.task))

    def phys_factors(self, state: SkyFlowState) -> jax.Array:
        """Per-world DR draw as fractional offsets from nominal, ``param/nominal − 1``
        [F, P]. Supervised sysid targets for the recurrent actor's aux decoder: the
        policy can only recover these by feeling how ITS world responds, so decoding
        them forces implicit online system identification into the hidden state.
        Columns whose nominal is 0 (disabled terms, e.g. k_v2) stay 0.

        Width is PINNED to the first 46 param keys (the pre-``w_slew`` contract):
        params appended after that (structural limits, nominal 0 -> the offset column
        is identically 0 -> no sysid information) are excluded so every existing
        checkpoint's aux head keeps its shape and stays restorable."""
        nom = self._params_nominal[None, :46]
        return jnp.where(jnp.abs(nom) > 1e-8, state.params[:, :46] / jnp.where(
            jnp.abs(nom) > 1e-8, nom, 1.0) - 1.0, 0.0)

    def metrics(self, state: SkyFlowState) -> dict[str, jax.Array]:
        pos_ned, vel_ned, _, _ = plant.pose_ned(state.plant)
        crash_lo, crash_hi = self.task.term_labels
        m = {
            "skyflow/airborne_frac": state.airborne.mean(),
            "skyflow/speed_mps": jnp.linalg.norm(vel_ned, axis=1).mean(),
            "skyflow/alt_m": (-pos_ned[:, 2]).mean(),
            # outcome mix (EMA of per-step fractions) — which failure dominates. Mutually
            # exclusive, so for a terminating-success task these sum to train/done_frac.
            "skyflow/term_disarm": state.term_ema[0],
            "skyflow/term_flyaway": state.term_ema[1],
            "skyflow/term_ground": state.term_ema[2],
            f"skyflow/{crash_lo}": state.term_ema[3],
            f"skyflow/{crash_hi}": state.term_ema[4],
            # ran out of step budget rather than ending for a reason — the number that says
            # whether raising env.max_ep would buy anything.
            "skyflow/term_timeout": state.term_ema[5],
            # last completed episode, averaged over the fleet. ep_len is the TRUE latched
            # length; train/ep_len_est (1/done_frac) is a geometric approximation that is
            # biased above max_ep under a hard cap, and the trainer drops it when this exists.
            "skyflow/ep_return": state.ep_ret.mean(),
            "skyflow/ep_len": state.ep_len.mean(),
        }
        if self._split_lost:
            # ran out of the PER-GATE budget (lost_steps / lost_range) rather than hitting
            # the frame. Split out because term_gate_miss silently carried both: a frame hit
            # says "aim", a lost says "too slow to reach the next gate" — opposite fixes.
            m["skyflow/term_lost"] = state.term_ema[6]
            # mean Chebyshev offset at the gate plane, over crossings. Only the gate task in
            # GATE mode reports crossings (its hover mode relabels to term_lost), and the
            # ratio is 0/0 before the first one — clamp the denominator rather than emit NaN.
            m["skyflow/gate_miss_margin_m"] = (
                state.miss_ema[0] / jnp.maximum(state.miss_ema[1], 1e-12))
        for k, v in self.task.scalar_metrics(state.plant, state.task, state.ep_reach).items():
            m[f"skyflow/{k}"] = v
        return m

    def fpv(self, state: SkyFlowState, n: int = 1) -> jax.Array:
        """First-person render for the first ``n`` worlds — the FPV the policy
        sees, for the trainer's viz clip. Delegated to the task."""
        return self.task.fpv(state.plant, state.task, n)

    def fpv_scene(self, state: SkyFlowState, n: int = 1) -> jax.Array:
        """Viz-only RGB FPV (black sky / WHITE floor / #ff4000 gates) for the first
        ``n`` worlds — the wandb clip look with a cosmetic ground reference, even
        when the policy consumes 1-ch coverage. Delegated to the task; only tasks
        that render a floor expose it (the viz path falls back to :meth:`fpv`)."""
        return self.task.fpv_scene(state.plant, state.task, n)

    def attitude_axes(self, state: SkyFlowState) -> jax.Array:
        """Body forward + up axes in world NED, for the trainer's attitude glyph
        (viz only — never on the training hot path). Returns ``[F, 2, 3]``:
        ``[:, 0]`` = body-x (forward, FRD) and ``[:, 1]`` = body-up (thrust, −body-z).
        Two axes fix the full orientation, so the viz draws a little forward-bar +
        up-tick "flag" per sample and you read roll/pitch/yaw at a glance."""
        _, _, quat_ned, _ = plant.pose_ned(state.plant)
        R = plant.rot_matrix(quat_ned)                 # [F,3,3] body→world NED
        return jnp.stack([R[:, :, 0], -R[:, :, 2]], axis=1)

    def close(self) -> None:
        if self.firmware is not None:
            self.firmware.close()
