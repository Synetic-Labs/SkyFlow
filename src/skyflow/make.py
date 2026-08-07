"""Factory for the functional SkyFlow env — platform (``env``) + objective (``task``).

Reads the ``env`` config for the shared platform (firmware, plant, DR, camera)
and the sibling top-level ``task`` config for the objective, then hands both to
:class:`~skyflow.env.SkyFlowEnv` (which builds the task via
``task.name``). Only the keys a config actually carries are forwarded, so the
task / env defaults stay the fallback. Heavy imports (cudaflight FFI) live inside
the env, so this module imports cleanly anywhere; the ImportError is raised by the
caller (``make_functional``) with the firmware-extra hint.
"""

from __future__ import annotations

from typing import Any

# platform + camera/vision knobs (the `env` config); always forwarded.
_PLATFORM_KEYS = (
    "control", "differentiable", "scramble_ep_phase",
    "control_hz", "max_ep", "stuck_after", "airframe", "crash_penalty",
    "act_calm_weight", "act_calm_center", "act_calm_deadzone", "act_calm_scale",
    "act_smooth_weight", "act_smooth_scale",
    "bounds_xy_m", "bounds_z_m", "ground_tilt_limit_rad",
    "physics_rando_scale", "disturbance_scale",
    # per-STEP disturbance knobs — forwardable because disturb_poke_prob is a per-control-step
    # probability and therefore rate-dependent (at 30 Hz the same value is 3x fewer pokes per
    # second of flight than at 90 Hz).
    "disturb_wind_acc", "disturb_wind_tau_s", "disturb_poke_prob",
    "disturb_poke_vel_mps", "disturb_poke_rate_rps",
    "act_delay_min_steps", "act_delay_steps", "act_delay_nominal_steps",
    "warmstart_steps",
    "obs_frame_stack", "vision", "obs_rgb", "obs_retina",
    "asymmetric_critic", "cam_height", "cam_width", "cam_fov_x_deg", "cam_fov_y_deg",
    "cam_mount_pitch_deg", "cam_offset_body", "cam_supersample",
    "cam_pitch_jitter_deg", "cam_roll_jitter_deg",
    "cam_hz", "cam_hz_jitter", "obs_stale_prob",
    "imu_scale_dr", "accel_noise_std", "gyro_noise_std",
    "mask_noise_scale", "mask_noise_hold", "mask_outer_grow_m", "mask_unet_ckpt",
    "motor_perm", "eeprom", "device_index", "settle_ms", "firmware_backend",
)
# per-task objective knobs (the `task` config), keyed by task.name.
_TASK_KEYS: dict[str, tuple[str, ...]] = {
    "hover": (
        "spawn_north_m", "spawn_east_m", "spawn_radius_m", "spawn_alt_m",
        "spawn_vel_mps", "spawn_tilt_rad", "spawn_yaw_rad", "spawn_ground_frac",
        "spawn_air_motor_norm", "spawn_rando_scale",
        "target_half_xy_m", "target_alt_lo_m", "target_alt_hi_m", "resampling_time_s",
        "safe_half_xy_m", "safe_alt_lo_m", "safe_alt_hi_m",
        "viz_outer_half_xy_m", "viz_outer_alt_lo_m", "viz_outer_alt_hi_m",
        "hold_radius_m",
        "pos_weight", "pos_lambda", "hold_bonus_weight", "hold_bonus_lambda",
        "prog_weight", "yaw_weight", "yaw_lambda", "vel_weight", "rate_weight",
        "yaw_rate_weight",
        "rel_pos_xy_scale", "rel_pos_z_scale", "lin_vel_xy_scale", "lin_vel_z_scale",
        "obs_noise_pos_m", "obs_noise_vel_mps", "obs_noise_rot"),
}
# config list fields SkyFlowEnv/tasks want as plain tuples.
_TUPLE_KEYS = ["mask_outer_grow_m", "motor_perm", "cam_offset_body",
               "spawn_alt_m", "spawn_dist_m"]


def register_task_keys(name: str, keys: tuple[str, ...],
                       tuple_keys: tuple[str, ...] = ()) -> None:
    """Declare which ``task`` config fields to forward for a registered task.

    The companion to :func:`skyflow.tasks.register_task`: that one teaches the env how
    to BUILD your task, this one teaches :func:`make_skyflow` which config fields belong
    to it. Only needed if you drive the env from config objects rather than constructing
    :class:`~skyflow.env.SkyFlowEnv` directly. ``tuple_keys`` names any of them that
    arrive as config lists but must reach the task as tuples.
    """
    _TASK_KEYS[name] = keys
    for k in tuple_keys:
        if k not in _TUPLE_KEYS:
            _TUPLE_KEYS.append(k)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def make_skyflow(cfg: Any, task_cfg: Any = None):
    """Build the env from the platform config ``cfg`` (= ``env``) and the objective
    config ``task_cfg`` (= root ``task``; ``None`` → the default hover task)."""
    from .env import SkyFlowEnv

    task = _get(task_cfg, "name", "hover")
    kw: dict[str, Any] = {}
    for k in _PLATFORM_KEYS:
        v = _get(cfg, k)
        if v is not None:
            kw[k] = v
    for k in _TASK_KEYS.get(task, ()):
        v = _get(task_cfg, k)
        if v is not None:
            kw[k] = v
    for tk in _TUPLE_KEYS:
        if tk in kw:
            kw[tk] = tuple(int(i) for i in kw[tk]) if tk == "motor_perm" else tuple(kw[tk])
    return SkyFlowEnv(int(_get(cfg, "fleet", 1024)), task=task, **kw)
