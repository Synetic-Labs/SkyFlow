"""
Fleet-batched dynamics — the only SkyFlow module that imports skyflow_dynamics.

The generated JAX backend (skyflow_dynamics.backends.jax, aliased sfd) supplies
single-vehicle functions code-generated from the SymPy spec; this module vmaps each one
once, centrally, over the leading fleet axis and assembles the backend's flat input
layout. No other module touches the flat layouts, and nothing in SkyFlow writes a force,
torque, or sensor equation — a missing physics term goes through the SkyFlow-Dynamics
INTAKE protocol and is consumed here, never implemented here (DESIGN.md §1, §5).

Layouts (spec calling convention, n = N_ROTORS = 4):

    plant  [F, 17] = x_W(3), v_W(3), q_wxyz(4), ω_B(3), Ω(4)       SI, rad/s, wxyz Hamilton
    inputs [F, 13] = Ω_c(4), v_wind_W(3), F_ext_W(3), τ_ext_B(3)   assembled here only
    params [F, P]  = pack_params order (param_slices gives the name → index map)

The motor model is fixed to "first_order" for v0.2; the asymmetric model is the same
backend behind the `motor_model` keyword (the env surfaces it as a config field when it
lands). These wrappers create no arrays beyond input concatenation, so precision follows
the ambient JAX config (DESIGN.md §3): x64 test runs reproduce the backend's golden 1e-9
tolerances; the env feeds float32 for rollouts.
"""

from functools import cache

import jax
import jax.numpy as jnp
from skyflow_dynamics.backends import jax as sfd
from skyflow_dynamics.backends.jax import pack_params, param_slices

N_ROTORS = 4
STATE_DIM = 13 + N_ROTORS  # 17

#: Crazyflie 2.0 reference parameter dict (spec SCHEMA row + its harness-side `limits`
#: entry), re-exported through the backend so params.py never imports skyflow_dynamics.
CRAZYFLIE = sfd.parameters.CRAZYFLIE

__all__ = [
    "CRAZYFLIE",
    "N_ROTORS",
    "STATE_DIM",
    "imu",
    "pack_params",
    "param_slices",
    "statedot",
    "substep",
    "throttle_to_omega",
]


def _assemble_inputs(omega_cmd, wind_vel, f_ext, tau_ext):
    """Backend input rows [F,13]: Ω_c rad/s, v_wind (world), F_ext N (world), τ_ext N·m (body)."""
    return jnp.concatenate([omega_cmd, wind_vel, f_ext, tau_ext], axis=-1)


@cache
def _substep_fleet(motor_model: str, per_world_w_max: bool = False):
    """vmap of one backend RK4 step + reference post-step over the fleet axis.

    ``per_world_w_max`` maps the rotor-speed ceiling over the fleet axis too ([F,4]
    rows — the battery-sag trait); the default keeps the scalar-constant path
    bit-identical for every existing caller."""
    step = sfd.rk4_step_fn(N_ROTORS, motor_model)

    def one(s, u, p, dt, w_min, w_max):
        return sfd.post_step(step(s, u, p, dt), w_min, w_max)

    return jax.vmap(one, in_axes=(0, 0, 0, None, None, 0 if per_world_w_max else None))


def substep(
    plant,
    omega_cmd,
    wind_vel,
    f_ext,
    tau_ext,
    params,
    dt,
    w_min,
    w_max,
    *,
    motor_model: str = "first_order",
):
    """
    One physics substep for the whole fleet → plant' [F,17].

    Backend RK4 (all inputs zero-order-held across dt seconds) followed by the reference
    post-step the golden vectors pin: quaternion renormalization and rotor-speed clip to
    [w_min, w_max] rad/s — the airframe's `limits` entry, passed in by the caller.
    omega_cmd [F,4] rad/s; wind_vel [F,3] world; f_ext [F,3] N world; tau_ext [F,3] N·m body.
    """
    u = _assemble_inputs(omega_cmd, wind_vel, f_ext, tau_ext)
    per_world = jnp.ndim(w_max) > 0  # [F,4] battery-sag rows vs the scalar constant
    return _substep_fleet(motor_model, per_world)(plant, u, params, dt, w_min, w_max)


@cache
def _statedot_fleet(motor_model: str):
    return jax.vmap(sfd.statedot_fn(N_ROTORS, motor_model), in_axes=(0, 0, 0))


def statedot(
    plant, omega_cmd, wind_vel, f_ext, tau_ext, params, *, motor_model: str = "first_order"
):
    """Continuous ṡ rows [F,17] — diagnostics only; the sim advances through substep()."""
    u = _assemble_inputs(omega_cmd, wind_vel, f_ext, tau_ext)
    return _statedot_fleet(motor_model)(plant, u, params)


# IMU mounting fixed at the body origin with identity orientation: sensor frame = body FLU.
# Plain Python floats stay weak-typed in JAX, so the measurement dtype follows the plant.
_IMU_OFFSET = (0.0, 0.0, 0.0)
_IMU_MOUNT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@cache
def _imu_fleet(motor_model: str):
    f = sfd.imu_fn(N_ROTORS, motor_model)

    def one(s, u, p):
        return f(s, u, p, _IMU_OFFSET, _IMU_MOUNT)

    return jax.vmap(one, in_axes=(0, 0, 0))


def imu(plant, omega_cmd, wind_vel, params, *, motor_model: str = "first_order"):
    """
    Exact generated IMU at the body origin, identity mount → (accel [F,3], gyro [F,3]).

    accel is specific force in body FLU, m/s² — (0, 0, +g) at exact hover; gyro is body
    rate, rad/s. External pokes are not fed to the IMU (frozen §5 signature: F_ext, τ_ext
    zero). Noise, bias, and scale corruption stay in sensors.py per the spec's sensor
    boundary.
    """
    zeros = jnp.zeros_like(wind_vel)
    u = _assemble_inputs(omega_cmd, wind_vel, zeros, zeros)
    out = _imu_fleet(motor_model)(plant, u, params)
    return out[:, :3], out[:, 3:]


def throttle_to_omega(u, w_min, w_max, k):
    """
    Normalized throttle u ∈ [0,1] [F,4] → commanded rotor speed Ω_c [F,4] rad/s via the
    verified curve Ω_c = (Ω_max-Ω_min)·√(k·u² + (1-k)·u) + Ω_min (spec.motor). The
    generated function is elementwise and broadcasts over the fleet axis, so it needs no
    vmap; endpoints are exact (u=0 → w_min, u=1 → w_max) for any blend k ∈ [0,1].
    """
    return sfd.throttle_to_speed_fn()(u, w_min, w_max, k)
