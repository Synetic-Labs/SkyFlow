"""
Estimator-error models — the L5 (policy-observation) corruption layer (ERRORS.md).

A deployed policy never observes the true state: it observes an estimator (mocap
rigid-body fit today, VIO later). This module models that estimator's error as a
stochastic dynamical system on the PLANT STATE, not on the observation vector:

    x_est = corrupt(x_true),  built from three components per channel group:
        bias   trait    U(-b, +b), drawn once per episode, constant within it
        ou     drift    OU process, stationary std sigma, correlation time tau
        white  process  N(0, std), fresh every control step

Corrupting the state (and letting the task build its observation from x_est) is
what makes the error physical for free: gate-frame quantities derive from the
corrupted position exactly like a planner fed by a real estimator, the attitude
error is one small rotation (the corrupted quaternion stays a unit quaternion, so
every derived rotation matrix stays a valid rotation), task constants (flight
plan, validity flags) are never corrupted, and any observation-history ring
stores the estimate AS IT ARRIVED — frozen, the way a real estimator's history is.

Channel groups (3 wide each, packed [12] in this order):

    pos   m       world position error
    vel   m/s     world velocity error
    att   rad     rotation-vector error, applied as q_est = q_err(delta) * q_true
    rate  rad/s   body-rate estimate error (deploy rates come from the FC gyro)

plus a relative white error on the rotor-speed estimate (eRPM telemetry class)
and a DROPOUT event: with p_drop per step the estimator repeats its last emitted
estimate for a geometric-mean duration (mocap occlusion, VIO tracking loss) —
staleness, not noise.

Profile values are LITERATURE-CLASS: right order of magnitude for the named
estimator, sourced below. Replace them with measured values when deploy residual
logs and the sit-still Allan bench exist — the fields map one-to-one (Allan
white-noise density -> white, bias instability -> bias, flat-region knee -> tau).

The determinism charter holds: nothing here runs inside the ODE or the firmware
path, every draw enters through explicitly passed keys, and obs_error=None (or
the "none" profile, all widths zero) leaves observation VALUES bit-identical.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array

#: Channel groups, 3 wide each, packed [12] in this order.
GROUPS = ("pos", "vel", "att", "rate")

#: Per-group error tables. Sources (literature class, to be replaced by measured):
#:   mocap — Vicon/OptiTrack rigid-body class, indoor volume: marker-centroid
#:     jitter 0.2-1 mm RMS (white pos), wand-calibration + marker-frame offset
#:     1-3 mm / 0.2-0.4 deg (bias pos/att), differentiated-position velocity
#:     1-3 cm/s (white vel); rate error = post-arm-cal FC gyro (BMI270 class).
#:     Occlusion: brief marker loss, ~100 ms holds.
#:   vio — cm-class onboard visual-inertial odometry on small quads: cm position
#:     with seconds-scale drift (OU), yaw drift dominates attitude, tracking
#:     hiccups longer and more frequent than mocap occlusion.
#: Every entry: bias = uniform half-width (trait), ou = (stationary std, tau s)
#: (drift), white = per-step std (process).
PROFILES: dict[str, dict] = {
    "none": {
        "pos": dict(bias=0.0, ou=(0.0, 1.0), white=0.0),
        "vel": dict(bias=0.0, ou=(0.0, 1.0), white=0.0),
        "att": dict(bias=0.0, ou=(0.0, 1.0), white=0.0),
        "rate": dict(bias=0.0, ou=(0.0, 1.0), white=0.0),
        "rotor_rel": 0.0,
        "p_drop": 0.0,
        "drop_mean_steps": 1.0,
    },
    "mocap": {
        "pos": dict(bias=0.002, ou=(0.0005, 30.0), white=0.001),  # m
        "vel": dict(bias=0.0, ou=(0.0, 1.0), white=0.02),  # m/s
        "att": dict(bias=0.005, ou=(0.0, 1.0), white=0.004),  # rad (~0.3 / 0.23 deg)
        "rate": dict(bias=0.001, ou=(0.0, 1.0), white=0.005),  # rad/s, post-arm-cal gyro
        "rotor_rel": 0.01,  # eRPM telemetry ~1 %
        "p_drop": 0.005,  # ~one occlusion per 2 s at 100 Hz
        "drop_mean_steps": 10.0,  # ~100 ms holds
    },
    "vio": {
        "pos": dict(bias=0.01, ou=(0.02, 3.0), white=0.005),  # m — cm class + drift
        "vel": dict(bias=0.01, ou=(0.02, 3.0), white=0.02),  # m/s
        "att": dict(bias=0.01, ou=(0.01, 5.0), white=0.005),  # rad — yaw drift dominates
        "rate": dict(bias=0.001, ou=(0.0, 1.0), white=0.005),  # rad/s
        "rotor_rel": 0.01,
        "p_drop": 0.01,
        "drop_mean_steps": 20.0,  # ~200 ms tracking hiccups
    },
}

#: Config keys accepted by resolve_obs_error beyond the required "profile".
_CFG_KEYS = ("profile", "bias_frac", "ou_frac", "white_frac", "p_drop", "drop_mean_steps")


@dataclass(frozen=True)
class ObsErrorSpec:
    """Resolved estimator-error setting: profile widths x fractions, all static
    Python floats (jit constants). Built by resolve_obs_error, consumed by the
    draw/advance/corrupt functions below."""

    bias: tuple[float, ...]  # [12] uniform half-widths (trait)
    ou_sigma: tuple[float, ...]  # [12] OU stationary std (drift)
    ou_tau: tuple[float, ...]  # [4] per-group correlation time, s
    white: tuple[float, ...]  # [12] per-step std (process)
    rotor_rel: float  # relative white half-width on rotor speeds
    p_drop: float  # per-step dropout probability (event — never scaled)
    drop_mean_steps: float  # mean hold duration, control steps


def resolve_obs_error(cfg: dict | None) -> ObsErrorSpec | None:
    """DomainRand.obs_error dict -> ObsErrorSpec (None = the feature is off).

    cfg: {"profile": name, optional "bias_frac"/"ou_frac"/"white_frac" (>= 0
    multipliers on the profile's widths — the master DR scale folds in here),
    optional "p_drop"/"drop_mean_steps" overrides}. Unknown keys and unknown
    profiles raise loudly."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"dr.obs_error must be a dict or None, got {type(cfg).__name__}")
    bad = sorted(set(cfg) - set(_CFG_KEYS))
    if bad:
        raise ValueError(f"unknown dr.obs_error keys {bad}; accepted: {_CFG_KEYS}")
    name = cfg.get("profile")
    if name not in PROFILES:
        raise ValueError(
            f"unknown dr.obs_error profile {name!r}; registered: {sorted(PROFILES)}"
        )
    prof = PROFILES[name]
    fracs = {}
    for key in ("bias_frac", "ou_frac", "white_frac"):
        v = float(cfg.get(key, 1.0))
        if v < 0.0:
            raise ValueError(f"dr.obs_error.{key} must be >= 0, got {v}")
        fracs[key] = v
    p_drop = float(cfg.get("p_drop", prof["p_drop"]))
    if not 0.0 <= p_drop < 1.0:
        raise ValueError(f"dr.obs_error.p_drop must be in [0, 1), got {p_drop}")
    mean_steps = float(cfg.get("drop_mean_steps", prof["drop_mean_steps"]))
    if mean_steps < 1.0:
        raise ValueError(f"dr.obs_error.drop_mean_steps must be >= 1, got {mean_steps}")
    bias, sigma, white, tau = [], [], [], []
    for g in GROUPS:
        e = prof[g]
        bias += [fracs["bias_frac"] * float(e["bias"])] * 3
        sigma += [fracs["ou_frac"] * float(e["ou"][0])] * 3
        white += [fracs["white_frac"] * float(e["white"])] * 3
        if float(e["ou"][1]) <= 0.0:
            raise ValueError(f"profile {name!r} group {g!r}: ou tau must be > 0")
        tau.append(float(e["ou"][1]))
    return ObsErrorSpec(
        bias=tuple(bias),
        ou_sigma=tuple(sigma),
        ou_tau=tuple(tau),
        white=tuple(white),
        rotor_rel=fracs["white_frac"] * float(prof["rotor_rel"]),
        p_drop=p_drop,
        drop_mean_steps=mean_steps,
    )


def draw_bias(key: Array, f: int, spec: ObsErrorSpec) -> Array:
    """Per-episode bias trait rows [F,12]: per-channel U(-b, +b). Zero widths give
    exactly-zero columns, so the leaf always exists and stays inert."""
    half = jnp.asarray(spec.bias, jnp.float32)
    return half * jax.random.uniform(key, (f, 12), jnp.float32, -1.0, 1.0)


def advance_ou(ou: Array, key: Array, dt: float, spec: ObsErrorSpec) -> Array:
    """One control step of the drift process [F,12] — the wind-gust discretization:
    exact decay exp(-dt/tau) per group plus the variance-matched kick, so the
    stationary std is exactly ou_sigma for any step size."""
    alpha = jnp.repeat(
        jnp.asarray([math.exp(-dt / t) for t in spec.ou_tau], jnp.float32), 3
    )
    sigma = jnp.asarray(spec.ou_sigma, jnp.float32)
    kick = sigma * jnp.sqrt(1.0 - alpha * alpha)
    return alpha * ou + kick * jax.random.normal(key, ou.shape, jnp.float32)


def _quat_mul(a: Array, b: Array) -> Array:
    """Hamilton product [F,4] wxyz."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _rotvec_quat(v: Array) -> Array:
    """Rotation vector [F,3] rad -> unit quaternion [F,4] wxyz (small-angle safe)."""
    ang = jnp.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * ang
    small = ang < 1e-8
    scale = jnp.where(small, 0.5, jnp.sin(half) / jnp.where(small, 1.0, ang))
    return jnp.concatenate([jnp.cos(half), v * scale], axis=-1)


def corrupt_plant(plant: Array, bias: Array, ou: Array, key: Array, spec: ObsErrorSpec) -> Array:
    """The estimator's state estimate [F,17] from the true plant [F,17].

    Additive (bias + ou + white) on position, velocity and body rate; the attitude
    channel sum is a rotation VECTOR applied as one small rotation on the true
    quaternion (renormalized — the estimate is always a valid rotation); rotor
    speeds get a relative white error (eRPM class), floored at zero. The white
    draws are fresh every call; bias and ou are the threaded states."""
    k_white, k_rotor = jax.random.split(key)
    white = jnp.asarray(spec.white, jnp.float32) * jax.random.normal(
        k_white, bias.shape, jnp.float32
    )
    e = bias + ou + white
    pos = plant[:, 0:3] + e[:, 0:3]
    vel = plant[:, 3:6] + e[:, 3:6]
    quat = _quat_mul(_rotvec_quat(e[:, 6:9]), plant[:, 6:10])
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    omega = plant[:, 10:13] + e[:, 9:12]
    rel = spec.rotor_rel * jax.random.uniform(
        k_rotor, (plant.shape[0], 4), jnp.float32, -1.0, 1.0
    )
    rotors = jnp.maximum(plant[:, 13:17] * (1.0 + rel), 0.0)
    return jnp.concatenate([pos, vel, quat, omega, rotors], axis=-1)


def draw_hold(key: Array, f: int, spec: ObsErrorSpec) -> Array:
    """Dropout hold durations [F] int32 >= 1: ceil of an exponential with the
    profile's mean — the geometric-class duration of an occlusion/tracking loss."""
    u = jax.random.uniform(key, (f,), jnp.float32, 1e-7, 1.0)
    d = jnp.ceil(-spec.drop_mean_steps * jnp.log(u))
    return jnp.maximum(d, 1.0).astype(jnp.int32)
