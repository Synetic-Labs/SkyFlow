"""
Airframe registry and domain randomization (DESIGN.md §6).

An Airframe is a spec parameter dict (skyflow_dynamics SCHEMA keys, validated by
pack_params) plus the harness-side quantities the spec charter keeps out of the ODE: the
rotor-speed operating limits (the dict's `limits` entry) and the throttle-curve blend of
the command map. Randomization is multiplicative — per-entry factors 1 + scale·U(−b, +b)
with brackets from DR_BRACKETS — so zero-valued nominals stay exactly zero and the
structural keys (spin, axis, grav) are never touched. The same routine runs at reset and
at in-jit auto-reset respawn.
"""

import copy
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.dynamics import CRAZYFLIE, N_ROTORS, pack_params, param_slices


@dataclass(frozen=True)
class Airframe:
    """One vehicle: spec parameters plus harness-side command-map and limit constants."""

    name: str
    values: dict  # spec SCHEMA row (+ its `limits` entry), pack_params-validated
    rotor_speed_min: float  # rad/s — post-step clip floor and throttle-map Ω_min
    rotor_speed_max: float  # rad/s — post-step clip ceiling and throttle-map Ω_max
    throttle_k: float  # blend of the verified curve Ω_c = ΔΩ·√(k·u² + (1−k)·u) + Ω_min


def _from_spec(name: str, values: dict, throttle_k: float) -> Airframe:
    """Airframe from a spec parameter dict, limits lifted from its `limits` entry."""
    values = copy.deepcopy(values)  # registry rows never alias the spec's module dict
    limits = values["limits"]
    return Airframe(
        name=name,
        values=values,
        rotor_speed_min=float(limits["rotor_speed_min"]),
        rotor_speed_max=float(limits["rotor_speed_max"]),
        throttle_k=throttle_k,
    )


#: Built-in vehicles. Crazyflie throttle_k = 1.0 keeps the command map linear in u
#: (√(u²) = u): no measured throttle-curve blend exists for the brushed Crazyflie, so the
#: neutral setting of the verified curve is the honest default.
AIRFRAMES: dict[str, Airframe] = {
    "crazyflie": _from_spec("crazyflie", CRAZYFLIE, throttle_k=1.0),
}


def register_airframe(name: str, airframe: Airframe) -> None:
    """
    Add a vehicle to the registry. Validates the parameter dict through pack_params
    (rejects double-counted aero terms, bad spin signs, non-unit thrust axes) and refuses
    name collisions — shadowing a registered vehicle silently is never what anyone wants.
    """
    if name in AIRFRAMES:
        raise ValueError(f"airframe {name!r} is already registered")
    pack_params(airframe.values)
    AIRFRAMES[name] = airframe


#: Structural keys excluded from randomization: spin signs and thrust axes define the
#: airframe's geometry class, grav is the world, not the vehicle (DESIGN.md §6).
NEVER_JITTER: tuple[str, ...] = ("spin", "axis", "grav")

#: SCHEMA key → half-width b of the multiplicative jitter factor 1 + scale·U(−b, +b).
#: §6 fixes mass/inertia/ct*/cq*/tau_m/k_d/k_z/r_prop; the rest follow their families:
#: motor-response coefficients like tau_m (0.20), rotor inertia like inertia (0.15),
#: secondary aero like k_d/k_z (0.30), geometry not jittered like r_prop (0.0).
DR_BRACKETS: dict[str, float] = {
    "mass": 0.10,
    "inertia": 0.15,
    "ct0": 0.15,
    "ct1": 0.15,
    "ct2": 0.15,
    "cq0": 0.15,
    "cq1": 0.15,
    "cq2": 0.15,
    "tau_m": 0.20,
    "ka1": 0.20,
    "ka2": 0.20,
    "kd1": 0.20,
    "kd2": 0.20,
    "I_rot": 0.15,
    "c_D": 0.30,
    "c_L": 0.30,
    "k_d": 0.30,
    "k_z": 0.30,
    "k_flap": 0.30,
    "k_h": 0.30,
    "k_angle": 0.30,
    "k_hor": 0.30,
    "k_v2": 0.30,
    "rotor_pos": 0.0,
    "r_prop": 0.0,
}


@lru_cache(maxsize=1)
def _jitter_brackets() -> np.ndarray:
    """
    Per-entry bracket vector b [P] in pack_params order: DR_BRACKETS spread through
    param_slices, with the NEVER_JITTER keys hard-masked to zero. Layout is derived, never
    hardcoded, so it survives spec parameter additions.
    """
    slices = param_slices(N_ROTORS)
    dim = 1 + max(int(idx.max()) for idx in slices.values())
    b = np.zeros(dim, np.float32)
    for name, idx in slices.items():
        b[idx] = DR_BRACKETS.get(name, 0.0)
    for name in NEVER_JITTER:
        b[slices[name]] = 0.0
    return b


def sample_params(key, airframe: Airframe, fleet: int, scale: float):
    """
    Per-world randomized flat parameter rows [fleet, P] float32 (pack_params order).

    Each stored entry gets an independent multiplicative factor 1 + scale·U(−b, +b) with
    b from DR_BRACKETS (log-uniform-style: symmetric relative jitter about the nominal).
    Consequences of the multiplicative form: zero nominals stay exactly zero, scale=0
    returns the nominal row bit-exactly, and spin/axis/grav are never jittered (masked
    through param_slices). Callers keep scale·max(b) < 1 so factors stay positive.
    """
    nominal = jnp.asarray(pack_params(airframe.values), jnp.float32)
    b = _jitter_brackets()
    u = jax.random.uniform(key, (fleet, nominal.shape[0]), jnp.float32, -1.0, 1.0)
    return nominal * (1.0 + scale * b * u)
