"""
Airframe registry and domain randomization (DESIGN.md §6).

An Airframe is a spec parameter dict (skyflow_dynamics SCHEMA keys, validated by
pack_params) plus the harness-side quantities the spec charter keeps out of the ODE: the
rotor-speed operating limits (the dict's `limits` entry) and the throttle-curve blend of
the command map. Randomization is multiplicative — per-entry factors 1 + scale·U(-b, +b)
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
    throttle_k: float  # blend of the verified curve Ω_c = ΔΩ·√(k·u² + (1-k)·u) + Ω_min


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

#: SCHEMA key → half-width b of the multiplicative jitter factor 1 + scale·U(-b, +b).
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


#: Correlated factor stage (DESIGN.md §6, measured 2026-08-22): one shared draw per
#: named PHYSICAL CAUSE, multiplied onto every SCHEMA key in its group before the
#: per-entry residual jitter. Head-to-head at equal ±20% width on ct2+cq2 (32 draws,
#: arm→climb→hover-hold): shared factor + ±3% residuals flies 0.88; the independent
#: per-entry structure flies 0.12 — independence across ct2/cq2 (and across rotors)
#: builds vehicles no prop can be, and the firmware mixer cannot fly them. Groups are
#: disjoint; keys outside every group draw only their per-entry residual.
FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    # Air density + prop condition: thrust, drag-torque and every aero term move
    # TOGETHER (rho scales them all; the cq2/ct2 ratio is prop geometry).
    "air_prop": ("ct0", "ct1", "ct2", "cq0", "cq1", "cq2", "c_D", "c_L",
                 "k_d", "k_z", "k_flap", "k_h", "k_angle", "k_hor", "k_v2"),
    # Payload / battery swap. Guarded by TW_FLOOR below.
    "mass": ("mass",),
    # Mass distribution; the prop+bell inertia rides the same build change.
    "inertia": ("inertia", "I_rot"),
}

#: Group → (lo, hi) of the shared factor 1 + scale·U(lo, hi). Asymmetric on purpose:
#: air density spans ISA -30% (hot high site) to +15..20% (cold sea level) and prop
#: wear only degrades; payload only adds mass beyond the as-measured nominal build.
FACTOR_LIMITS: dict[str, tuple[float, float]] = {
    "air_prop": (-0.30, 0.20),
    "mass": (-0.20, 0.40),
    "inertia": (-0.25, 0.40),
}

#: Thrust-to-weight floor for the mass guard: the measured liftoff knife edge
#: (mass x1.5 lifts at T/W 1.30; mass x1.3 + ct2 x0.85 fails at 1.27). The guard clamps
#: the mass DRAW so drawn thrust still lifts the drawn vehicle — it never pushes mass
#: below nominal (an airframe whose nominal T/W is already low is what it is).
TW_FLOOR = 1.3

#: Per-entry residual half-widths used INSTEAD of DR_BRACKETS when the factor stage is
#: on: the shared factors carry the wide common channels, so the per-entry draws model
#: only what is truly independent — prop-to-prop matching (measured 2.5% median torque
#: mismatch on the Air75; ±3% already sits at the stock-tune flyability edge), per-axis
#: mass-distribution asymmetry, and fit noise on the loose aero terms.
RESIDUAL_BRACKETS: dict[str, float] = {
    "mass": 0.0,     # single scalar — the mass factor IS the draw
    "inertia": 0.05,
    "ct0": 0.02, "ct1": 0.02, "ct2": 0.02,
    "cq0": 0.02, "cq1": 0.02, "cq2": 0.02,
    "tau_m": 0.20,   # no factor group: keeps its family width
    "ka1": 0.20, "ka2": 0.20, "kd1": 0.20, "kd2": 0.20,
    "I_rot": 0.05,
    "c_D": 0.10, "c_L": 0.10,
    "k_d": 0.10, "k_z": 0.10, "k_flap": 0.10, "k_h": 0.10,
    "k_angle": 0.10, "k_hor": 0.10, "k_v2": 0.10,
    "rotor_pos": 0.0,
    "r_prop": 0.0,
}


@lru_cache(maxsize=8)
def _jitter_brackets(
    overrides: tuple[tuple[str, float], ...] = (), residual: bool = False
) -> np.ndarray:
    """
    Per-entry bracket vector b [P] in pack_params order: DR_BRACKETS (or
    RESIDUAL_BRACKETS when the factor stage is on) spread through param_slices, with
    the NEVER_JITTER keys hard-masked to zero. Layout is derived, never hardcoded, so
    it survives spec parameter additions. `overrides` replaces individual half-widths
    (sorted (key, b) pairs — the hashable form of DomainRand.brackets); unknown or
    structural keys are rejected loudly.
    """
    base = RESIDUAL_BRACKETS if residual else DR_BRACKETS
    slices = param_slices(N_ROTORS)
    dim = 1 + max(int(idx.max()) for idx in slices.values())
    b = np.zeros(dim, np.float32)
    for name, idx in slices.items():
        b[idx] = base.get(name, 0.0)
    for name, half_width in overrides:
        if name in NEVER_JITTER:
            raise ValueError(f"bracket override for structural key {name!r} (never jittered)")
        if name not in slices:
            raise ValueError(f"bracket override for unknown SCHEMA key {name!r}")
        if half_width < 0.0:
            raise ValueError(f"bracket override {name!r} must be >= 0, got {half_width}")
        b[slices[name]] = half_width
    for name in NEVER_JITTER:
        b[slices[name]] = 0.0
    return b


@lru_cache(maxsize=8)
def _factor_tables(
    overrides: tuple[tuple[str, tuple[float, float]], ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Factor-stage tables: 0/1 masks [G, P] (FACTOR_GROUPS order) and (lo, hi) limits
    [G, 2] after merging `overrides` (sorted (group, (lo, hi)) pairs — the hashable
    form of DomainRand.factors). Unknown groups and inverted limits are rejected
    loudly; the NEVER_JITTER keys can never appear in a group.
    """
    slices = param_slices(N_ROTORS)
    dim = 1 + max(int(idx.max()) for idx in slices.values())
    names = list(FACTOR_GROUPS)
    masks = np.zeros((len(names), dim), np.float32)
    for gi, group in enumerate(names):
        for key in FACTOR_GROUPS[group]:
            if key in NEVER_JITTER:
                raise ValueError(f"factor group {group!r} names structural key {key!r}")
            masks[gi, slices[key]] = 1.0
    limits = np.array([FACTOR_LIMITS[g] for g in names], np.float32)
    for group, lim in overrides:
        if group not in FACTOR_GROUPS:
            raise ValueError(f"factor override for unknown group {group!r}; "
                             f"groups: {sorted(FACTOR_GROUPS)}")
        lo, hi = (float(x) for x in lim)
        if lo > hi:
            raise ValueError(f"factor group {group!r}: lo {lo} > hi {hi}")
        limits[names.index(group)] = (lo, hi)
    return masks, limits


def max_bracket(brackets=None, residual: bool = False) -> float:
    """
    Largest effective half-width after merging `brackets` overrides (over the residual
    table when the factor stage is on). The env validates scale·max_bracket < 1 at
    construction, so multiplicative factors stay positive (a factor <= 0 would flip or
    zero a physical parameter).
    """
    overrides = () if not brackets else tuple(sorted(brackets.items()))
    return float(_jitter_brackets(overrides, residual).max())


def factor_floor(factors=None) -> float:
    """
    Largest |lo| over the factor groups after merging `factors` overrides (0.0 when the
    factor stage is off). The env validates scale·factor_floor < 1 at construction for
    the same positivity reason as max_bracket; group names are validated loudly here.
    """
    if factors is None:
        return 0.0
    overrides = tuple(sorted((k, tuple(v)) for k, v in factors.items()))
    _, limits = _factor_tables(overrides)
    return float(np.abs(limits[:, 0]).max())


def apply_tw_guard(rows, nominal, w_max):
    """
    Clamp the mass entries of drawn parameter rows [F,P] so the drawn thrust at the
    rotor-speed ceiling still lifts the drawn vehicle at TW_FLOOR. ``w_max`` is the
    ceiling — the airframe scalar, or per-world [F] rows once the battery-sag trait has
    been drawn (the env re-applies this guard with the sagged ceiling: a tired pack
    buys less thrust, so the same payload rule must see it). Never pushes mass below
    the nominal build — a vehicle whose thrust alone cannot make TW_FLOOR is what it
    is. Idempotent, and a stricter (lower) ceiling only tightens the clamp.
    """
    slices = param_slices(N_ROTORS)
    w = jnp.asarray(w_max, jnp.float32)
    w = w[:, None] if w.ndim == 1 else w
    thrust_max = (rows[:, slices["ct0"]].sum(-1)
                  + (rows[:, slices["ct1"]] * w).sum(-1)
                  + (rows[:, slices["ct2"]] * w * w).sum(-1))
    grav = nominal[slices["grav"]][0]  # never jittered
    mass_cap = jnp.maximum(thrust_max / (TW_FLOOR * grav), nominal[slices["mass"]][0])
    m_idx = slices["mass"]
    return rows.at[:, m_idx].set(jnp.minimum(rows[:, m_idx], mass_cap[:, None]))


def sample_params(key, airframe: Airframe, fleet: int, scale: float, brackets=None,
                  factors=None):
    """
    Per-world randomized flat parameter rows [fleet, P] float32 (pack_params order).

    Stage 1 (per-entry residual): every stored entry gets an independent multiplicative
    factor 1 + scale·U(-b, +b). With `factors=None` this is the whole draw and b comes
    from DR_BRACKETS — the legacy sampler, bit-exact. `brackets` (a {SCHEMA key:
    half-width} mapping, e.g. DomainRand.brackets) replaces individual half-widths.

    Stage 2 (correlated, on when `factors` is not None): one shared draw per
    FACTOR_GROUPS entry, 1 + scale·U(lo, hi) with (lo, hi) from FACTOR_LIMITS merged
    with `factors` ({group: (lo, hi)}; {} = all defaults), multiplies every key in the
    group. Stage 1 then runs over RESIDUAL_BRACKETS instead of DR_BRACKETS. Two guards
    keep drawn vehicles physical, and both bound the DRAW, never the airframe: the mass
    draw is clamped so drawn thrust still lifts the drawn vehicle at TW_FLOOR (never
    below nominal mass), and drawn Izz is capped at (Ixx + Iyy) times the nominal's own
    planarity ratio (flat-body limit, shape-relative).

    Consequences of the multiplicative form: zero nominals stay exactly zero, scale=0
    returns the nominal row bit-exactly (both stages), and spin/axis/grav are never
    jittered (masked through param_slices). Callers keep scale·max(b) < 1 and
    scale·factor_floor < 1 so factors stay positive.
    """
    nominal = jnp.asarray(pack_params(airframe.values), jnp.float32)
    overrides = () if not brackets else tuple(sorted(brackets.items()))
    if factors is None:
        b = _jitter_brackets(overrides)
        u = jax.random.uniform(key, (fleet, nominal.shape[0]), jnp.float32, -1.0, 1.0)
        return nominal * (1.0 + scale * b * u)

    k_res, k_fac = jax.random.split(key)
    b = _jitter_brackets(overrides, residual=True)
    u = jax.random.uniform(k_res, (fleet, nominal.shape[0]), jnp.float32, -1.0, 1.0)
    row = nominal * (1.0 + scale * b * u)

    f_over = tuple(sorted((k, tuple(v)) for k, v in factors.items()))
    masks, limits = _factor_tables(f_over)
    g = jax.random.uniform(k_fac, (fleet, masks.shape[0]), jnp.float32,
                           limits[:, 0], limits[:, 1])
    row = row * (1.0 + scale * (g @ masks))

    row = apply_tw_guard(row, nominal, airframe.rotor_speed_max)
    slices = param_slices(N_ROTORS)

    # Planar guard on the draw, relative to the vehicle's own shape: a flat body has
    # Izz = Ixx + Iyy, but measured EFFECTIVE inertias sit slightly above it (ducts,
    # flap softening — the crazyflie nominal ratio is 1.010, the Air75 fit 1.047), so
    # the cap preserves the nominal ratio instead of forcing the textbook limit.
    # Never binds at scale=0: the nominal row passes through bit-exactly.
    i_idx = slices["inertia"]  # pack order: xx, yy, zz, xy, xz, yz
    nom_ratio = nominal[i_idx[2]] / (nominal[i_idx[0]] + nominal[i_idx[1]])
    # The 1e-6 slack absorbs float32 ratio-reconstruction rounding so the unjittered
    # nominal passes through bit-exactly at scale=0 (a few ulp, never a real widening).
    zz_cap = (row[:, i_idx[0]] + row[:, i_idx[1]]) * jnp.maximum(1.0, nom_ratio) * (1.0 + 1e-6)
    row = row.at[:, i_idx[2]].set(jnp.minimum(row[:, i_idx[2]], zz_cap))
    return row
