"""Airframe parameters for the SkyFlow analytic plant.

One entry per drone, each the SINGLE SOURCE OF TRUTH for that airframe's numbers:

* :func:`air75_ii_racer` — **BETAFPV Air75 II Racer** (1S 75 mm ducted tinywhoop,
  0802 25000KV + GF 1614). Every coefficient SYSTEM-IDENTIFIED from real Vicon
  flight — see the §SYSTEM-IDENTIFIED block.

To add your own airframe: write a function returning :class:`PlantParams` and pass it to
:func:`register_airframe`. Identifying the coefficients against real or simulated flight
is the work; the plant itself is airframe-agnostic.

Select at the env with :func:`airframe_params` (``SkyFlowConfig.airframe``).
Every airframe states its CORE coefficients explicitly — :class:`PlantParams`
gives those no defaults, so nothing silently inherits another drone's physics; the
optional EXTENSION terms (Hill thrust map, inflow-deficit, wake, command-yaw)
default to 0 = disabled, so an airframe opts into one only by naming it. The paper's
Table II reference values (Diermayr et al. 2025, arXiv 2510.14783, the A2RL-class
quad the plant STRUCTURE comes from) are the published starting point for a re-fit.

The plant is mass-normalised the way the paper is: the force/torque coefficients
already fold in mass and inertia, so there is no explicit ``mass`` term in the
equations of motion (see :mod:`plant`). Domain randomisation multiplies these
nominal values per-world (Table III brackets) — see :func:`randomization_scale`.

── PHYSICAL vs SIM-FIDELITY (this code drives BOTH the sim twin and, later, real
   hardware) ──────────────────────────────────────────────────────────────────
The plant equations are a GENERAL quadrotor model: the SAME math drives every
airframe, real or sim — only the numbers change. So a real drone is a RE-FIT, not
a rewrite (see :func:`air75_ii_racer`, every coefficient system-identified from
real Vicon flight). All of these terms are genuine physics and transfer directly,
just re-fit per airframe: thrust (``k_w``, Hill ``sc_*``), motor lag
(``tau``/``tau_down``), drag (``k_x``/``k_y``/``k_x2``/``k_y2``/``k_v2``), inflow
(``k_angle``/``k_lin``/``k_lin2``/``k_hor``/``k_axg``/``k_wake``), roll/pitch
torque (``k_p*``/``k_q*`` ∝W²), rate damping (``k_d*``), gyroscopic coupling
(``J_*``).

ONE term is NOT real physics — the yaw-torque FORM:
  * PHYSICAL (real hardware): yaw is propeller reaction/drag torque ∝ ROTOR SPEED,
    the linear ``k_r1..4``·W terms (SkyDreamer default; the real Air75 uses them).
    Yaw authority varies with collective, as a real prop's does.
  * SIM-FIDELITY (``k_ru``): a COMMAND-LINEAR yaw torque (flat authority across
    collective). That is NOT prop physics — it exists to reproduce simulators that
    model yaw torque as ∝ the yaw-command mix. Do NOT enable ``k_ru`` on a
    real-hardware airframe; use ``k_r1..4`` (∝W) there instead.
The two are additive terms, selected purely by which coefficient is nonzero, so the
SAME plant serves sim and real without a code flag — a real airframe just sets
``k_ru=0`` and a nonzero ``k_r``, the game twin does the reverse.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

# The canonical coefficient ORDER. `plant` imports this same tuple (as `_PK`) to index
# the param row, so the two can never drift; `PlantParams.to_array` flattens in this order.
_PARAM_KEYS = (
    "k_x", "k_y", "k_w", "k_x2", "k_y2", "k_angle", "k_hor", "k_v2",
    "k_p1", "k_p2", "k_p3", "k_p4", "k_q1", "k_q2", "k_q3", "k_q4",
    "k_r1", "k_r2", "k_r3", "k_r4", "k_r5", "k_r6", "k_r7", "k_r8",
    "J_x", "J_y", "J_z", "tau", "k", "w_min", "w_max", "r_prop",
    "k_dp", "k_dq", "k_dr", "tau_down",
    # optional S-shaped command→thrust map (Hill): sc_tmax=0 disables it (→ the sqrt
    # blend). When >0 the per-motor rotor target is Wc=√(T_hill(u)/(4·k_w)) with
    # T_hill(u)=sc_tmax·u^sc_p/(sc_u50^sc_p+u^sc_p) — a real motor's deadband+saturation.
    "sc_tmax", "sc_u50", "sc_p",
    # optional SPEED-polynomial axial-inflow deficit (thrust ∝ 1−k_lin·v−k_lin2·v·|v|,
    # i.e. loss ∝ T·v — momentum-theory power balance). Alternative to the k_angle·α
    # form, whose 1/tip-speed slope scaling some ground-truth datasets contradict.
    # 0 (default) disables → bit-identical for airframes that use k_angle.
    "k_lin", "k_lin2",
    # optional DYNAMIC-INFLOW (wake) deficit + its decay lag + the inflow clip floor:
    # after a thrust cut the developed wake persists (~tau_wake) and chokes the rotor —
    # thrust undershoots, transiently through zero (hence eta_floor < 0 on airframes
    # that model it). k_wake=0 disables the wake term; eta_floor=0.05 is the historic
    # rail. See plant._deriv.
    "k_wake", "tau_wake", "eta_floor",
    # optional Tq-COUPLED axial-inflow deficit (Glauert advance ratio): thrust loss
    # ∝ k_axg·vbz/√T (the induced-velocity scale rt=√(k_w·ΣW²)). Unlike k_lin (constant
    # slope in v), this SHRINKS as thrust rises — it captures the 2-D (climb-rate,
    # torque) over-lift surface that shows up in the low-torque/high-climb corner.
    # 0 (default) disables → bit-identical for airframes that don't set it.
    "k_axg",
    # optional COMMAND-LINEAR yaw torque: Mz gains ±k_ru·ue, where ue is the Hill-INVERSE of the
    # (lagged) rotor state = an "effective duty" — so the term is linear in the pilot command
    # (⇒ FLAT yaw authority across collective; neither ∝W (droops) nor ∝W² (humps at the Hill
    # inflection) is flat) yet carries the motor lag via W. This models simulators that compute
    # yaw torque from the yaw-command mix rather than physical RPM² reaction, expressed in the
    # plant's post-curve-RPM state. 0 (default) disables → bit-identical.
    # ⚠ SIM-FIDELITY, not real prop physics — real hardware uses the ∝W k_r1..4 terms above
    # (yaw authority varies with collective). See the module PHYSICAL-vs-SIM-FIDELITY note.
    "k_ru",
)
# (A "w_slew" spin-up slew-limit key was trialled and REMOVED 2026-07-23 - see the
# rejection note at the motor-lag block in plant._deriv. Appending param keys resizes
# the row: keep env.phys_factors / infer aux width pinned to the first 46 keys so
# existing checkpoints stay restorable.)


@dataclass(frozen=True)
class PlantParams:
    """46 scalar coefficients of the SkyDreamer quad model (nominal, per-airframe).

    36 CORE coefficients (32 paper-structure + the 3 rate-damping terms k_dp/k_dq/k_dr,
    0 = paper-identical model, + the motor spin-down lag tau_down, == tau = symmetric =
    paper-identical) followed by 10 optional EXTENSION coefficients (Hill thrust map,
    inflow-deficit, wake, command-yaw). Fields mirror the paper's notation. The core
    fields have NO default — every airframe states them explicitly, so an entry can
    never silently inherit another drone's physics; the extensions default to 0 =
    disabled (bit-identical to the paper model). ``to_array`` flattens them in
    ``_PARAM_KEYS`` order into the row the vectorised dynamics consumes; domain
    randomisation broadcasts + jitters that row to ``[F, 46]``.
    """

    # aerodynamic drag / thrust (body frame)
    k_x: float
    k_y: float
    k_w: float
    k_x2: float
    k_y2: float
    k_angle: float
    k_hor: float
    k_v2: float
    # roll torque (motor differential)
    k_p1: float
    k_p2: float
    k_p3: float
    k_p4: float
    # pitch torque
    k_q1: float
    k_q2: float
    k_q3: float
    k_q4: float
    # yaw torque (drag + spin-up reaction)
    k_r1: float
    k_r2: float
    k_r3: float
    k_r4: float
    k_r5: float
    k_r6: float
    k_r7: float
    k_r8: float
    # gyroscopic coupling
    J_x: float
    J_y: float
    J_z: float
    # motor first-order lag (s) and thrust-curve blend
    tau: float
    k: float
    # commanded-RPM range (rad/s) the sqrt curve maps into
    w_min: float
    w_max: float
    # propeller radius (m) — sets the angle-of-attack / horizontal-flow scaling
    r_prop: float
    # angular-rate damping torque −k_d·ω (body frame), added to Mx/My/Mz. The paper
    # model has no rate-damping term (0 = bit-identical to the paper). Set these when
    # the airframe shows a hard rate PLATEAU under a sustained motor differential,
    # which a torque-only model overshoots.
    k_dp: float
    k_dq: float
    k_dr: float
    # motor spin-DOWN first-order lag (s). Spin-UP uses ``tau``. Equal to ``tau`` =>
    # symmetric first-order lag (paper-identical). Set it shorter than ``tau`` for
    # motors that spin down faster than they spin up, as most do under prop drag.
    tau_down: float
    # S-curve thrust map (Hill); sc_tmax=0 → disabled (sqrt-blend). See _PARAM_KEYS note.
    sc_tmax: float = 0.0
    sc_u50: float = 0.68
    sc_p: float = 2.85
    # speed-polynomial inflow deficit; 0 → disabled (k_angle form). See _PARAM_KEYS note.
    k_lin: float = 0.0
    k_lin2: float = 0.0
    # dynamic-inflow wake deficit; k_wake=0 → disabled (tau_wake then inert).
    k_wake: float = 0.0
    tau_wake: float = 0.114
    # inflow clip floor; 0.05 = the historic rail (thrust can't reverse). Airframes
    # modelling the post-chop wake choke set it negative. See _PARAM_KEYS note.
    eta_floor: float = 0.05
    # Tq-coupled axial (Glauert vbz/√T) inflow deficit; 0 → disabled. See _PARAM_KEYS note.
    k_axg: float = 0.0
    # command-linear (Hill-inverse effective-duty) yaw torque coeff; 0 → disabled. Flat-in-
    # collective yaw (a game-style sim's probable yaw model). ⚠ SIM-FIDELITY only — a real drone
    # uses the ∝W k_r1..4 terms instead (see the module PHYSICAL-vs-SIM-FIDELITY note).
    k_ru: float = 0.0

    def to_array(self) -> jax.Array:
        """Flatten to a ``[46]`` float32 row in ``_PARAM_KEYS`` order."""
        return jnp.asarray([getattr(self, k) for k in _PARAM_KEYS], jnp.float32)


# ── BETAFPV Air75 II Racer — SYSTEM-IDENTIFIED (Phase I+II, 2026-07-13) ──────
# Fitted from the real drone: datasets/SYSID_Air_75_II_Racer (8 Vicon sessions,
# 2026-07-12) + SYSID_Air_75_II_Racer_phase_II (9 sessions, 2026-07-13: high-rate
# doublets, coupling+flips, 2.6 m drops, 4.3 m/s dashes; AUW 38.5 g).
# Torques from per-BURST
# LSQ on the scripted doublets/flips (high S/N beats both the chirp-2SLS D-term bias
# and trajectory-fit EIV attenuation); yaw damping from sync-spin steady plateaus;
# thrust + inflow + drag calibrated against MODELLED W(U) so the coefficients are
# PIPELINE-consistent (the sim never sees measured RPM — the motor curve's biases
# must live inside k_w/k_hor, and do). Full provenance lives with the airframe's
# physics write-up and the dataset READMEs.
# Real rotor speed -> plant units by s = 3000/w_max_real; already converted below.
#
# Headlines vs the Phase-I-only fit (git history of this function):
#   k_p    3.75e-5 -> 2.72e-5 real (plant 9.73e-5): the chirp-regime value was ~35 %
#          hot for real maneuvers. Bursts: chirps 3.9e-5, 460-560°/s doublets
#          2.7-2.9e-5 (operating point, SHIPPED), deep flips 1.8e-5. The spread is
#          rate + aero-state softening the plant can't represent -> DR envelope.
#   k_q    ~unchanged 2.47e-5 real (pitch barely softens; k_p/k_q asymmetry at the
#          operating point is NOT the inertia ratio — see J_z).
#   yaw    k_r -3.22e-2 -> -1.78e-2 real and k_dr 8.15 -> 1.70: the chirp-2SLS pair
#          was jointly inflated (collinear at chirp frequencies). Sync-spin plateaus
#          refute k_dr=8.15 outright (2.7 rad/s needs <350 rad/s of differential).
#          J_z=-0.192 fitted clean (yaw 2SLS R² 0.98 — rotor gyro can't touch yaw).
#   J_x/J_y  set physical (-0.75/+0.83 from Iyy/Ixx≈1.45, Izz/Ixx≈2.2); free fits
#          are unphysical proxies. k_dp=k_dq=0 (free fits give ANTI-damping = eRPM
#          phase-lag artefact, would destabilise the sim).
#   k_w    calibrated through the pipeline: Vicon hover force / ΣW_model(U)². The
#          IMU under-reads |a| by ~8 % and the motor curve maps hover U ~4.5 % high
#          in W — Phase-I's acc-based k_w only worked because those cancelled.
#          Hover throttle ≈ 0.265, TWR ≈ 6.0 mid-pack (sag ~13 %/pack NOT modelled).
#   inflow k_angle=2.4, k_hor=2.0, k_v2=0: selected on the regime-weighted replay
#          metric inside the row-plausible box. Truth is asymmetric: climb obeys
#          momentum theory (~2.3) but powered descents show NO inflow gain (VRS) and
#          idle-W falls are near-ballistic to the -4.5 m/s measured (hence k_v2=0 —
#          sink faster than ~5 m/s is unmeasured extrapolation, DR territory).
#          Steep POWERED descent (<-2 m/s) stays ~+3 m/s² over-modelled regardless.
#   drag   linear-in-v·ΣW wins to 5 m/s (dashes): quad terms fit 0 with inconsistent
#          signs — the ducted frame is linear-dominated. k_x2/k_y2 stay 0.
#
# Validation (GT-attitude 1 s force replay, 2212 windows over BOTH campaigns,
# old -> new): ALL p50 0.258 -> 0.222 m/s; drops 0.434 -> 0.275 (p90 1.03 -> 0.84);
# dash 0.235 -> 0.197. Rate holdout (agile session, never fitted): burst k_eff
# 2.79e-5/2.23e-5 vs shipped 2.72e-5/2.47e-5. Hover-trim note: the real FC hovers
# with ~+19 roll / -40 pitch rad/s²-equivalent of motor-trim asymmetry (per-motor
# k + CoG offset) — kept OUT of these symmetric coefficients (locked decision);
# the firmware I-term absorbs it in sim exactly as on the bench.
_S_SYSID = 3000.0 / 5678.0   # plant-units per real rad/s at the fitted w_max


def air75_ii_racer() -> PlantParams:
    """BETAFPV Air75 II Racer — every coefficient FITTED from real Vicon flight
    (Phase I+II, 2026-07-13). THE single source of truth for the Racer's numbers;
    see the module-note block above for what each value means and how it was
    identified. The eeprom PIDs were tuned on the real drone, so closed-loop sim
    behaviour with these params is the self-consistent pairing.
    """
    return PlantParams(
        k_w=1.644e-06, k=0.572, w_min=75.5, w_max=3000.0,
        tau=0.0392, tau_down=0.0392, r_prop=0.0379,   # symmetric lag (== tau)
        k_x=1.529e-04, k_y=1.866e-04, k_x2=0.0, k_y2=0.0,
        k_v2=0.0, k_angle=2.4, k_hor=2.0,
        k_p1=9.734e-05, k_p2=9.734e-05, k_p3=9.734e-05, k_p4=9.734e-05,
        k_q1=8.838e-05, k_q2=8.838e-05, k_q3=8.838e-05, k_q4=8.838e-05,
        k_r1=-3.367e-02, k_r2=-3.367e-02, k_r3=-3.367e-02, k_r4=-3.367e-02,
        k_r5=-2.044e-03, k_r6=-2.044e-03, k_r7=-2.044e-03, k_r8=-2.044e-03,
        J_x=-0.75, J_y=0.83, J_z=-0.192,
        k_dp=0.0, k_dq=0.0, k_dr=1.70,
    )


AIRFRAME_PARAMS = {
    "air75_ii_racer": air75_ii_racer,
}


def register_airframe(name: str, factory: Callable[[], PlantParams]) -> None:
    """Register ``factory`` under ``name`` so ``env.airframe=<name>`` resolves to it.

    The supported way to fly a drone SkyFlow does not ship: identify your own
    coefficients, wrap them in a function returning :class:`PlantParams`, and register
    it. The plant is airframe-agnostic — nothing downstream of here special-cases the
    built-ins. Re-registering a name replaces it.
    """
    AIRFRAME_PARAMS[name] = factory


def airframe_params(name: str) -> PlantParams:
    """Resolve an airframe name (``SkyFlowConfig.airframe``) to its :class:`PlantParams`."""
    try:
        return AIRFRAME_PARAMS[name]()
    except KeyError:
        raise ValueError(
            f"unknown airframe {name!r}; expected one of {sorted(AIRFRAME_PARAMS)}. "
            f"Register your own with skyflow.params.register_airframe({name!r}, factory)."
        ) from None


# Domain-randomisation brackets (Table III): motor limits ±20 %, everything else
# ±30 %, sampled per-world at reset and multiplied onto the nominal row. `k` is
# additionally clipped to [0, 1] (it is a blend weight).
_DR_WIDE = 0.30   # ±30 %
_DR_MOTOR = 0.20  # ±20 % (w_min, w_max)


def randomization_scale(key: jax.Array, fleet: int, scale: float = 1.0) -> jax.Array:
    """Per-world multiplicative jitter for the param row → ``[F, len(_PARAM_KEYS)]``.

    ``scale`` master-scales the spread toward 1.0 (0 = no DR, identical worlds;
    1 = the paper's brackets; >1 widens). Returned factors multiply
    ``PlantParams.to_array()``; the caller clips ``k`` to [0, 1] afterwards.
    """
    kw, km = jax.random.split(key)
    lo_wide, hi_wide = 1.0 - scale * _DR_WIDE, 1.0 + scale * _DR_WIDE
    lo_mot, hi_mot = 1.0 - scale * _DR_MOTOR, 1.0 + scale * _DR_MOTOR
    factors = jax.random.uniform(kw, (fleet, len(_PARAM_KEYS)), minval=lo_wide, maxval=hi_wide)
    # w_min (idx 29), w_max (idx 30) use the tighter motor bracket
    mot = jax.random.uniform(km, (fleet, 2), minval=lo_mot, maxval=hi_mot)
    factors = factors.at[:, 29:31].set(mot)
    factors = factors.at[:, 36:39].set(1.0)   # S-curve map params are a fixed shape — never DR'd
    factors = factors.at[:, 43].set(1.0)      # eta_floor is a clip rail, not physics — never DR'd
    return factors


assert tuple(f.name for f in fields(PlantParams)) == _PARAM_KEYS, (
    "PlantParams field order must match _PARAM_KEYS (the dynamics param layout)"
)
