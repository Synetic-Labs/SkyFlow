"""SkyDreamer quadrotor dynamics as a pure-JAX, fleet-batched plant.

A faithful JAX/RK4 port of the analytic model in the SkyDreamer paper (Diermayr
et al. 2025, arXiv 2510.14783) — the same closed-form equations of motion the
authors integrate (their sim is JAX + RK4 @ 2.2 ms), reimplemented here so the
whole rollout stays inside one jitted ``lax.scan`` on the GPU. It is a direct
translation of a numba reference implementation of the same equations of motion,
cross-checked term-for-term against it.

Frames. World is Z-up (gravity −z); body is FLU (x forward, y left, z up), with
thrust along +body-z and the quaternion Hamilton wxyz. This is the SkyDreamer
convention. When the plant is driven behind a Betaflight flight controller
(``env.py``), the synthesised IMU is converted to the firmware's NED / FRD frame
by the ``(x, −y, −z)`` flip (see :func:`synth_sensors`) — the same flip crazyflow
uses at its firmware seam.

State layout (18): ``[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r, w1..w4, d_wake]``
— world position, world velocity, attitude quaternion, body rates, 4 normalised
motor states in [−1, 1], and the wake (dynamic-inflow) state ``d_wake`` = deviation
of the developed induced-flow scale s from its rotor-implied equilibrium
rt = √(k_w·ΣW²). Stored as the DEVIATION so that 0 = equilibrium — a fresh
``make_state`` starts wake-settled with no spawn transient, and airframes with
``k_wake = 0`` are bit-identical to the 17-state model in every output.

Actuator input. The model's motor command is ``U ∈ [0, 1]`` per rotor. In the
paper the *policy* sets it directly; here **Betaflight sets it** — the firmware's
per-motor output (already in [0, 1]) is fed straight in as ``U`` (see
:func:`step`). The motor's own first-order lag (``tau``) and sqrt thrust curve
stay in the plant, so we do NOT double-count actuator dynamics (Betaflight emits
an instantaneous command; the physical motor lag lives here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .params import _PARAM_KEYS as _PK

# Motor-state normalisation constants (paper / numba reference): the stored motor
# state w_i ∈ [−1, 1] maps to an actual RPM in [W_MIN_N, W_MAX_N].
_W_MIN_N = 0.0
_W_MAX_N = 3000.0
_G = 9.81

# IMU synthesis (shared with crazyflow's firmware seam).
_SEA_LEVEL_PA = 101325.0
_BARO_SCALE_M = 8434.0

STATE_DIM = 18

# Wake BUILD lag (s): the developed inflow tracks a RISING rotor thrust essentially
# at the motor timescale (measured: no measurable extra lag on up-steps — the
# high-thrust rotor drives its column up quickly); only the DECAY after a thrust
# cut is slow (``tau_wake``, ~114 ms: the air column's momentum persists). Constant,
# not a param: it is the "no extra dynamics" limit, inert whenever k_wake = 0.
_TAU_WAKE_BUILD = 0.02


def _flip_xyz(v: jax.Array) -> jax.Array:
    """FLU↔FRD body / Z-up↔NED world: negate y and z. Batched ``[..., 3]``."""
    return v * jnp.array([1.0, -1.0, -1.0], v.dtype)


def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Hamilton product of wxyz quaternions, batched ``[..., 4]``."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_normalize(q: jax.Array) -> jax.Array:
    """Normalise wxyz quaternions, falling back to identity near zero. Batched."""
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    ident = jnp.zeros_like(q).at[..., 0].set(1.0)
    return jnp.where(norm < 1e-9, ident, q / jnp.maximum(norm, 1e-12))


def rot_matrix(q: jax.Array) -> jax.Array:
    """Body→world rotation matrix R(q), batched ``[F, 4] -> [F, 3, 3]``."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = jnp.stack([
        1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
        2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx),
        2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2),
    ], axis=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def euler_to_quat(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    """ZYX Euler (rad) → wxyz quaternion, batched (broadcasts the three inputs)."""
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    return jnp.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


# Derivative clamp for safe_sqrt: d/dx sqrt(x) is capped at 1/(2·√eps) = 500. Normal-
# regime slopes here are O(1), so the cap leaves ~3 orders of headroom for real signal
# while keeping BPTT finite; APG's global-norm clip absorbs the residual spikes.
_SAFE_SQRT_EPS = 1e-6


@jax.custom_jvp
def safe_sqrt(x: jax.Array) -> jax.Array:
    """sqrt whose FORWARD value is bit-exact ``jnp.sqrt`` (numba-parity preserved) but
    whose derivative is clamped near 0 — sqrt'(0) = ∞ would NaN BPTT (APG). Same shape
    as ``brax.math.safe_arcsin``: the clip lives only in the tangent."""
    return jnp.sqrt(x)


@safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sqrt(x), dx / (2.0 * jnp.sqrt(jnp.maximum(x, _SAFE_SQRT_EPS)))


def safe_norm(x: jax.Array, axis: int = -1) -> jax.Array:
    """L2 norm with a finite (zero) gradient at x = 0, unlike ``jnp.linalg.norm``
    whose vjp is x/‖x‖ = NaN at an exactly-zero row (jax#3058 — hit for real at
    rest spawns: zero velocity/rates). Per-row, fleet-batched: brax's
    ``safe_norm`` gates on a GLOBAL allclose and is only safe under vmap."""
    return safe_sqrt(jnp.sum(x * x, axis=axis))


def commanded_rotor_speed(U: jax.Array, k: jax.Array, k_w: jax.Array,
                          w_min: jax.Array, w_max: jax.Array, sc_tmax: jax.Array,
                          sc_u50: jax.Array, sc_p: jax.Array) -> jax.Array:
    """Steady-state rotor-speed target Wc [F, 4] (RPM) for motor commands U [F, 4].

    The static command→rotor map: an optional S-shaped Hill thrust curve when
    ``sc_tmax > 0`` (deadband + saturation the sqrt-blend can't represent; per-motor
    target Wc = √(T_hill(u)/(4·k_w)) so k_w·ΣWc² reproduces the measured
    T_hill(u) = sc_tmax·u^p/(u50^p + u^p)), else the paper's sqrt blend. Args after
    ``U`` are the [F] parameter columns. Inverse of :func:`duty_equivalent`. Shared
    by :func:`_deriv` and the estimation force-model adapter
    (``force_model.SkyflowSpecificForce``) — keep single-source.
    """
    Uc = jnp.clip(U, 0.0, 1.0)
    Wc_blend = ((w_max - w_min)[:, None]
                * safe_sqrt(k[:, None] * Uc**2 + (1.0 - k)[:, None] * Uc)
                + w_min[:, None])
    up = Uc ** sc_p[:, None]
    hill = sc_tmax[:, None] * up / (sc_u50[:, None] ** sc_p[:, None] + up + 1e-9)
    Wc_s = safe_sqrt(jnp.clip(hill, 0.0, None) / (4.0 * k_w[:, None] + 1e-12))
    return jnp.where(sc_tmax[:, None] > 0.0, Wc_s, Wc_blend)


def _deriv(state: jax.Array, U: jax.Array, p: jax.Array
           ) -> tuple[jax.Array, jax.Array]:
    """State derivative + body-frame specific force, batched over the fleet.

    ``state`` [F, 18], ``U`` [F, 4] motor commands in [0, 1], ``p`` [F, 46]
    parameter rows (see params._PARAM_KEYS order). Returns ``(dstate [F, 18],
    spec_body [F, 3])`` where spec_body = [Dx, Dy, T] is the non-gravity body
    acceleration an accelerometer measures (FLU body).
    """
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
    quat = state[:, 6:10]
    pr, qr, rr = state[:, 10], state[:, 11], state[:, 12]
    w = state[:, 13:17]                                   # normalised motor states [F, 4]
    d_wake = state[:, 17]                                 # induced-flow deviation s − rt

    (k_x, k_y, k_w, k_x2, k_y2, k_angle, k_hor, k_v2,
     k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4,
     k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8,
     J_x, J_y, J_z, tau, k, w_min, w_max, r_prop,
     k_dp, k_dq, k_dr, tau_down, sc_tmax, sc_u50, sc_p,
     k_lin, k_lin2, k_wake, tau_wake, eta_floor, k_axg, k_ru) = (p[:, i] for i in range(46))

    # motor state (normalised) -> actual RPM; command U -> commanded RPM (the static
    # command→rotor map, Hill or sqrt-blend — see commanded_rotor_speed).
    W = (w + 1.0) / 2.0 * (_W_MAX_N - _W_MIN_N) + _W_MIN_N          # [F, 4]
    Wc = commanded_rotor_speed(U, k, k_w, w_min, w_max, sc_tmax, sc_u50, sc_p)
    # Asymmetric first-order rotor lag: spin-up (Wc>W) uses tau, spin-down uses
    # tau_down. Equal => the paper's single-tau model (air75). A rotor can spin
    # down faster than it spins up (a measured 45 ms up / 22 ms down asymmetry on
    # one simulated airframe). safe_sqrt/where stays smooth
    # for BPTT — the branch is on the sign of (Wc−W), not a non-diff of the rate.
    tau_eff = jnp.where(Wc > W, tau[:, None], tau_down[:, None])    # [F, 4]
    d_W = (Wc - W) / tau_eff                                        # [F, 4]
    # NOTE a spin-UP SLEW-LIMIT extension (d_W capped on rises; fast small-signal tau)
    # was hypothesised, implemented and REJECTED 2026-07-23: the launch tape is a
    # gradual ramp (never engages a slew) and a clean from-idle collective step shows
    # the pure first-order rise at tau=40 ms matches the measured response (t63 56 ms
    # both, best-fit tau 40 ms). Don't re-add without new step-response evidence.

    W1, W2, W3, W4 = W[:, 0], W[:, 1], W[:, 2], W[:, 3]
    dW1, dW2, dW3, dW4 = d_W[:, 0], d_W[:, 1], d_W[:, 2], d_W[:, 3]

    R = rot_matrix(quat)                                            # body->world [F,3,3]
    # body velocity = R^T v
    v_world = jnp.stack([vx, vy, vz], axis=-1)
    v_body = jnp.einsum("fji,fj->fi", R, v_world)                   # R^T v
    vbx, vby, vbz = v_body[:, 0], v_body[:, 1], v_body[:, 2]

    w_bar = jnp.mean(W, axis=1)
    w_sum = jnp.sum(W, axis=1)
    w_sum_sq = jnp.sum(W**2, axis=1)

    denom = r_prop * w_bar + 1e-6
    alpha = jnp.arctan2(vbz, denom)
    v_hor = safe_sqrt(vbx**2 + vby**2)
    mu_hor = jnp.arctan2(v_hor, denom)

    Dx = -k_x * vbx * w_sum - k_x2 * vbx * jnp.abs(vbx)
    Dy = -k_y * vby * w_sum - k_y2 * vby * jnp.abs(vby)
    # Axial-inflow term must DAMP vertical motion: climbing (vbz>0 in FLU) raises the
    # rotor inflow, which REDUCES thrust at fixed RPM (momentum theory) → natural vD
    # damping. The ported sign (+k_angle·α) did the opposite — climb ADDED thrust →
    # exponential vertical runaway ("no gravity / thrust weird"). Negate to restore it.
    #
    # The linear-in-angle inflow correction is a FIT, valid only for the small advance
    # angles the paper's race data covers (|α|, μ ≲ 0.25 rad). arctan2 lets the angles
    # reach ±π/2 (e.g. fast descent at idle RPM → tiny denom), where the extrapolated
    # line is wildly unphysical: thrust −4× in a climb (motors sucking the drone down,
    # harder with MORE throttle) or +6× in a descent. Rail the total multiplier to a
    # momentum-theory-plausible band instead: fixed-RPM thrust decays toward ~0 in a
    # fast climb and grows toward windmill-brake ~2× in a fast descent, never reverses.
    #
    # k_lin/k_lin2 are the alternative SPEED-polynomial deficit (thrust loss ∝
    # T·(k_lin·v + k_lin2·v·|v|), momentum-theory power balance): ground-truth system
    # identification showed the per-duty inflow-ratio slopes are constant-in-v per unit
    # T — NOT ∝ 1/tip-speed as the α-form assumes (implied k_angle drifted 1.31→3.41
    # across duty). Airframes pick one form (or mix): k_angle=0 + k_lin/k_lin2>0 where
    # that was measured; 0 (default) keeps the others bit-identical.
    #
    # k_wake is the DYNAMIC-INFLOW deficit: the developed wake scale s = rt + d_wake
    # lags the rotor-implied equilibrium rt = √(k_w·ΣW²) — instantaneous on rises
    # (_TAU_WAKE_BUILD), slow (tau_wake ≈ 114 ms) after thrust cuts, where the air
    # column's momentum persists and chokes the rotor: thrust transiently UNDERSHOOTS
    # its destination (measured: −4..−9 m/s² × ~150 ms after collective down-steps,
    # through ZERO and beyond — hence eta_floor < 0 for such an airframe; a first-order
    # motor lag cannot undershoot). Deficit ∝ flow-MOMENTUM ratio (s/rt)² − 1 (the ratio²
    # form beat ratio 0.68 vs 1.21 shape loss). d_wake ẋ needs ṙt (chain through the
    # motor states): ḋ = −d/τ_eff − k_w·Σ(W·Ẇ)/rt.
    rt = jnp.maximum(safe_sqrt(k_w * w_sum_sq), 1e-3)
    ratio = jnp.maximum(1.0 + d_wake / rt, 0.0)
    wake_def = k_wake * (ratio**2 - 1.0)
    tau_w_eff = jnp.where(d_wake > 0.0, tau_wake, _TAU_WAKE_BUILD)
    d_dwake = -d_wake / tau_w_eff - k_w * jnp.sum(W * d_W, axis=1) / rt

    # k_axg: Tq-coupled axial deficit (Glauert advance ratio vbz/√T, rt=√(k_w·ΣW²) is the
    # induced-velocity scale ∝√T). Shrinks as thrust rises — the 2-D (vcl,Tq) over-lift
    # surface k_lin's constant slope can't capture. 0 → inert.
    inflow = jnp.clip(1.0 - k_angle * alpha - k_lin * vbz - k_lin2 * vbz * jnp.abs(vbz)
                      - k_axg * vbz / rt - wake_def + k_hor * mu_hor, eta_floor, 2.5)
    T = k_w * inflow * w_sum_sq - k_v2 * vbz * jnp.abs(vbz)
    spec_body = jnp.stack([Dx, Dy, T], axis=-1)                     # [F, 3]

    # ...+ (−k_d·ω) angular-rate damping (0 for the Air75 II Racer; models the aero
    # rate saturation a simulator twin can show). Opposes the body rate on each axis.
    Mx = -k_p1 * W1**2 - k_p2 * W2**2 + k_p3 * W3**2 + k_p4 * W4**2 + J_x * qr * rr - k_dp * pr
    My = -k_q1 * W1**2 + k_q2 * W2**2 - k_q3 * W3**2 + k_q4 * W4**2 + J_y * pr * rr - k_dq * qr
    # yaw reaction torque. PHYSICAL form (real hardware): linear-in-rotor-speed k_r·W (SkyDreamer
    # default; the real Air75 uses it) — yaw authority varies with collective, as a real prop's does.
    # SIM-FIDELITY form (k_ru): a throttle-FLAT command-linear torque, for matching a game-style
    # simulator whose yaw is ∝ command rather than prop reaction — do NOT enable on a real airframe;
    # use k_r there. The two are additive, selected by which coeff is nonzero (see params.py
    # PHYSICAL-vs-SIM-FIDELITY note). A simulator twin may set k_r=0, k_ru≠0 when a measured
    # yaw-authority grid shows its yaw flat across collective (∝W droops, ∝W² humps at the Hill
    # inflection, only ∝command is flat). The flat term drives off the Hill-INVERSE of
    # the (lagged) rotor state W — an "effective duty" ue: linear in the pilot command (⇒ flat
    # authority) yet carrying the motor spin-up LAG through W, so both the authority shape and the
    # transient match the measured sim (raw-command has no lag → yaw crashes on command drop). Guarded
    # for the sc_tmax=0 (no-Hill) airframes. All extra coeffs default 0 → linear-only, bit-identical.
    # ``smax`` is a SAFE positive upper thrust: real sc_tmax where a Hill map exists, else a big
    # constant so the sc_tmax=0 (sqrt-blend) airframes keep a POSITIVE, finite ue base — otherwise
    # (sc_tmax−Tc) goes negative and ue=base^(1/sc_p) is NaN, whose gradient poisons the jnp.where
    # mask (the classic where-NaN trap → the grad-finite-at-idle test). Floor Tc AWAY from 0 too:
    # ue ∝ Tc^(1/sc_p) has infinite slope at Tc→0 (idle, W≈0); flight thrust is ≥~11/motor so the
    # 1.0 floor never bites in the operating band, it only removes the idle-gradient singularity.
    has_hill = sc_tmax[:, None] > 0.0
    smax = jnp.where(has_hill, sc_tmax[:, None], 1.0e6)
    T_mot = 4.0 * k_w[:, None] * W**2
    Tc = jnp.clip(T_mot, 1.0, smax * 0.999)
    ue = sc_u50[:, None] * (Tc / (smax - Tc)) ** (1.0 / (sc_p[:, None] + 1e-9))
    ue = jnp.where(has_hill, ue, 0.0)                              # only defined with a Hill map
    ue1, ue2, ue3, ue4 = ue[:, 0], ue[:, 1], ue[:, 2], ue[:, 3]
    Mz = (-k_r1 * W1 + k_r2 * W2 + k_r3 * W3 - k_r4 * W4
          + k_ru * (-ue1 + ue2 + ue3 - ue4)
          - k_r5 * dW1 + k_r6 * dW2 + k_r7 * dW3 - k_r8 * dW4 + J_z * pr * qr - k_dr * rr)

    # world acceleration = R·[Dx, Dy, T] − g ẑ
    acc_world = jnp.einsum("fij,fj->fi", R, spec_body)
    acc_world = acc_world.at[:, 2].add(-_G)

    # quaternion rate = 0.5 · q ⊗ [0, p, q, r]
    omega_quat = jnp.stack([jnp.zeros_like(pr), pr, qr, rr], axis=-1)
    dquat = 0.5 * quat_mul(quat, omega_quat)

    # motor-state rate: RPM rate rescaled back into the [−1, 1] normalisation
    dw = d_W / (_W_MAX_N - _W_MIN_N) * 2.0

    dstate = jnp.concatenate([
        v_world,                                    # dpos
        acc_world,                                  # dvel
        dquat,                                      # dquat
        jnp.stack([Mx, My, Mz], axis=-1),           # drates
        dw,                                         # dmotors
        d_dwake[:, None],                           # dwake
    ], axis=-1)
    return dstate, spec_body


def rk4_step(state: jax.Array, U: jax.Array, p: jax.Array, dt: float) -> jax.Array:
    """One RK4 integration step of the plant; quaternion renormalised after.

    ``U`` (motor command) is held constant across the step (zero-order hold),
    matching a firmware output that is fixed within a 1 ms substep.
    """
    k1, _ = _deriv(state, U, p)
    k2, _ = _deriv(state + 0.5 * dt * k1, U, p)
    k3, _ = _deriv(state + 0.5 * dt * k2, U, p)
    k4, _ = _deriv(state + dt * k3, U, p)
    new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    quat = quat_normalize(new[:, 6:10])
    return new.at[:, 6:10].set(quat)


def step(state: jax.Array, motors: jax.Array, p: jax.Array, dt: float,
         motor_perm: jax.Array | None = None) -> jax.Array:
    """Advance the plant by ``dt`` under Betaflight motor outputs.

    ``motors`` [F, 4] ∈ [0, 1] are the firmware's per-rotor commands. They are
    reordered by ``motor_perm`` (Betaflight motor index → SkyDreamer W1..W4;
    the identity by default — see README.md §Calibration, this MUST be verified)
    and fed as the model's command ``U``.
    """
    U = motors if motor_perm is None else motors[:, motor_perm]
    return rk4_step(state, U, p, dt)


def duty_equivalent(state: jax.Array, p: jax.Array) -> jax.Array:
    """Per-rotor DUTY-EQUIVALENT of the current (lagged) rotor state: the command
    u ∈ [0, 1] whose steady-state rotor target equals the rotor's speed right now —
    i.e. the static command→Wc map inverted at W. This is what an actuator-status
    telemetry wire typically reports (measured on a step tape: a lagged response
    that SETTLES AT THE COMMAND, in duty units), so the motor obs built from this
    matches the deploy wire in units AND transient, for both thrust maps:

      * Hill S-curve (sc_tmax>0):  T = 4·k_w·W²,  u = u50·(T/(tmax−T))^(1/p)
      * sqrt blend:  Wn = √(k·u² + (1−k)·u)  →  the positive root of the quadratic

    Returns [F, 4] in [0, 1]. The armed idle floor (~0.05 duty) is applied by the
    caller (task obs), not here."""
    w = state[:, 13:17]                                             # normalised [-1, 1]
    W = (w + 1.0) / 2.0 * (_W_MAX_N - _W_MIN_N) + _W_MIN_N          # [F, 4] rotor speed
    keys = {k_: i for i, k_ in enumerate(_PK)}
    k_w = p[:, keys["k_w"]][:, None]
    kb = p[:, keys["k"]][:, None]
    w_min = p[:, keys["w_min"]][:, None]
    w_max = p[:, keys["w_max"]][:, None]
    sc_tmax = p[:, keys["sc_tmax"]][:, None]
    sc_u50 = p[:, keys["sc_u50"]][:, None]
    sc_p = p[:, keys["sc_p"]][:, None]

    # Hill inverse: T(W) back through T_hill
    T = 4.0 * k_w * W**2
    frac = jnp.clip(T / jnp.clip(sc_tmax - T, 1e-6, None), 0.0, None)
    u_hill = jnp.clip(sc_u50 * frac ** (1.0 / jnp.clip(sc_p, 1e-3, None)), 0.0, 1.0)

    # sqrt-blend inverse: k·u² + (1−k)·u − Wn² = 0 (positive root; k→0 → u = Wn²)
    Wn = jnp.clip((W - w_min) / jnp.clip(w_max - w_min, 1e-6, None), 0.0, 1.0)
    kc = jnp.clip(kb, 1e-6, 1.0)
    u_blend = jnp.clip(
        (-(1.0 - kc) + safe_sqrt((1.0 - kc) ** 2 + 4.0 * kc * Wn**2)) / (2.0 * kc), 0.0, 1.0)

    return jnp.where(sc_tmax > 0.0, u_hill, u_blend)


def specific_force_body(state: jax.Array, p: jax.Array) -> jax.Array:
    """Body-frame specific force [F, 3] (accelerometer signal, FLU).

    Depends only on the motor *state* (W) and velocity, not the command U, so no
    motor input is needed — an accelerometer reads the thrust the spun-up rotors
    actually produce.
    """
    zeros = jnp.zeros((state.shape[0], 4), state.dtype)
    _, spec = _deriv(state, zeros, p)
    return spec


def synth_sensors(state: jax.Array, p: jax.Array) -> jax.Array:
    """Synthesise the Betaflight IMU + baro packet [F, 7] in NED / FRD.

    Columns: gyro (body rates, rad/s) [0:3], specific force (m/s²) [3:6], baro
    pressure (Pa) [6:7]. SkyDreamer is FLU/Z-up; the firmware wants FRD/NED, so
    body rates and specific force are flipped ``(x, −y, −z)`` and the baro is the
    isothermal atmosphere at the world altitude — identical recipe to
    the crazyflow reference env's sensor synthesis, just sourced from the SkyDreamer plant.
    """
    gyro_frd = _flip_xyz(state[:, 10:13])
    spec_frd = _flip_xyz(specific_force_body(state, p))
    alt = state[:, 2:3]                                  # world z (up)
    baro = _SEA_LEVEL_PA * jnp.exp(-alt / _BARO_SCALE_M)
    return jnp.concatenate([gyro_frd, spec_frd, baro], axis=1).astype(jnp.float32)


def make_state(pos: jax.Array, vel: jax.Array, quat: jax.Array,
               rates: jax.Array, motors: jax.Array,
               wake: jax.Array | None = None) -> jax.Array:
    """Assemble a [F, 18] state from its parts (quaternion wxyz, normalised motors).

    ``wake`` is the induced-flow DEVIATION from equilibrium (see module docstring);
    None (the default) starts wake-settled — correct for spawns and GT reseeds.
    """
    if wake is None:
        wake = jnp.zeros((pos.shape[0], 1), pos.dtype)
    return jnp.concatenate([pos, vel, quat, rates, motors, wake],
                           axis=-1).astype(jnp.float32)


def to_pose(state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """(pos, vel, quat_wxyz, body_rates) in the SkyDreamer (Z-up/FLU) frame."""
    return state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]


# 180° rotation about body-x (wxyz): conjugating by it maps FLU↔FRD body and
# Z-up↔NED world in one shot — the quaternion twin of _flip_xyz. Self-inverse,
# so the NED→FLU spawn conversion reuses it.
_QF_FLIP = jnp.array([0.0, 1.0, 0.0, 0.0], jnp.float32)


def _quat_flip(quat: jax.Array) -> jax.Array:
    qf = jnp.broadcast_to(_QF_FLIP, quat.shape)
    return quat_mul(qf, quat_mul(quat, qf))


def pose_ned(state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """(pos, vel, quat_wxyz, body_rates) in NED/FRD — the frame the gate task,
    analytic renderer and reward all operate in (matches crazyflow's _ned_state).
    The plant integrates in Z-up/FLU; this converts its pose for the task layer."""
    pos, vel, quat, rates = to_pose(state)
    return _flip_xyz(pos), _flip_xyz(vel), _quat_flip(quat), _flip_xyz(rates)


def pose_from_ned(pos_ned: jax.Array, vel_ned: jax.Array, quat_ned: jax.Array,
                  rates_ned: jax.Array) -> tuple[jax.Array, ...]:
    """Inverse of :func:`pose_ned` — build plant-frame pose parts from NED (for
    spawning: the gate/spawn geometry is authored in NED). The flips are self-inverse."""
    return (_flip_xyz(pos_ned), _flip_xyz(vel_ned), _quat_flip(quat_ned),
            _flip_xyz(rates_ned))
