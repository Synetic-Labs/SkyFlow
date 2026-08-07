"""Faithful SkyDreamer reward, fleet-batched in JAX.

A JAX port of the paper-faithful gate-reward math of SkyDreamer
(Diermayr et al. 2025, arXiv 2510.14783):

    r_t = 5·r_prog − r_rate + r_gate

* **r_gate — continuous transit credit.** The gate volume (thickness ``t_g``)
  is a slab of depth ``t_g`` along the gate normal, centred on the opening. As
  the drone sweeps *forward* through the slab it earns ``(90/t_g)·(1 −
  chebyshev/d_g)₊`` per metre of NEW depth crossed (Chebyshev = max
  lateral/vertical miss from centre, clipped to the opening half-size ``d_g``;
  NEW = beyond the carried ``depth_ratchet``, the anti-farming telescoping).
  Because the task advances the target on the CENTRE crossing, the pass step
  also CASHES OUT the remaining depth to the far face, so every forward pass
  totals exactly ``90·centering`` at any speed — verified equal to running the
  paper's three sampling planes (pre-gate/centre/post-gate, 30 each) to full
  traverse at every lateral offset. Taking that discretisation to its continuum
  limit keeps the forward intent (same total, same square-Chebyshev lateral
  centring, same disjoint hand-off with r_prog) while giving first-order
  (APG/BPTT) learners a **dense, unbiased** gradient through the whole volume:
  forward value *is* the backward slope — no step function, no straight-through
  surrogate. This is the de facto SkyDreamer gate reward used here (it replaces
  the original 3-plane +30 spikes).
* **r_prog** — reduction in distance to the pre-gate point, zeroed inside the
  gate volume (|signed_dist| < t_g/2) where r_gate takes over (disjoint support),
  "for ease of implementation" (paper).
* **r_rate** — ``1/(2·f_c·1e5)·(exp(min(‖Ω‖₁,17))−1)`` gyro-saturation penalty.

Optional extensions (off by default, keep the core paper-faithful): a
``calm_weight·‖Ω‖²`` calmness penalty, and a multiplicative ``speed_gain`` that
scales the ``5·r_prog + r_gate`` progress+gate term by a Gaussian slow-speed band
(full credit ≤ target, rolling off above) so the policy is paid only for SLOW
progress — the guard and calm penalty stay subtractive outside that scaling.

The discrete *events* — ``passed`` (centre plane crossed inside the opening) and
``gate_miss`` (any plane crossed outside it) — are unchanged from the paper's
3-plane crossing test; they drive metrics/termination, not the gradient, so they
stay discrete. ``plant.safe_norm`` keeps r_prog's norms NaN-safe (finite gradient)
at an exact-zero argument so a BPTT run can't be NaN-poisoned. r_prog and r_rate
are otherwise bit-identical to the paper.

This function is **frame-agnostic**: pass positions and the gate axes in one
consistent frame (the env uses NED, shared with the analytic renderer and
crossing classifier). It returns only the reward and the pass / miss events;
the env owns the terminal set (ground / out-of-bounds / firmware crash) and the
paper's "zero the reward on any terminal" rule, so ground/descent signs live in
exactly one place.

Single-gate scope for now; the math already generalises per-gate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import plant

_RATE_SAT = 17.0   # ‖Ω‖₁ clamp inside the exp (paper)

# Total transit credit for a perfectly centred pass — matches the paper's
# 3 planes × 30. The continuous term below delivers this as (90/t_g) per metre
# of forward depth swept through the volume, integrating to _GATE_CREDIT.
_GATE_CREDIT = 90.0


class RewardOut(NamedTuple):
    reward: jax.Array     # [F] float32 (before the env's terminal zeroing)
    passed: jax.Array     # [F] bool — centre plane crossed within the opening
    gate_miss: jax.Array  # [F] bool — a plane crossed outside the opening (frame hit)
    # Deepest clipped depth credited so far (the ratchet, see gate_reward). Thread it
    # back in as next step's ``depth_ratchet``; reset to −t_g/2 when the target gate
    # changes. On a pass step it reads +t_g/2 (the cash-out credited the whole slab).
    # Callers that don't thread it get the legacy (farmable) behaviour.
    depth: jax.Array      # [F] float32
    # -- miss-margin telemetry (metrics only; never touches the gradient) --------
    # A binary pass/miss says the drone failed but not by how much. These carry the
    # Chebyshev offset at the moment the CENTRE plane was crossed, so the env can
    # average it over crossings: << gate_half = threading it, ~gate_half = clipping
    # the frame, >> gate_half = not really aiming at the gate at all.
    crossed: jax.Array      # [F] bool — centre plane crossed this step (pass OR miss)
    cross_cheby: jax.Array  # [F] float32 — that crossing's offset, m (0 where no crossing)


def gate_reward(
    prev_pos: jax.Array, pos: jax.Array, gyro: jax.Array, *,
    gate_center: jax.Array, gate_normal: jax.Array, gate_right: jax.Array,
    gate_up: jax.Array, gate_half: float, gate_thickness: float,
    control_freq: float, prog_weight: float = 5.0, calm_weight: float = 0.0,
    vel: jax.Array | None = None,
    speed_gain_target: float = 0.0, speed_gain_sigma: float = 2.0,
    depth_ratchet: jax.Array | None = None,
) -> RewardOut:
    """Evaluate the SkyDreamer gate reward for the ``prev_pos → pos`` step.

    ``gate_*`` describe one gate in the caller's frame: centre, unit
    through-normal, unit lateral (right) and vertical (up) in-plane axes, opening
    half-size ``gate_half`` (d_g) and volume thickness ``gate_thickness`` (t_g).
    ``gyro`` is body rates (rad/s). Returns reward + pass/miss events.
    """
    seg = pos - prev_pos                                        # [F, 3]
    half_t = 0.5 * gate_thickness
    F = pos.shape[0]

    gate_miss = jnp.zeros(F, bool)
    passed = jnp.zeros(F, bool)

    # --- discrete pass / miss events (unchanged from the paper's 3-plane test) ---
    # These booleans drive metrics + termination only; the reward itself is the
    # continuous term below, so nothing here contributes to the gradient.
    for i, offset in enumerate((-half_t, 0.0, half_t)):
        plane_center = gate_center + offset * gate_normal       # [3]
        prev_sd = jnp.sum((plane_center - prev_pos) * gate_normal, axis=1)
        sd = jnp.sum((plane_center - pos) * gate_normal, axis=1)
        crossed = (prev_sd > 0.0) & (sd <= 0.0)
        denom = prev_sd - sd
        alpha = prev_sd / jnp.where(jnp.abs(denom) < 1e-9, 1e-9, denom)
        cp = prev_pos + alpha[:, None] * seg                   # crossing point [F, 3]
        off = cp - plane_center
        y_g = jnp.abs(jnp.sum(off * gate_right, axis=1))
        z_g = jnp.abs(jnp.sum(off * gate_up, axis=1))
        chebyshev = jnp.maximum(y_g, z_g)
        gate_miss = gate_miss | (crossed & (chebyshev > gate_half))
        if i == 1:                                             # centre plane = "the pass"
            passed = crossed & (chebyshev <= gate_half)
            centre_crossed = crossed
            cross_cheby = jnp.where(crossed, chebyshev, 0.0)

    # --- r_gate: continuous transit credit (continuum limit of the 3 planes) -----
    # Signed depth along the normal (positive = past the centre plane). Reward the
    # forward depth swept within the volume this step, weighted by lateral centring
    # evaluated at the current position (depth-independent). ∫ over a centred pass
    # = (90/t_g)·t_g·1 = 90, matching the paper's 3×30. Piecewise-linear (C⁰) in
    # position: dense, finite gradient everywhere inside the slab, zero outside it
    # (where r_prog carries the pull) and outside the opening (a miss earns none).
    #
    # RATCHETED (2026-07-27): credit only NEW maximum depth. Without the ratchet the
    # term paid every forward re-sweep — measured: oscillating inside the FRONT half
    # of the slab (never crossing the centre plane, so neither `passed` nor
    # `gate_miss` ever fires and the episode never ends) collected 44.4/cycle, a
    # discounted 286–422 vs 81 for honestly racing the course — a 3.5–5× dominant
    # strategy, made maximally attractive by the crawl profiles' slow-speed gain.
    # `depth_ratchet` is the deepest clipped depth already paid (carried in
    # GateTaskState.gate_depth, reset on gate advance); a centred transit still
    # integrates to exactly 90 because max-depth increments telescope, and the
    # forward gradient through the slab is unchanged wherever credit is live.
    s = jnp.sum((pos - gate_center) * gate_normal, axis=1)
    prev_s = jnp.sum((prev_pos - gate_center) * gate_normal, axis=1)
    clip_s = jnp.clip(s, -half_t, half_t)
    base = jnp.clip(prev_s, -half_t, half_t)
    if depth_ratchet is not None:
        base = jnp.maximum(base, depth_ratchet)
    swept = jnp.maximum(clip_s - base, 0.0)                    # NEW forward depth only
    depth_out = jnp.maximum(base, clip_s)
    off_c = pos - gate_center
    y_c = jnp.abs(jnp.sum(off_c * gate_right, axis=1))
    z_c = jnp.abs(jnp.sum(off_c * gate_up, axis=1))
    cheby_c = jnp.maximum(y_c, z_c)
    centering = jnp.clip(1.0 - cheby_c / gate_half, 0.0, None)
    # BACK-HALF CASH-OUT (2026-07-27): the task advances the target on the CENTRE
    # crossing, so the slab depth past wherever the crossing step lands was never
    # credited — measured, a pass collected 45·centering (front half) while the old
    # 3-plane code collected 60·centering (pre+centre planes) and the PAPER intends
    # 90·centering (all three). On the pass step, credit the remaining depth to the
    # far face at the crossing centering. Every forward pass now totals exactly
    # 90·centering at ANY speed (this also removes the speed-dependent overshoot
    # artifact, where faster passes collected more back-half by landing deeper).
    # A miss cashes out nothing (`passed` requires the opening), and the env zeroes
    # the step on the miss terminal regardless.
    swept = swept + jnp.where(passed, half_t - depth_out, 0.0)
    depth_out = jnp.where(passed, half_t, depth_out)
    reward = (_GATE_CREDIT / gate_thickness) * centering * swept

    # --- r_prog: distance reduction toward the pre-gate point, zeroed in volume ---
    pre_gate = gate_center - half_t * gate_normal
    signed_dist = jnp.sum((gate_center - pos) * gate_normal, axis=1)
    in_volume = jnp.abs(signed_dist) < half_t
    # plant.safe_norm: forward-identical to jnp.linalg.norm, but with a finite
    # gradient at an exactly-zero argument so a BPTT (APG) run can't be NaN-poisoned
    # by a coincidental exact hit of the pre-gate point.
    dist_pre = plant.safe_norm(pre_gate - pos, axis=1)
    prev_dist_pre = plant.safe_norm(pre_gate - prev_pos, axis=1)
    r_prog = jnp.where(in_volume, 0.0, prev_dist_pre - dist_pre)
    reward = reward + prog_weight * r_prog

    # --- multiplicative SLOW-SPEED GAIN (speed_gain_target > 0 → on) --------------
    # Scale the progress+gate reward by HOW SLOW the drone is: full credit at/below
    # `speed_gain_target`, smooth Gaussian roll-off above (width `speed_gain_sigma`).
    # This GATES the reward rather than adding to it (cf. the task's additive speed
    # band): no progress → ~0 reward regardless of speed (can't be farmed by
    # loitering), and a fast ballistic punch → gain→0 → earns nothing. So the policy
    # only gets paid for SLOW progress — a gentle, controlled approach and pass.
    # Applied to progress+gate ONLY; the r_rate guard and calm penalty below stay
    # SUBTRACTIVE (never scale a safety penalty by slowness). σ=2.0 keeps a signal
    # alive at launch speed (slow_band(5 m/s)≈0.22) so a cold policy can still learn
    # to reach the gate then slow down; DROP σ TO ~1.0 if it still flies too hot.
    if speed_gain_target > 0.0:
        speed = plant.safe_norm(vel, axis=1)
        over = jnp.maximum(0.0, speed - speed_gain_target)
        gain = jnp.exp(-0.5 * (over / speed_gain_sigma) ** 2)
        reward = reward * gain

    # --- r_rate: gyro-saturation guard (paper §II-C) — UNCHANGED, always applied.
    # Only bites near _RATE_SAT ≈ 17 rad/s (the sensor limit).
    gyro_l1 = jnp.sum(jnp.abs(gyro), axis=1)
    rate_scale = 1.0 / (2.0 * control_freq * 1e5)
    r_rate = rate_scale * (jnp.exp(jnp.minimum(gyro_l1, _RATE_SAT)) - 1.0)
    reward = reward - r_rate

    # --- r_calm: NEW gyro-calmness term, k·‖Ω‖² over all axes (calm_weight = 0 → off).
    # Added ON TOP of the guard above: bites smoothly across the operating range so it
    # discourages the unproductive launch spin, while a smooth low-rate line pays little.
    if calm_weight:
        reward = reward - calm_weight * jnp.sum(jnp.square(gyro), axis=1)

    return RewardOut(reward=reward.astype(jnp.float32), passed=passed, gate_miss=gate_miss,
                     depth=depth_out.astype(jnp.float32),
                     crossed=centre_crossed, cross_cheby=cross_cheby.astype(jnp.float32))
