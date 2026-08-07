"""Sensor noise for the analytic gate mask — make the perfect render look like
a real mask pipeline's output.

The reference "GateNet" is not a network: it is a classical HSV orange filter
over the camera frame (H 0–13, S ≥ 41, V ≥ 102, 2×2 opening, cyan racing-line
reclaim). Running that filter over real captured laps shows these artifact
families the clean render lacks:

1. **Band holes** — the gate face carries dark branding (lettering, checker
   decals, sponsor logos) that the filter excludes, punching chunks out of
   the frame band; distant gates fragment into partial rings.
2. **Occluders** — an indoor scene puts structure between drone and gate:
   pillars and ceiling trusses. Modeled as two black shapes: a thin
   band at ANY angle with a finite random length (a beam/edge crossing part of
   the view), and a RARE thick dead-straight full-span band that is either
   VERTICAL (a pillar up close) or HORIZONTAL (a ceiling truss / beam), chosen
   50/50.
3. **Glow false positives** — gate light bathing nearby surfaces reads as
   orange: occasional large amorphous blobs.
4. **Speckle** — faint scattered floor-glow dots (sub-pixel at 640×360, so
   they downsample to faint values, never full white).
5. **Resampling blur** — JPEG + downsample rounds small rings; the sim's
   crisp squares occasionally get a light 3×3 blur to match.

BINARY DISCIPLINE: the real mask is binary at 640×360; the 64×64 policy view
is its area-downsample, so intermediate values exist ONLY as sub-pixel
averaging at edges — never as flat grey. Every artifact here erases to 0 or
adds at 1.0, with a ~1 px anti-alias ramp at its boundary; only the speckle is
faint (it represents sub-pixel dots).

PERSISTENCE: a world's artifacts are a pure function of its per-family keys
(:data:`NOISE_FAMILIES` of them), so *holding* a key across control steps
freezes that artifact on screen — the way real decal holes, pillars and glow
stay in view for many frames instead of flickering in and out per step.
:func:`noise_state_init` / :func:`noise_state_step` carry (keys, ttl) through
an env's task state: each (world, family) draw lives a random 1..``hold``
frames, then is resampled. A caller that carries no noise state draws
:func:`fresh_noise_keys` every frame instead — same distribution, artifacts
redraw i.i.d. per frame.

:func:`corrupt_mask` applies all of it to a rendered soft mask ([F, H, W] in
[0, 1]): pure JAX, static shapes, branchless (Bernoulli gates are ``where``
masks), so it jits and runs inside the training scan at negligible cost next
to the ray-cast render. Outward glow widening of the band is handled
geometrically in ``render_masks(..., outer_grow=...)``, not here — tasks that
persist it draw the width from the extra :data:`GROW_FAMILY` key slot
(:func:`grow_from_keys`). Tuned by eye against real captured frames run
through the same HSV filter.

:func:`erasure_at` answers the INVERSE question — "is this pixel under a hole
or an occluder" — at arbitrary coordinates, without rendering. It exists so a
consumer that never sees the mask can stay consistent with it: the training
filter's synthetic corner detector samples it
at each projected gate corner, so a pillar that hides a gate from the policy's
CNN also hides it from the filter. The occluder shapes therefore have ONE
definition each (:func:`_stick_at`, :func:`_pillar_at`, called with grid
coordinates by the renderer and point coordinates by the sampler) and the
erasing families' defaults are module constants, so the two cannot drift apart.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Independent artifact families, one persistent key slot each (in order):
# band holes, occluder band, pillar, glow blob, speckle, blur.
NOISE_FAMILIES = 6

# -- ERASING-family defaults, shared by corrupt_mask and erasure_at ------------
# Module-level rather than inline defaults because TWO functions must agree on
# them: the renderer that paints these artifacts into the policy's mask, and
# :func:`erasure_at`, which asks whether a given pixel is under one. A copied
# literal in the second signature is a silent-drift hazard — the whole point of
# erasure_at is that the filter loses exactly the corners the policy's mask
# lost, and a stale duplicate of `pillar_width` would break that quietly.
# The ADDING families (glow blob, speckle) and the blur have no such twin, so
# their defaults stay inline on corrupt_mask.
HOLE_CELLS = 12
HOLE_THR = 0.24
HOLE_SOFT = 0.08
HOLE_STRENGTH = 1.0
BAND_PROB = 0.2
BAND_WIDTH = (1.25, 3.25)
BAND_LEN = (18.0, 70.0)
PILLAR_PROB = 0.05
# the pop-up vertical bar's width: 10%..50% of the 64 px frame (6.4..32 px),
# so it spans a tenth of the view up to half of it (2026-07-23, was 10..22 px).
PILLAR_WIDTH = (6.4, 32.0)
# Tasks that also persist the renderer's outward glow bleed (render_masks
# ``outer_grow``) append ONE extra key slot after the artifact families, so the
# glow width rotates on the same 1..hold clock as everything else.
GROW_FAMILY = NOISE_FAMILIES
N_FAMILIES_GROW = NOISE_FAMILIES + 1


# -- persistent-noise state (carried through an env's task state) -------------

def _split_raw(key: jax.Array, n: int) -> jax.Array:
    """Split ``key`` into ``n`` keys as RAW uint32 ``[n, 2]`` — accepts either a
    new-style typed key or an old-style uint32 key, so callers can carry the
    result in a plain-array pytree and ``jnp.where``-blend it on auto-reset."""
    ks = jax.random.split(key, n)
    return ks if ks.dtype == jnp.uint32 else jax.random.key_data(ks)


def fresh_noise_keys(key: jax.Array, f: int, *,
                     families: int = NOISE_FAMILIES) -> jax.Array:
    """Raw per-world family keys ``[f, families, 2]`` drawn fresh from ``key`` —
    for callers that carry no noise state (artifacts then redraw i.i.d. every
    frame, no persistence)."""
    return _split_raw(key, f * families).reshape(f, families, 2)


def noise_state_init(key: jax.Array, f: int, *, families: int = NOISE_FAMILIES,
                     hold: int = 16) -> tuple[jax.Array, jax.Array]:
    """Fresh persistent-noise state for ``f`` worlds: raw keys ``[f, families, 2]``
    (uint32) + remaining-lifetime ``ttl [f, families]`` (int32, uniform 1..hold,
    so lifetimes are staggered from the first frame)."""
    kk, kt = jax.random.split(key)
    keys = fresh_noise_keys(kk, f, families=families)
    ttl = jax.random.randint(kt, (f, families), 1, hold + 1, dtype=jnp.int32)
    return keys, ttl


def noise_state_step(key: jax.Array, keys: jax.Array, ttl: jax.Array, *,
                     hold: int = 16) -> tuple[jax.Array, jax.Array]:
    """Advance the persistence state one frame: (world, family) slots whose ttl
    ran out get a fresh key + a fresh lifetime ~ U{1..hold}; live slots keep
    their key (their artifact stays exactly where it was) and count down.
    ``hold=1`` degenerates to a resample every frame (per-frame i.i.d.)."""
    f, fam = ttl.shape
    kk, kt = jax.random.split(key)
    fresh = _split_raw(kk, f * fam).reshape(f, fam, 2)
    fresh_ttl = jax.random.randint(kt, (f, fam), 1, hold + 1, dtype=jnp.int32)
    expired = ttl <= 0
    keys = jnp.where(expired[..., None], fresh, keys)
    ttl = jnp.where(expired, fresh_ttl, ttl) - 1
    return keys, ttl


def grow_from_keys(noise_keys: jax.Array, mean: float, sd: float) -> jax.Array:
    """Per-world outer-glow grow width ``[F]`` (for ``render_masks(outer_grow=)``)
    drawn from the carried :data:`GROW_FAMILY` slot — a held key keeps the glow
    width steady across frames. ``noise_keys`` must carry
    :data:`N_FAMILIES_GROW` families."""
    gk = jax.random.wrap_key_data(noise_keys[:, GROW_FAMILY])
    return jnp.clip(mean + sd * jax.vmap(jax.random.normal)(gk), 0.0, None)


# -- per-world artifact draws (vmapped over the fleet by corrupt_mask_*) ------

def _blob_field1(key: jax.Array, h: int, w: int, cells: int) -> jax.Array:
    """Smooth low-frequency random field in ~[0, 1]: uniform noise on a coarse
    ``cells``×``cells`` grid, bilinearly upsampled to [h, w]. Thresholding
    it yields blobby regions of ~(h/cells) px scale."""
    u = jax.random.uniform(key, (cells, cells))
    return jax.image.resize(u, (h, w), method="linear")


def _stick_at(key: jax.Array, h: int, w: int,
              xs: jax.Array, ys: jax.Array, *, prob: float,
              width_lo: float, width_hi: float,
              len_lo: float, len_hi: float) -> jax.Array:
    """:func:`_stick1`'s coverage evaluated at ARBITRARY pixel coordinates.

    ``xs``/``ys`` broadcast against each other; passing the full pixel grid
    reproduces ``_stick1`` bit-for-bit (that is how ``_stick1`` is defined), and
    passing an ``[M]`` pair of corner coordinates samples the same artifact at M
    points for a hundredth of the cost. One definition, two call shapes — see
    :func:`erasure_at` for why the second one exists."""
    kon, kc, kw, kl, ka = jax.random.split(key, 5)
    on = jax.random.bernoulli(kon, prob)
    c = jax.random.uniform(kc, (2,))
    cxx, cyy = c[0] * w, c[1] * h
    half_w = 0.5 * jax.random.uniform(kw, (), minval=width_lo, maxval=width_hi)
    half_l = 0.5 * jax.random.uniform(kl, (), minval=len_lo, maxval=len_hi)
    a = jax.random.uniform(ka, (), minval=0.0, maxval=jnp.pi)
    ux, uy = jnp.cos(a), jnp.sin(a)
    dx, dy = xs - cxx, ys - cyy
    t = dx * ux + dy * uy                                            # along the band
    d = dy * ux - dx * uy                                            # across it
    cov = (jnp.clip(half_w - jnp.abs(d) + 1.0, 0.0, 1.0)
           * jnp.clip(half_l - jnp.abs(t) + 1.0, 0.0, 1.0))
    return on * cov


def _stick1(key: jax.Array, h: int, w: int, **kw) -> jax.Array:
    """Thin occluder segment coverage [h, w] in [0, 1] (1 = blocked): a
    Bernoulli(prob) gate, then a rectangle of random width and FINITE random
    length, centred anywhere in frame at ANY angle. Soft ~1 px edge; interior
    fully opaque."""
    return _stick_at(key, h, w,
                     jnp.arange(w, dtype=jnp.float32)[None, :],      # [1, w]
                     jnp.arange(h, dtype=jnp.float32)[:, None], **kw)   # [h, 1]


def _pillar_at(key: jax.Array, h: int, w: int,
               xs: jax.Array, ys: jax.Array, *, prob: float,
               width_lo: float, width_hi: float) -> jax.Array:
    """:func:`_pillar1`'s coverage at arbitrary pixel coordinates (see
    :func:`_stick_at` for the grid-vs-points contract)."""
    kon, kc, kw, ko = jax.random.split(key, 4)
    on = jax.random.bernoulli(kon, prob)
    vertical = jax.random.bernoulli(ko)                             # else horizontal
    half = 0.5 * jax.random.uniform(kw, (), minval=width_lo, maxval=width_hi)
    # centre along the perpendicular axis (x for a vertical bar, y for a
    # horizontal one); overshoot capped so an ON bar is never entirely
    # off-screen. h == w for the square policy view, but stay general.
    over = min(0.06 * max(h, w), 0.5 * width_lo)
    span = jnp.where(vertical, w, h).astype(jnp.float32)
    c = jax.random.uniform(kc, (), minval=-over, maxval=span + over)
    cov = jnp.where(vertical,
                    jnp.clip(half - jnp.abs(xs - c) + 1.0, 0.0, 1.0),   # [1, w]
                    jnp.clip(half - jnp.abs(ys - c) + 1.0, 0.0, 1.0))   # [h, 1]
    return on * cov                                                 # -> [h, w]


def _pillar1(key: jax.Array, h: int, w: int, **kw) -> jax.Array:
    """Thick dead-straight full-span occluder — a VERTICAL full-height column
    (a pillar up close) or, 50/50, a HORIZONTAL full-width band (a ceiling
    truss / beam): Bernoulli(prob) gate, random width and a random position
    along the perpendicular axis (may clip the frame edge). Soft ~1 px edge;
    interior fully opaque. Returns [h, w]."""
    return _pillar_at(key, h, w,
                      jnp.arange(w, dtype=jnp.float32)[None, :],     # [1, w]
                      jnp.arange(h, dtype=jnp.float32)[:, None], **kw)  # [h, 1]


def _blur3(m: jax.Array) -> jax.Array:
    """3×3 binomial blur ([1,2,1]⊗[1,2,1]/16) via shifted adds — cheap and
    conv-free. Edge pixels reuse their own value (edge padding)."""
    p = jnp.pad(m, ((0, 0), (1, 1), (1, 1)), mode="edge")
    r = (p[:, :-2, :] + 2.0 * p[:, 1:-1, :] + p[:, 2:, :]) * 0.25   # vertical
    return (r[:, :, :-2] + 2.0 * r[:, :, 1:-1] + r[:, :, 2:]) * 0.25


def corrupt_mask(
    noise_keys: jax.Array,
    mask: jax.Array,
    *,
    scale: float = 1.0,
    # band holes (branding decals / far-gate dropout): blobby full erasure.
    # thr shapes coverage, cells the chunk size (more cells = smaller chunks).
    hole_cells: int = HOLE_CELLS,
    hole_thr: float = HOLE_THR,
    hole_soft: float = HOLE_SOFT,
    hole_strength: float = HOLE_STRENGTH,
    # occluders (pillars / ceiling trusses), px at the mask resolution: one
    # thin band at any angle with finite length, rare thick straight full-span
    # bar (vertical column or horizontal beam, 50/50).
    band_prob: float = BAND_PROB,
    band_width: tuple[float, float] = BAND_WIDTH,
    band_len: tuple[float, float] = BAND_LEN,
    pillar_prob: float = PILLAR_PROB,
    pillar_width: tuple[float, float] = PILLAR_WIDTH,
    # glow false positives: rare large additive blobs, binary interior.
    blob_prob: float = 0.02,
    blob_cells: int = 5,
    blob_thr: float = 0.80,
    blob_soft: float = 0.03,
    # floor-glow speckle: sparse faint salt (sub-pixel dots, so never white).
    speckle_p: float = 0.0008,
    # occasional light blur: rounds small rings like the real JPEG+resample.
    # (0.10 -> 0.07, 2026-07-21: toned down a tad — the artifact churn already
    # softens plenty once holds stretch to ~1 s.)
    blur_prob: float = 0.07,
) -> jax.Array:
    """Corrupt a rendered gate mask [F, H, W] (soft, [0, 1]) like the real HSV
    pipeline, drawing each world's artifacts from its per-family keys
    ``noise_keys [F, ≥NOISE_FAMILIES, 2]`` (raw uint32). Keys CARRIED across
    steps (``noise_state_init``/``noise_state_step``) redraw the identical
    artifact each frame — noise stays put; keys drawn per frame
    (``fresh_noise_keys``) give i.i.d. flicker. ``scale`` is the master knob:
    0 = return the mask untouched (statically, when passed as a Python float);
    it scales every artifact's COVERAGE/PROBABILITY (hole coverage,
    band/blob/blur odds, speckle density) while values stay binary, so any
    scale still looks like a thresholded mask.
    """
    if isinstance(scale, (int, float)) and scale <= 0.0:
        return mask

    def p(x: float) -> float:
        return min(1.0, x * scale)

    def one(ks: jax.Array, m: jax.Array) -> tuple[jax.Array, jax.Array]:
        h, w = m.shape
        k_hole, k_band, k_pil, k_blob, k_salt, k_blur = (ks[i] for i in range(NOISE_FAMILIES))

        # 1. band holes — erase blobby chunks (only bites where the mask is lit)
        holes = jnp.clip(
            (hole_thr * scale - _blob_field1(k_hole, h, w, hole_cells)) / hole_soft,
            0.0, 1.0)
        m = m * (1.0 - hole_strength * holes)

        # 2. occluders — black shapes in front of everything so far
        occ = _stick1(k_band, h, w, prob=p(band_prob),
                      width_lo=band_width[0], width_hi=band_width[1],
                      len_lo=band_len[0], len_hi=band_len[1])
        occ = jnp.maximum(occ, _pillar1(k_pil, h, w, prob=p(pillar_prob),
                                        width_lo=pillar_width[0], width_hi=pillar_width[1]))
        m = m * (1.0 - occ)

        # 3. glow blob — Bernoulli gate, high-threshold cut of a coarse field:
        # zero-to-a-few large patches, interior 1.0, ~thin AA edge
        k_blob_f, k_blob_on = jax.random.split(k_blob)
        on = jax.random.bernoulli(k_blob_on, p(blob_prob))
        blob = jnp.clip((_blob_field1(k_blob_f, h, w, blob_cells) - blob_thr) / blob_soft,
                        0.0, 1.0)
        m = jnp.maximum(m, on * blob)

        # 4. speckle — sparse faint single-pixel salt (sub-pixel dots downsampled)
        k_salt_p, k_salt_v = jax.random.split(k_salt)
        salt = (jax.random.bernoulli(k_salt_p, p(speckle_p), (h, w))
                * jax.random.uniform(k_salt_v, (h, w), minval=0.15, maxval=0.5))
        m = jnp.maximum(m, salt)

        # 5. occasional light blur — gated here, applied batched below
        return m, jax.random.bernoulli(k_blur, p(blur_prob))

    keys = jax.random.wrap_key_data(noise_keys[:, :NOISE_FAMILIES, :])   # [F, 6] typed
    out, blur_on = jax.vmap(one)(keys, mask)
    return jnp.where(blur_on[:, None, None], _blur3(out), out)


def _blob_field_at(key: jax.Array, h: int, w: int, cells: int,
                   xs: jax.Array, ys: jax.Array) -> jax.Array:
    """:func:`_blob_field1` sampled at arbitrary pixel coordinates.

    Reproduces ``jax.image.resize(..., "linear")``'s half-pixel-centre mapping —
    output pixel *i* reads input coordinate ``(i + 0.5) * cells / n - 0.5``,
    clamped at the edges — so a point sample equals the full render's value at
    that pixel. Unlike the two occluders this is a re-implementation rather than
    a shared definition (the grid path goes through ``jax.image.resize``, which
    takes no coordinates), so ``test_mask_noise_point_sampler_matches_render``
    pins the two together."""
    u = jax.random.uniform(key, (cells, cells))
    gx = jnp.clip((xs + 0.5) * (cells / w) - 0.5, 0.0, cells - 1.0)
    gy = jnp.clip((ys + 0.5) * (cells / h) - 0.5, 0.0, cells - 1.0)
    x0, y0 = jnp.floor(gx).astype(jnp.int32), jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, cells - 1)
    y1 = jnp.minimum(y0 + 1, cells - 1)
    fx, fy = gx - x0, gy - y0
    top = u[y0, x0] * (1.0 - fx) + u[y0, x1] * fx
    bot = u[y1, x0] * (1.0 - fx) + u[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def erasure_at(
    noise_keys: jax.Array,
    pts_px: jax.Array,
    h: int,
    w: int,
    *,
    scale: float = 1.0,
    hole_cells: int = HOLE_CELLS,
    hole_thr: float = HOLE_THR,
    hole_soft: float = HOLE_SOFT,
    hole_strength: float = HOLE_STRENGTH,
    band_prob: float = BAND_PROB,
    band_width: tuple[float, float] = BAND_WIDTH,
    band_len: tuple[float, float] = BAND_LEN,
    pillar_prob: float = PILLAR_PROB,
    pillar_width: tuple[float, float] = PILLAR_WIDTH,
) -> jax.Array:
    """``[F, M]`` in [0, 1] — how much of the mask this world's ERASING artifacts
    remove at each of M query pixels. 0 = untouched, 1 = fully erased.

    The inverse question to :func:`corrupt_mask`: instead of "what does the mask
    look like", "is *this* pixel under a hole or an occluder". It exists so a
    consumer that never sees the rendered mask — the training filter's synthetic
    corner detector — can lose exactly the corners the policy's mask lost. A
    pillar that hides gate 5 from the CNN must hide it from the filter too, or
    the filter is reading through an occluder and the policy learns to trust an
    information channel with no deploy analogue.

    ``pts_px`` is ``[F, M, 2]`` as **(x, y) = (column, row)** in the SAME pixel
    frame as ``mask`` — a caller working at a different resolution must rescale
    first. ``noise_keys`` is the carried ``[F, >=NOISE_FAMILIES, 2]`` state, so
    the answer is a pure function of the same keys that painted the frame.

    Only the two ERASING families are summed — band holes and occluders
    (thin band + pillar) — composed exactly as ``corrupt_mask`` composes them
    (multiplicative survival, so ``1 - (1-hole)(1-occ)``). The ADDING families
    (glow blob, speckle) are excluded on purpose: they are false POSITIVES, and
    a false positive does not hide a real corner — modelling it here would
    delete detections the real detector would still make. The blur is excluded
    too: it softens edges by ~1 px, it does not erase.

    Cost is O(M) analytic evaluations against O(H*W) for a render, so calling
    this per gate corner every frame is free next to the render it mirrors.
    """
    if isinstance(scale, (int, float)) and scale <= 0.0:
        return jnp.zeros(pts_px.shape[:-1], pts_px.dtype)

    def p(x: float) -> float:
        return min(1.0, x * scale)

    def one(ks: jax.Array, pts: jax.Array) -> jax.Array:
        xs, ys = pts[..., 0], pts[..., 1]
        k_hole, k_band, k_pil = ks[0], ks[1], ks[2]
        holes = jnp.clip(
            (hole_thr * scale - _blob_field_at(k_hole, h, w, hole_cells, xs, ys)) / hole_soft,
            0.0, 1.0) * hole_strength
        occ = _stick_at(k_band, h, w, xs, ys, prob=p(band_prob),
                        width_lo=band_width[0], width_hi=band_width[1],
                        len_lo=band_len[0], len_hi=band_len[1])
        occ = jnp.maximum(occ, _pillar_at(k_pil, h, w, xs, ys, prob=p(pillar_prob),
                                          width_lo=pillar_width[0],
                                          width_hi=pillar_width[1]))
        return 1.0 - (1.0 - holes) * (1.0 - occ)

    keys = jax.random.wrap_key_data(noise_keys[:, :NOISE_FAMILIES, :])   # [F, 6] typed
    return jax.vmap(one)(keys, pts_px)
