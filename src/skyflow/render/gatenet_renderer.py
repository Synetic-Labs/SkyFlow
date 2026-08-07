"""Analytic gate segmentation-mask renderer (pure JAX, fleet-batched).

A vision-based gate policy can't carry a real renderer inside the all-JAX
training loop (the firmware×crazyflow rollout is one jitted scan over the whole
fleet — a host-side MuJoCo render per step would be the throughput wall). But
the only thing the policy needs to *see* is a binary segmentation mask of the
gate frames, exactly what GateNet produces from a real camera frame at
deployment. Flat gates are trivial geometry, so we render that mask
analytically.

Method — per-pixel ray-cast against each gate's SOLID frame (not forward
polygon projection): back-project each pixel into a world ray and intersect it
with the gate solid — the outer box (outer rectangle × ``depths[g]`` thick
along the normal) minus the inner-opening prism — via slab intervals in the
gate's own axis frame. A pixel is marked when the ray's inside-the-outer-box
interval has any part outside the opening interval and in front of the camera.
This renders the true silhouette: at oblique views the bars' side faces widen
the band exactly as a physical gate does; ``depths[g] = 0`` degenerates to the
flat-plane band. Ray-casting handles partial visibility and gate-behind-camera
for free. Frames are the only geometry and they all belong to the same mask
class, so the union over gates needs no depth sorting: a frame seen through
another gate's opening, or occluded by a nearer frame, marks the pixel either
way. The mask is exact up to rasterization.

Gates are arbitrary in number and placement (``GateSet``): per-gate centre,
heading, pitch and frame dimensions, batched as arrays so one render serves
any course layout. Pitch tilts a gate forward/backward about its lateral axis
(up-positive — the through-axis normal gains an upward component), giving each
gate its own in-plane vertical axis; upright gates are the pitch = 0 case.

Frames (shared with the crazyflow reference env): world NED (x north, y east, z down),
body FRD (x forward, y right, z down), quaternion wxyz. The camera looks forward
along body +x, tilted down by ``mount_pitch_deg``; its image frame is the
standard pinhole convention (x right, y down, z forward / optical axis).

This module is deliberately self-contained (jax + jax.numpy only) so it can be
unit-tested and validated against a MuJoCo segmentation render in isolation
before it is wired into the gate task.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import jax
import jax.numpy as jnp


def _rotate(quat_wxyz: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate body vector ``v`` into the world frame (R(q) v), batched over the
    leading axis. The active counterpart of crazyflow's `_rotate_inv` (R(q)^T v)."""
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return jnp.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], axis=-1)


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera mounted on the drone body (FRD).

    The optical axis is body +x rotated down by ``mount_pitch_deg`` about the
    body right axis (+y); the image frame is x-right / y-down / z-forward.
    ``fov_x_deg`` / ``fov_y_deg`` are the horizontal / vertical fields of view in
    degrees; they may differ (the real BetaFPV C03, undistorted then squashed to a
    square frame, has fx≠fy — ~99° H, ~80° V), so the focal lengths are tracked
    separately. ``offset_body`` is the camera position relative to the body origin
    in FRD metres (small; it mostly matters at very close range).
    """

    height: int = 32
    width: int = 32
    fov_x_deg: float = 99.0         # horizontal FOV (BetaFPV C03, undistorted)
    fov_y_deg: float = 79.8         # vertical FOV
    mount_pitch_deg: float = 40.0   # downward tilt of the optical axis
    offset_body: tuple[float, float, float] = (0.02, 0.0, -0.05)  # FRD: 2cm fwd, 5cm up
    # Anti-aliasing: cast supersample² rays per output pixel and average their
    # hits into a soft coverage in [0, 1]. >1 stops a distant gate's thin frame
    # band from aliasing out of a low-res mask, and yields soft edges that train
    # better and match GateNet's probabilistic output. Cost scales as ss².
    supersample: int = 2

    @cached_property
    def focal(self) -> tuple[float, float]:
        """Focal lengths (fx, fy) in pixels: f = (size/2) / tan(fov/2)."""
        fx = (self.width / 2.0) / math.tan(math.radians(self.fov_x_deg) / 2.0)
        fy = (self.height / 2.0) / math.tan(math.radians(self.fov_y_deg) / 2.0)
        return (fx, fy)

    @cached_property
    def principal_point(self) -> tuple[float, float]:
        """Image centre (cx, cy) in pixels."""
        return (self.width / 2.0, self.height / 2.0)

    @property
    def R_body_from_cam(self) -> jax.Array:
        """3x3 rotation mapping a camera-frame vector into the body (FRD) frame.

        Camera axes expressed in body FRD, with a downward mount pitch θ:
          right   (cam +x) = body +y                  = [0, 1, 0]
          down    (cam +y) = forward × right          = [-sinθ, 0, cosθ]
          forward (cam +z) = body +x tilted down by θ = [cosθ, 0, sinθ]
        The columns of R_body_from_cam are these body-frame axis vectors.
        """
        t = math.radians(self.mount_pitch_deg)
        c, s = math.cos(t), math.sin(t)
        right = (0.0, 1.0, 0.0)
        down = (-s, 0.0, c)
        forward = (c, 0.0, s)
        # columns = camera axes in body coords
        return jnp.array([
            [right[0], down[0], forward[0]],
            [right[1], down[1], forward[1]],
            [right[2], down[2], forward[2]],
        ], dtype=jnp.float32)

    @property
    def ray_dirs_cam(self) -> jax.Array:
        """Ray directions in the camera frame at the supersampled grid, shape
        [H·ss, W·ss, 3] (un-normalised; z = 1). Sub-pixel j maps to base-pixel
        coordinate (j + 0.5)/ss and through the pinhole as [(u-cx)/f, (v-cy)/f, 1];
        cx/cy/f are in base-pixel units. Sampling at sub-pixel centres keeps the
        exact left/right + up/down symmetry about (cx, cy)."""
        cx, cy = self.principal_point
        fx, fy = self.focal
        ss = self.supersample
        us = ((jnp.arange(self.width * ss, dtype=jnp.float32) + 0.5) / ss - cx) / fx
        vs = ((jnp.arange(self.height * ss, dtype=jnp.float32) + 0.5) / ss - cy) / fy
        uu, vv = jnp.meshgrid(us, vs, indexing="xy")     # [H·ss, W·ss]
        return jnp.stack([uu, vv, jnp.ones_like(uu)], axis=-1)


@dataclass(frozen=True)
class GateSet:
    """G flat rectangular gates in the world (NED), batched as arrays.

    Every gate lies in a plane facing along its normal: heading ``yaws[g]``
    about the world down-axis, then pitched ``pitches[g]`` about the gate's
    lateral axis (up-positive — a positive pitch tilts the through axis
    upward). A gate's frame band is the region inside its outer rectangle but
    outside its inner opening. Half-extents are metres from centre:
    ``inner_half[g]`` = (lateral, vertical) of the opening, ``outer_half[g]``
    likewise of the outer edge.

    Construct with :meth:`build` (shared frame dimensions — the common case of
    one physical gate design placed G times) or fill the arrays directly for
    per-gate sizes. All fields are [G, ...] jnp arrays; the set is a fixed
    world-frame constant that rides inside jitted programs as baked-in values
    (like the single gate centre it generalizes).
    """

    centers: jax.Array                      # [G, 3] NED gate centres
    yaws: jax.Array                         # [G] heading of each gate normal (rad)
    inner_half: jax.Array                   # [G, 2] (lateral, vertical) opening half-extents
    outer_half: jax.Array                   # [G, 2] outer-edge half-extents
    pitches: jax.Array = field(default=None)     # type: ignore[assignment]  # [G] rad, up-positive
    depths: jax.Array = field(default=None)      # type: ignore[assignment]  # [G] m along the normal (0 = flat)
    # world-frame plane axes, derived from yaws+pitches in __post_init__; [G, 3] each.
    # (laterals stay horizontal — pitch rotates about them; verticals are in-plane.)
    normals: jax.Array = field(default=None)     # type: ignore[assignment]
    laterals: jax.Array = field(default=None)    # type: ignore[assignment]
    verticals: jax.Array = field(default=None)   # type: ignore[assignment]

    # Defaults are the real orange gate: ~0.55 m square opening, ~0.225 m orange
    # frame band per side (outer ~1.0 m square). The triangular/mitred corners of
    # the physical gate are approximated as a square-cornered band — a small
    # difference at the four corners; refine if sim-to-real keys on it.
    DEFAULT_INNER_HALF = (0.275, 0.275)
    DEFAULT_FRAME_WIDTH = 0.225

    def __post_init__(self) -> None:
        if self.pitches is None:
            object.__setattr__(self, "pitches", jnp.zeros_like(self.yaws))
        if self.depths is None:
            object.__setattr__(self, "depths", jnp.zeros_like(self.yaws))
        if self.normals is None or self.laterals is None or self.verticals is None:
            c, s = jnp.cos(self.yaws), jnp.sin(self.yaws)
            ct, st = jnp.cos(self.pitches), jnp.sin(self.pitches)
            zeros = jnp.zeros_like(c)
            # yaw about world down, then pitch about the gate's lateral axis
            # (up-positive): normal gains an upward (-z NED) component; the
            # in-plane vertical = lateral x normal tilts with it.
            object.__setattr__(self, "normals",
                               jnp.stack([ct * c, ct * s, -st], axis=-1).astype(jnp.float32))
            object.__setattr__(self, "laterals",
                               jnp.stack([-s, c, zeros], axis=-1).astype(jnp.float32))
            object.__setattr__(self, "verticals",
                               jnp.stack([-c * st, -s * st, -ct], axis=-1).astype(jnp.float32))

    @classmethod
    def build(
        cls,
        centers_ned: Sequence[Sequence[float]] | jax.Array,
        yaws: Sequence[float] | jax.Array,
        inner_half: tuple[float, float] = DEFAULT_INNER_HALF,
        frame_width: float = DEFAULT_FRAME_WIDTH,
        pitches: Sequence[float] | jax.Array | None = None,
        depth: float = 0.0,
    ) -> GateSet:
        """G gates of one physical design at arbitrary centres/headings.

        ``centers_ned`` [G, 3] and ``yaws`` [G] place each gate, ``pitches``
        [G] (optional, rad, up-positive) tilts them; ``inner_half`` (lateral,
        vertical), ``frame_width`` (band width per side, so outer = inner +
        frame_width) and ``depth`` (frame thickness along the normal, m;
        0 = flat plane) are shared by all gates.
        """
        centers = jnp.asarray(centers_ned, jnp.float32).reshape(-1, 3)
        yaws_a = jnp.asarray(yaws, jnp.float32).reshape(-1)
        if centers.shape[0] != yaws_a.shape[0]:
            raise ValueError(
                f"{centers.shape[0]} centers but {yaws_a.shape[0]} yaws")
        g = centers.shape[0]
        pitches_a = None
        if pitches is not None:
            pitches_a = jnp.asarray(pitches, jnp.float32).reshape(-1)
            if pitches_a.shape[0] != g:
                raise ValueError(f"{g} centers but {pitches_a.shape[0]} pitches")
        iw, ih = float(inner_half[0]), float(inner_half[1])
        fw = float(frame_width)
        inner = jnp.broadcast_to(jnp.array([iw, ih], jnp.float32), (g, 2))
        outer = jnp.broadcast_to(jnp.array([iw + fw, ih + fw], jnp.float32), (g, 2))
        depths = jnp.full((g,), float(depth), jnp.float32)
        return cls(centers=centers, yaws=yaws_a, inner_half=inner,
                   outer_half=outer, pitches=pitches_a, depths=depths)

    @classmethod
    def single(
        cls,
        center_ned: tuple[float, float, float] = (3.0, 0.0, -1.5),
        yaw: float = 0.0,
        inner_half: tuple[float, float] = DEFAULT_INNER_HALF,
        outer_half: tuple[float, float] | None = None,
        pitch: float = 0.0,
        depth: float = 0.0,
    ) -> GateSet:
        """One gate — the original single-gate task geometry."""
        ow, oh = (outer_half if outer_half is not None
                  else (inner_half[0] + cls.DEFAULT_FRAME_WIDTH,
                        inner_half[1] + cls.DEFAULT_FRAME_WIDTH))
        return cls(
            centers=jnp.asarray([center_ned], jnp.float32),
            yaws=jnp.asarray([yaw], jnp.float32),
            inner_half=jnp.asarray([inner_half], jnp.float32),
            outer_half=jnp.asarray([[ow, oh]], jnp.float32),
            pitches=jnp.asarray([pitch], jnp.float32),
            depths=jnp.asarray([depth], jnp.float32))

    def __len__(self) -> int:
        return int(self.centers.shape[0])


# World up (NED down negated). NOT a gate axis — gates carry their own
# per-pitch in-plane verticals (gates.verticals); this is for world-frame
# consumers like the gate task's spawn altitude jitter.
VERTICAL = jnp.array([0.0, 0.0, -1.0], jnp.float32)

_BIG = 1e30


def _slab(oo: jax.Array, dd: jax.Array, h) -> tuple[jax.Array, jax.Array]:
    """Parameter interval (lo, hi) where |oo + t·dd| <= h — one axis of a
    box/prism intersection. Empty when lo > hi. A direction component ~0 gives
    the full line when the origin is inside the slab, else the empty interval.
    Broadcasts (oo, dd, h alike); h may be 0 (degenerate flat slab)."""
    par = jnp.abs(dd) < 1e-9
    dd_s = jnp.where(par, 1e-9, dd)
    t1 = (-h - oo) / dd_s
    t2 = (h - oo) / dd_s
    inside = jnp.abs(oo) <= h
    lo = jnp.where(par, jnp.where(inside, -_BIG, _BIG), jnp.minimum(t1, t2))
    hi = jnp.where(par, jnp.where(inside, _BIG, -_BIG), jnp.maximum(t1, t2))
    return lo, hi


def _frame_solid_interval(
    oo_n, dd_n, oo_l, dd_l, oo_v, dd_v, half_depth, ow, oh, iw, ih,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Slab intervals for the gate SOLID in gate-frame coordinates: the outer
    box (|n| <= half_depth, |lat| <= ow, |vert| <= oh) as [a0, a1], and the
    inner-opening prism (|lat| < iw, |vert| < ih, unbounded along the normal)
    as [b0, b1]. The solid is A \\ B — the four frame bars exactly (their
    union is the outer box minus the opening)."""
    n0, n1 = _slab(oo_n, dd_n, half_depth)
    l0, l1 = _slab(oo_l, dd_l, ow)
    v0, v1 = _slab(oo_v, dd_v, oh)
    a0 = jnp.maximum(n0, jnp.maximum(l0, v0))
    a1 = jnp.minimum(n1, jnp.minimum(l1, v1))
    i0, i1 = _slab(oo_l, dd_l, iw)
    j0, j1 = _slab(oo_v, dd_v, ih)
    b0 = jnp.maximum(i0, j0)
    b1 = jnp.minimum(i1, j1)
    return a0, a1, b0, b1


def _hits_solid(a0, a1, b0, b1, t_lo, t_hi) -> jax.Array:
    """True where [a0, a1] \\ [b0, b1] intersects [t_lo, t_hi]: some parameter
    range lies inside the outer box but outside the opening — before entering
    the hole, or after exiting it."""
    lo = jnp.maximum(a0, t_lo)
    before = lo <= jnp.minimum(jnp.minimum(a1, b0), t_hi)
    after = jnp.maximum(lo, b1) <= jnp.minimum(a1, t_hi)
    return before | after


def classify_crossings(
    prev_pos: jax.Array, pos: jax.Array, gates: GateSet,
    body_radius: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Classify the prev_pos→pos segment against every gate's SOLID frame.

    ``hit_frame``: the segment intersects the frame solid anywhere — the front
    face, a bar's side wall while transiting the opening obliquely, or the top
    of the frame during a descent that never crosses the gate plane. A clean
    ``pass`` crosses the gate's centre plane with the crossing point inside
    the opening AND never touches the solid; passes split by direction
    (forward = from the −normal side to the +normal side, the gate's facing
    direction). ``depths[g] = 0`` reduces to the classic flat-plane test.

    ``body_radius`` models the drone as a sphere instead of a point: the gate
    solid is inflated by r (Minkowski sum, square-cornered box approximation —
    outer + r, opening − r, depth + 2r) so a pass needs r of clearance from
    every bar and grazing within r of one is a hit. The RENDERED mask is
    untouched — the camera sees the physical gate; only contact inflates.

    Returns (pass_fwd, pass_bwd, hit_frame), each [F, G] bool. Pure geometry,
    shared by the gate task's pass/collision events; unit-testable without the
    firmware.
    """
    r = float(body_radius)
    d_prev = prev_pos[:, None, :] - gates.centers                      # [F, G, 3]
    seg = (pos - prev_pos)[:, None, :]                                 # [F, 1, 3]
    oo_n = jnp.sum(d_prev * gates.normals, axis=-1)                    # [F, G]
    dd_n = jnp.sum(seg * gates.normals, axis=-1)
    oo_l = jnp.sum(d_prev * gates.laterals, axis=-1)
    dd_l = jnp.sum(seg * gates.laterals, axis=-1)
    oo_v = jnp.sum(d_prev * gates.verticals, axis=-1)
    dd_v = jnp.sum(seg * gates.verticals, axis=-1)

    a0, a1, b0, b1 = _frame_solid_interval(
        oo_n, dd_n, oo_l, dd_l, oo_v, dd_v,
        0.5 * gates.depths + r,
        gates.outer_half[:, 0] + r, gates.outer_half[:, 1] + r,
        gates.inner_half[:, 0] - r, gates.inner_half[:, 1] - r)
    hit = _hits_solid(a0, a1, b0, b1, 0.0, 1.0)

    # clean pass: centre-plane sign change, crossing point inside the opening,
    # and the segment never touched the solid
    prev_sd, sd = oo_n, oo_n + dd_n
    crossed = (prev_sd * sd) < 0.0
    denom = prev_sd - sd
    alpha = prev_sd / jnp.where(jnp.abs(denom) < 1e-9, 1e-9, denom)
    cp_lat = jnp.abs(oo_l + alpha * dd_l)
    cp_vert = jnp.abs(oo_v + alpha * dd_v)
    in_inner = ((cp_lat < gates.inner_half[:, 0] - r)
                & (cp_vert < gates.inner_half[:, 1] - r))
    passed = crossed & in_inner & (~hit)
    forward = prev_sd < 0.0
    return passed & forward, passed & (~forward), hit


def render_masks(
    cam: CameraModel,
    gates: GateSet,
    pos_ned: jax.Array,
    quat_ned: jax.Array,
    *,
    R_body_from_cam: jax.Array | None = None,
    outer_grow: jax.Array | None = None,
) -> jax.Array:
    """Render the gate segmentation mask (union over all gates) for a fleet.

    Args:
      cam: camera model (intrinsics + nominal extrinsics).
      gates: gate geometry in the world (NED), any number of gates.
      pos_ned: drone positions, [F, 3] (NED).
      quat_ned: drone attitudes, [F, 4] (wxyz).
      R_body_from_cam: optional per-agent override of the body←camera rotation,
        [F, 3, 3] — lets the task inject per-episode extrinsic jitter (domain
        randomization). Defaults to the camera's nominal rotation for all agents.
      outer_grow: optional per-agent OUTWARD widening of every gate's outer
        edge, [F] metres — models the glow bleed of the real HSV mask, whose
        outer boundary sits wider than the physical frame while the inner
        boundary stays true (measured on real masks; see mask_noise.py).
        Only the render widens: contact geometry is untouched.

    Returns:
      mask: [F, H, W] float32 in [0, 1] — the fraction of a pixel's supersample²
      rays that hit any gate's frame band in front of the camera (binary when
      ``cam.supersample == 1``).
    """
    fleet = pos_ned.shape[0]
    H, W, ss = cam.height, cam.width, cam.supersample
    offset_body = jnp.asarray(cam.offset_body, jnp.float32)
    R_bc = cam.R_body_from_cam if R_body_from_cam is None else R_body_from_cam

    # camera origin in the world: body origin + R(q) · offset_body
    cam_origin = pos_ned + _rotate(quat_ned, jnp.broadcast_to(offset_body, (fleet, 3)))

    # per-pixel ray directions: cam frame → body frame → world frame.
    # ray_dirs_cam [H,W,3]; rotate into body with R_body_from_cam (per agent),
    # then into world with R(quat). Flatten the pixel grid for the matmuls.
    rays_cam = cam.ray_dirs_cam.reshape(-1, 3)                       # [P, 3]
    if R_body_from_cam is None:
        rays_body = rays_cam @ R_bc.T                               # [P, 3]
        rays_body = jnp.broadcast_to(rays_body, (fleet, *rays_body.shape))
    else:
        # per-agent rotation: [F,3,3] · [P,3] -> [F,P,3]
        rays_body = jnp.einsum("fij,pj->fpi", R_bc, rays_cam)
    rays_world = _rotate(quat_ned[:, None, :], rays_body)            # [F, P, 3]

    # Per-gate ray–solid intersection, unioned at subray resolution. A Python
    # loop over the (static, small) gate count keeps peak memory at F×P per
    # gate instead of materializing an F×P×G block; the frame bars share one
    # mask class, so a boolean OR is the exact composite (no depth sorting).
    grow = None if outer_grow is None else outer_grow[:, None]       # [F, 1]
    band_any = jnp.zeros(rays_world.shape[:2], bool)                 # [F, P]
    for g in range(len(gates)):
        oc = cam_origin - gates.centers[g]                           # [F, 3]
        # gate-frame coordinates: origin [F, 1] broadcast against dirs [F, P]
        oo_n = jnp.sum(oc * gates.normals[g], axis=-1)[:, None]
        oo_l = jnp.sum(oc * gates.laterals[g], axis=-1)[:, None]
        oo_v = jnp.sum(oc * gates.verticals[g], axis=-1)[:, None]
        dd_n = jnp.sum(rays_world * gates.normals[g], axis=-1)       # [F, P]
        dd_l = jnp.sum(rays_world * gates.laterals[g], axis=-1)
        dd_v = jnp.sum(rays_world * gates.verticals[g], axis=-1)
        ow, oh = gates.outer_half[g, 0], gates.outer_half[g, 1]
        if grow is not None:
            ow, oh = ow + grow, oh + grow                            # [F, 1]
        a0, a1, b0, b1 = _frame_solid_interval(
            oo_n, dd_n, oo_l, dd_l, oo_v, dd_v,
            0.5 * gates.depths[g], ow, oh,
            gates.inner_half[g, 0], gates.inner_half[g, 1])
        band_any = band_any | _hits_solid(a0, a1, b0, b1, 1e-6, _BIG)

    # [F, H·ss, W·ss] -> average each ss×ss block into a soft [F, H, W] coverage
    band = band_any.astype(jnp.float32).reshape(fleet, H, ss, W, ss)
    return band.mean(axis=(2, 4))


def render_floor(
    cam: CameraModel,
    pos_ned: jax.Array,
    quat_ned: jax.Array,
    *,
    R_body_from_cam: jax.Array | None = None,
    half_extent: float | None = None,
) -> jax.Array:
    """Render the ground-plane (NED z=0) coverage for a fleet — the WHITE FLOOR.

    A pixel is covered when its camera ray crosses the z=0 plane in FRONT of the
    camera (the drone flies above ground, NED z<0, so a downward ray t>0 hits it);
    everything above the horizon misses and stays sky. Same ray setup as
    :func:`render_masks`. ``half_extent`` optionally clips the floor to a
    ±half_extent metre square about the origin; ``None`` = the whole ground plane.

    Returns coverage ``[F, H, W]`` float32 in [0, 1] (soft via ``cam.supersample``).
    """
    fleet = pos_ned.shape[0]
    H, W, ss = cam.height, cam.width, cam.supersample
    offset_body = jnp.asarray(cam.offset_body, jnp.float32)
    R_bc = cam.R_body_from_cam if R_body_from_cam is None else R_body_from_cam

    cam_origin = pos_ned + _rotate(quat_ned, jnp.broadcast_to(offset_body, (fleet, 3)))
    rays_cam = cam.ray_dirs_cam.reshape(-1, 3)                       # [P, 3]
    if R_body_from_cam is None:
        rays_body = rays_cam @ R_bc.T
        rays_body = jnp.broadcast_to(rays_body, (fleet, *rays_body.shape))
    else:
        rays_body = jnp.einsum("fij,pj->fpi", R_bc, rays_cam)
    rays_world = _rotate(quat_ned[:, None, :], rays_body)            # [F, P, 3]

    oz = cam_origin[:, None, 2]                                     # [F, 1] NED z (down +)
    dz = rays_world[..., 2]                                         # [F, P]
    t = (0.0 - oz) / jnp.where(jnp.abs(dz) < 1e-9, 1e-9, dz)
    hit = (t > 1e-6) & (jnp.abs(dz) > 1e-9)                         # ground in front of the camera
    if half_extent is not None:
        nx = cam_origin[:, None, 0] + t * rays_world[..., 0]
        ny = cam_origin[:, None, 1] + t * rays_world[..., 1]
        hit = hit & (jnp.abs(nx) <= half_extent) & (jnp.abs(ny) <= half_extent)
    return hit.astype(jnp.float32).reshape(fleet, H, ss, W, ss).mean(axis=(2, 4))


def render_masks_perworld(
    cam: CameraModel,
    centers: jax.Array, normals: jax.Array, laterals: jax.Array, verticals: jax.Array,
    inner_half: jax.Array, outer_half: jax.Array, depths: jax.Array,
    pos_ned: jax.Array, quat_ned: jax.Array,
    *,
    R_body_from_cam: jax.Array | None = None,
    outer_grow: jax.Array | None = None,
) -> jax.Array:
    """:func:`render_masks` with PER-WORLD (per-episode) gate geometry.

    Same ray-cast against each gate's solid frame and the same OR-union over gates,
    but every gate array carries a leading FLEET axis so each world can see gates in
    a different place (:func:`render_masks` bakes a single fixed ``GateSet`` shared by
    the whole fleet). Used by tasks that RANDOMIZE gate placement per episode.

    Shapes: ``centers``/``normals``/``laterals``/``verticals`` are [F, G, 3];
    ``inner_half``/``outer_half`` [F, G, 2]; ``depths`` [F, G]. ``pos_ned`` [F, 3],
    ``quat_ned`` [F, 4]. ``R_body_from_cam`` [F, 3, 3] and ``outer_grow`` [F] match
    :func:`render_masks`. Returns the mask [F, H, W] in [0, 1]. The plane axes must be
    consistent with the yaw/pitch used to build them (same convention as
    ``GateSet.__post_init__``)."""
    fleet = pos_ned.shape[0]
    H, W, ss = cam.height, cam.width, cam.supersample
    offset_body = jnp.asarray(cam.offset_body, jnp.float32)
    R_bc = cam.R_body_from_cam if R_body_from_cam is None else R_body_from_cam

    cam_origin = pos_ned + _rotate(quat_ned, jnp.broadcast_to(offset_body, (fleet, 3)))

    rays_cam = cam.ray_dirs_cam.reshape(-1, 3)                       # [P, 3]
    if R_body_from_cam is None:
        rays_body = rays_cam @ R_bc.T
        rays_body = jnp.broadcast_to(rays_body, (fleet, *rays_body.shape))
    else:
        rays_body = jnp.einsum("fij,pj->fpi", R_bc, rays_cam)       # per-agent rotation
    rays_world = _rotate(quat_ned[:, None, :], rays_body)            # [F, P, 3]

    grow = None if outer_grow is None else outer_grow[:, None]       # [F, 1]
    G = centers.shape[1]
    band_any = jnp.zeros(rays_world.shape[:2], bool)                 # [F, P]
    for g in range(G):
        c = centers[:, g]                                           # [F, 3]
        nrm, lat, ver = normals[:, g], laterals[:, g], verticals[:, g]  # [F, 3] each
        oc = cam_origin - c                                         # [F, 3]
        oo_n = jnp.sum(oc * nrm, axis=-1)[:, None]                  # [F, 1]
        oo_l = jnp.sum(oc * lat, axis=-1)[:, None]
        oo_v = jnp.sum(oc * ver, axis=-1)[:, None]
        dd_n = jnp.sum(rays_world * nrm[:, None, :], axis=-1)       # [F, P]
        dd_l = jnp.sum(rays_world * lat[:, None, :], axis=-1)
        dd_v = jnp.sum(rays_world * ver[:, None, :], axis=-1)
        ow, oh = outer_half[:, g, 0][:, None], outer_half[:, g, 1][:, None]   # [F, 1]
        if grow is not None:
            ow, oh = ow + grow, oh + grow
        a0, a1, b0, b1 = _frame_solid_interval(
            oo_n, dd_n, oo_l, dd_l, oo_v, dd_v,
            0.5 * depths[:, g][:, None], ow, oh,
            inner_half[:, g, 0][:, None], inner_half[:, g, 1][:, None])
        band_any = band_any | _hits_solid(a0, a1, b0, b1, 1e-6, _BIG)

    band = band_any.astype(jnp.float32).reshape(fleet, H, ss, W, ss)
    return band.mean(axis=(2, 4))
