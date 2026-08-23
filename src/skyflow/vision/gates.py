"""
Gate geometry and course builders (DESIGN.md §2, §9).

A :class:`GateSet` is G rectangular racing gates batched as arrays: per-gate centre,
heading, pitch and frame dimensions, a fixed world-frame constant that rides inside jitted
programs as baked-in values. The frame band of a gate is the region inside its outer
rectangle but outside its inner opening; ``depths[g]`` gives the band a real thickness
along the normal (0 = flat plane).

Frame contract (DESIGN.md §3): every public surface speaks world z-up FLU — builders take
z-up centres (z = altitude), yaw about world +z (0 = +x, 90° = +y), pitch up-positive;
:func:`classify_crossings` takes z-up positions; the ``*_world`` properties read the
geometry back in z-up. The DATACLASS FIELDS are the renderer's internal NED arrays
(§3a) — construct through the builders, not the raw constructor, unless you are inside
vision/. The z-up→NED conversion happens exactly once, in :func:`_gateset_from_world`.

Geometry and crossing classification are validated against MuJoCo segmentation renders;
:func:`figure_eight` is specific to SkyFlow (DESIGN.md §9).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from skyflow.vision._ned import flip_xyz


@dataclass(frozen=True)
class GateSet:
    """
    G flat rectangular gates, batched as arrays.

    Every gate lies in a plane facing along its normal: heading ``yaws[g]``, then pitched
    ``pitches[g]`` about the gate's lateral axis (up-positive — a positive pitch tilts the
    through axis upward). Half-extents are metres from centre: ``inner_half[g]`` =
    (lateral, vertical) of the opening, ``outer_half[g]`` likewise of the outer edge.

    FIELDS ARE INTERNAL NED (DESIGN.md §3a): centres NED, yaws about world down, plane
    axes NED vectors — the frame the renderer math runs in. Public access goes
    through the builders (:meth:`build`, :meth:`single`, the course functions), which take
    world z-up FLU definitions, and the ``*_world`` properties, which read z-up back.

    Construct with :meth:`build` (shared frame dimensions — the common case of one
    physical gate design placed G times) or per-gate via the course builders. All fields
    are [G, ...] float32 jnp arrays.
    """

    centers: jax.Array                      # [G, 3] gate centres (internal NED)
    yaws: jax.Array                         # [G] heading of each gate normal (rad, internal NED)
    inner_half: jax.Array                   # [G, 2] (lateral, vertical) opening half-extents, m
    outer_half: jax.Array                   # [G, 2] outer-edge half-extents, m
    pitches: jax.Array = field(default=None)     # type: ignore[assignment]  # [G] rad, up-positive
    depths: jax.Array = field(default=None)      # type: ignore[assignment]  # [G] m along the normal (0 = flat)
    # world-frame plane axes, derived from yaws+pitches in __post_init__; [G, 3] each,
    # internal NED. (laterals stay horizontal — pitch rotates about them; verticals are
    # in-plane.)
    normals: jax.Array = field(default=None)     # type: ignore[assignment]
    laterals: jax.Array = field(default=None)    # type: ignore[assignment]
    verticals: jax.Array = field(default=None)   # type: ignore[assignment]

    # Defaults are a typical orange racing gate: ~0.55 m square opening, ~0.225 m
    # frame band per side (outer ~1.0 m square). The mitred corners of the physical gate
    # are approximated as a square-cornered band — a small difference at the four
    # corners; refine if sim-to-real keys on it.
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
        centers: Sequence[Sequence[float]] | jax.Array,
        yaws: Sequence[float] | jax.Array,
        inner_half: tuple[float, float] = DEFAULT_INNER_HALF,
        frame_width: float = DEFAULT_FRAME_WIDTH,
        pitches: Sequence[float] | jax.Array | None = None,
        depth: float = 0.0,
    ) -> "GateSet":
        """
        G gates of one physical design at arbitrary centres/headings, world z-up FLU.

        ``centers`` [G, 3] z-up metres (z = altitude above ground) and ``yaws`` [G] rad
        about world +z (0 = +x, 90° = +y) place each gate; ``pitches`` [G] (optional,
        rad, up-positive) tilts them. ``inner_half`` (lateral, vertical), ``frame_width``
        (band width per side, so outer = inner + frame_width) and ``depth`` (frame
        thickness along the normal, m; 0 = flat plane) are shared by all gates.
        """
        centers_a = jnp.asarray(centers, jnp.float32).reshape(-1, 3)
        yaws_a = jnp.asarray(yaws, jnp.float32).reshape(-1)
        if centers_a.shape[0] != yaws_a.shape[0]:
            raise ValueError(
                f"{centers_a.shape[0]} centers but {yaws_a.shape[0]} yaws")
        g = centers_a.shape[0]
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
        return _gateset_from_world(centers_a, yaws_a, inner, outer, pitches_a, depths)

    @classmethod
    def single(
        cls,
        center: tuple[float, float, float] = (3.0, 0.0, 1.5),
        yaw: float = 0.0,
        inner_half: tuple[float, float] = DEFAULT_INNER_HALF,
        outer_half: tuple[float, float] | None = None,
        pitch: float = 0.0,
        depth: float = 0.0,
    ) -> "GateSet":
        """One gate at a z-up world ``center`` (m), yaw rad about +z, pitch up-positive."""
        ow, oh = (outer_half if outer_half is not None
                  else (inner_half[0] + cls.DEFAULT_FRAME_WIDTH,
                        inner_half[1] + cls.DEFAULT_FRAME_WIDTH))
        return _gateset_from_world(
            jnp.asarray([center], jnp.float32),
            jnp.asarray([yaw], jnp.float32),
            jnp.asarray([inner_half], jnp.float32),
            jnp.asarray([[ow, oh]], jnp.float32),
            jnp.asarray([pitch], jnp.float32),
            jnp.asarray([depth], jnp.float32))

    def __len__(self) -> int:
        return int(self.centers.shape[0])

    # -- public z-up read-back (DESIGN.md §3: public APIs speak z-up FLU) --------------

    @property
    def centers_world(self) -> jax.Array:
        """[G, 3] gate centres in world z-up coordinates (z = altitude)."""
        return flip_xyz(self.centers)

    @property
    def normals_world(self) -> jax.Array:
        """[G, 3] unit facing directions, z-up world. A forward pass crosses along +normal."""
        return flip_xyz(self.normals)

    @property
    def laterals_world(self) -> jax.Array:
        """
        [G, 3] horizontal in-plane axes, z-up world. Sign follows the internal
        convention — the slab tests are symmetric in it, so only the line matters.
        """
        return flip_xyz(self.laterals)

    @property
    def verticals_world(self) -> jax.Array:
        """[G, 3] in-plane vertical axes, z-up world; points up for an upright gate."""
        return flip_xyz(self.verticals)


def _gateset_from_world(
    centers_world: jax.Array,
    yaws_world: jax.Array,
    inner_half: jax.Array,
    outer_half: jax.Array,
    pitches_world: jax.Array | None,
    depths: jax.Array,
) -> GateSet:
    """
    The single z-up→NED conversion site (DESIGN.md §3a): centres get the (x, -y, -z)
    flip; yaw flips sign (public yaw is about +z/up, the internal yaw about NED down);
    pitch passes through (up-positive in both frames — "up" survives the flip).
    """
    centers_ned = flip_xyz(jnp.asarray(centers_world, jnp.float32).reshape(-1, 3))
    yaws_ned = -jnp.asarray(yaws_world, jnp.float32).reshape(-1)
    pitches = (jnp.zeros_like(yaws_ned) if pitches_world is None
               else jnp.asarray(pitches_world, jnp.float32).reshape(-1))
    return GateSet(centers=centers_ned, yaws=yaws_ned, inner_half=inner_half,
                   outer_half=outer_half, pitches=pitches, depths=depths)


_BIG = 1e30


def _slab(oo: jax.Array, dd: jax.Array, h) -> tuple[jax.Array, jax.Array]:
    """
    Parameter interval (lo, hi) where |oo + t·dd| <= h — one axis of a box/prism
    intersection. Empty when lo > hi. A direction component ~0 gives the full line when
    the origin is inside the slab, else the empty interval. Broadcasts (oo, dd, h alike);
    h may be 0 (degenerate flat slab).
    """
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
    """
    Slab intervals for the gate SOLID in gate-frame coordinates: the outer box
    (|n| <= half_depth, |lat| <= ow, |vert| <= oh) as [a0, a1], and the inner-opening
    prism (|lat| < iw, |vert| < ih, unbounded along the normal) as [b0, b1]. The solid is
    A \\ B — the four frame bars exactly (their union is the outer box minus the opening).
    """
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
    """
    True where [a0, a1] \\ [b0, b1] intersects [t_lo, t_hi]: some parameter range lies
    inside the outer box but outside the opening — before entering the hole, or after
    exiting it.
    """
    lo = jnp.maximum(a0, t_lo)
    before = lo <= jnp.minimum(jnp.minimum(a1, b0), t_hi)
    after = jnp.maximum(lo, b1) <= jnp.minimum(a1, t_hi)
    return before | after


def classify_crossings(
    prev_pos: jax.Array, pos: jax.Array, gates: GateSet,
    body_radius: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Classify the prev_pos→pos segment against every gate's SOLID frame. Positions are
    [F, 3] world z-up FLU metres (converted to the internal NED once, here).

    ``hit_frame``: the segment intersects the frame solid anywhere — the front face, a
    bar's side wall while transiting the opening obliquely, or the top of the frame
    during a descent that never crosses the gate plane. A clean ``pass`` crosses the
    gate's centre plane with the crossing point inside the opening AND never touches the
    solid; passes split by direction (forward = along the gate's world normal, its facing
    direction). ``depths[g] = 0`` reduces to the classic flat-plane test.

    ``body_radius`` models the drone as a sphere instead of a point: the gate solid is
    inflated by r (Minkowski sum, square-cornered box approximation — outer + r,
    opening - r, depth + 2r) so a pass needs r of clearance from every bar and grazing
    within r of one is a hit. The RENDERED mask is untouched — the camera sees the
    physical gate; only contact inflates.

    Returns (pass_fwd, pass_bwd, hit_frame), each [F, G] bool. Pure geometry, shared by
    the gate task's pass/collision events; unit-testable without the env.
    """
    prev_ned = flip_xyz(prev_pos)
    pos_ned = flip_xyz(pos)
    r = float(body_radius)
    d_prev = prev_ned[:, None, :] - gates.centers                      # [F, G, 3]
    seg = (pos_ned - prev_ned)[:, None, :]                             # [F, 1, 3]
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


# -- course builders ---------------------------------------------------------------------
#
# The YAML-facing form is from_waypoints — a plain list of [x, y, alt, yaw_deg] or
# [x, y, alt, yaw_deg, pitch_deg] rows, one per gate, in flight order — so a course lives
# in a config file as data. The parametric builders (line, circle, figure_eight) generate
# standard shapes. All coordinates are world z-up FLU; yaw_deg is the heading of the
# gate's normal about world +z (0 = +x, 90 = +y), the intended flight direction THROUGH
# the gate: the course task counts a pass when the drone crosses along +normal.

Waypoint = Sequence[float]  # [x_m, y_m, alt_m, yaw_deg(, pitch_deg)]


def from_waypoints(
    waypoints: Sequence[Waypoint],
    inner_half: tuple[float, float] = GateSet.DEFAULT_INNER_HALF,
    frame_width: float = GateSet.DEFAULT_FRAME_WIDTH,
    depth: float = 0.0,
) -> GateSet:
    """
    Gates at explicit ``[x, y, alt, yaw_deg(, pitch_deg)]`` rows (z-up world metres,
    degrees), in flight order (pitch optional per row, up-positive, default 0). ``depth``
    is the frame thickness along the normal, shared by all gates (0 = flat plane).
    """
    if not waypoints:
        raise ValueError("waypoints must contain at least one gate")
    centers = [(float(w[0]), float(w[1]), float(w[2])) for w in waypoints]
    yaws = [math.radians(float(w[3])) for w in waypoints]
    pitches = [math.radians(float(w[4])) if len(w) > 4 else 0.0 for w in waypoints]
    return GateSet.build(centers, yaws, inner_half=inner_half,
                         frame_width=frame_width, pitches=pitches, depth=depth)


def line(
    n_gates: int,
    spacing_m: float = 4.0,
    alt_m: float = 1.5,
    yaw_deg: float = 0.0,
    inner_half: tuple[float, float] = GateSet.DEFAULT_INNER_HALF,
    frame_width: float = GateSet.DEFAULT_FRAME_WIDTH,
) -> GateSet:
    """
    ``n_gates`` in a straight line along the shared heading, ``spacing_m`` apart, the
    first gate at the origin. A drag strip — the simplest multi-gate course (it does not
    loop back; accept the long return leg or use a closed course).
    """
    yaw = math.radians(yaw_deg)
    dx, dy = math.cos(yaw), math.sin(yaw)
    rows = [[i * spacing_m * dx, i * spacing_m * dy, alt_m, yaw_deg]
            for i in range(n_gates)]
    return from_waypoints(rows, inner_half=inner_half, frame_width=frame_width)


def circle(
    n_gates: int,
    radius_m: float = 5.0,
    alt_m: float = 1.5,
    clockwise: bool = False,
    inner_half: tuple[float, float] = GateSet.DEFAULT_INNER_HALF,
    frame_width: float = GateSet.DEFAULT_FRAME_WIDTH,
) -> GateSet:
    """
    ``n_gates`` evenly around a circle, each facing along the direction of travel
    (tangent), so flying the loop passes every gate head-on. Counter-clockwise viewed
    from above (+z) unless ``clockwise``.
    """
    sense = -1.0 if clockwise else 1.0
    rows = []
    for i in range(n_gates):
        theta = sense * 2.0 * math.pi * i / n_gates
        x = radius_m * math.cos(theta)
        y = radius_m * math.sin(theta)
        # travel direction = d(position)/d(theta) · sense = (-sinθ, cosθ) · sense
        yaw = math.degrees(math.atan2(sense * math.cos(theta),
                                      sense * -math.sin(theta)))
        rows.append([x, y, alt_m, yaw])
    return from_waypoints(rows, inner_half=inner_half, frame_width=frame_width)


def figure_eight(
    k_gates_per_lobe: int,
    lobe_radius_m: float = 5.0,
    lobe_half_width_m: float = 3.0,
    alt_m: float = 1.5,
    inner_half: tuple[float, float] = GateSet.DEFAULT_INNER_HALF,
    outer_half: tuple[float, float] | None = None,
    depth: float = 0.0,
) -> GateSet:
    """
    The canonical figure-eight course: two ellipse lobes tangent at the crossover, 2·k
    gates yawed along the flight tangent, all at ``alt_m`` (DESIGN.md §9).

    Each lobe is an ellipse with semi-axes ``lobe_radius_m`` (along x — the apex sits
    2·lobe_radius_m from the crossover) by ``lobe_half_width_m`` (along y), the two
    lobes meeting at the origin. Per lobe the k gates sit at the evenly spaced ellipse
    angles φ = j·2π/(k+1), j = 1..k — the crossover slot stays empty, and consecutive
    crossover transits run along the two DIFFERENT diagonals (the alternating-crossing
    property the gate task pins). Flight order: right lobe (down-right first), then left
    lobe.

    The 6-gate default reproduces the nav-jax FigureEight map EXACTLY (z-up from its
    NED rows; nav-jax tests/test_gate_spawn.py): shoulders (±5, ∓3) m, apexes (±10, 0),
    1.5 m altitude — a 20 x 6 m footprint. At the 6-gate slots the ellipse tangents are
    axis-aligned, so the shoulders are flown straight along ±x and the apexes along +y,
    matching that map's yaws {0°, 90°, 180°} gate for gate. Gate 0 (first) is the right
    lobe's lower shoulder.

    ``outer_half`` defaults to inner + DEFAULT_FRAME_WIDTH per side; ``depth`` is the
    frame thickness along the normal (0 = flat plane). Returns the gates in flight order;
    the loop closes from the last gate back through the crossover to the first.
    """
    if k_gates_per_lobe < 1:
        raise ValueError("k_gates_per_lobe must be >= 1")
    k = int(k_gates_per_lobe)
    a = float(lobe_radius_m)  # lobe semi-major: apex at 2a from the crossover
    w = float(lobe_half_width_m)  # lobe semi-minor: half the course width
    centers: list[tuple[float, float, float]] = []
    yaws: list[float] = []
    for i in range(2 * k):
        side = 1.0 if i < k else -1.0  # right lobe first, then the left
        phi = (i % k + 1) * 2.0 * math.pi / (k + 1)
        centers.append(
            (side * a * (1.0 - math.cos(phi)), -w * math.sin(phi), float(alt_m))
        )
        # ellipse tangent in flight order: d/dφ = (side·a·sin φ, -w·cos φ)
        yaws.append(math.atan2(-w * math.cos(phi), side * a * math.sin(phi)))
    iw, ih = float(inner_half[0]), float(inner_half[1])
    ow, oh = (outer_half if outer_half is not None
              else (iw + GateSet.DEFAULT_FRAME_WIDTH, ih + GateSet.DEFAULT_FRAME_WIDTH))
    g = 2 * k
    inner = jnp.broadcast_to(jnp.array([iw, ih], jnp.float32), (g, 2))
    outer = jnp.broadcast_to(jnp.array([float(ow), float(oh)], jnp.float32), (g, 2))
    depths = jnp.full((g,), float(depth), jnp.float32)
    return _gateset_from_world(
        jnp.asarray(centers, jnp.float32), jnp.asarray(yaws, jnp.float32),
        inner, outer, None, depths)
