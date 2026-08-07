"""Course builders — arbitrary gate layouts as :class:`GateSet` values.

The gate task takes any ``GateSet``; these helpers cover the common ways of
making one. The YAML-facing form is :func:`from_waypoints` — a plain list of
``[north, east, alt, yaw_deg]`` or ``[north, east, alt, yaw_deg, pitch_deg]``
rows, one per gate, in flight order — so a course lives in a config file as
data. The parametric builders (:func:`line`, :func:`circle`) generate the same
rows for standard shapes; compose or extend them for anything fancier (a
figure-eight is two circles, a slalom is a line with alternating lateral
offsets, ...).

Conventions (matching gatenet_renderer / the crazyflow reference env): world NED — north, east in
metres, ``alt`` metres above ground (converted to NED z = −alt), ``yaw_deg``
the heading of the gate's normal (0 = +north, 90 = +east). The yaw is the
intended flight direction THROUGH the gate: the course task counts a pass when
the drone crosses from the −normal side to the +normal side. ``pitch_deg``
(optional, default 0) tilts the gate forward/backward about its lateral axis,
up-positive (the through axis gains an upward component).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .gatenet_renderer import GateSet

Waypoint = Sequence[float]  # [north_m, east_m, alt_m, yaw_deg(, pitch_deg)]


def from_waypoints(
    waypoints: Sequence[Waypoint],
    inner_half: tuple[float, float] = GateSet.DEFAULT_INNER_HALF,
    frame_width: float = GateSet.DEFAULT_FRAME_WIDTH,
    depth: float = 0.0,
) -> GateSet:
    """Gates at explicit ``[north, east, alt, yaw_deg(, pitch_deg)]`` rows, in
    flight order (pitch optional per row, up-positive degrees, default 0).
    ``depth`` is the frame thickness along the normal, shared by all gates
    (0 = flat plane)."""
    if not waypoints:
        raise ValueError("waypoints must contain at least one gate")
    centers = [(float(w[0]), float(w[1]), -float(w[2])) for w in waypoints]
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
    """``n_gates`` in a straight line along the shared heading, ``spacing_m``
    apart, the first gate at the origin. A drag strip — the simplest
    multi-gate course (note it does not loop back; pair with course_loop=False
    or accept the long return leg)."""
    yaw = math.radians(yaw_deg)
    dn, de = math.cos(yaw), math.sin(yaw)
    rows = [[i * spacing_m * dn, i * spacing_m * de, alt_m, yaw_deg]
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
    """``n_gates`` evenly around a circle, each facing along the direction of
    travel (tangent), so flying the loop passes every gate head-on. Counter-
    clockwise (viewed from above, north up) unless ``clockwise``."""
    sense = -1.0 if clockwise else 1.0
    rows = []
    for i in range(n_gates):
        theta = sense * 2.0 * math.pi * i / n_gates
        north = radius_m * math.cos(theta)
        east = radius_m * math.sin(theta)
        # travel direction = d(position)/d(theta) · sense = (−sinθ, cosθ) · sense
        yaw = math.degrees(math.atan2(sense * math.cos(theta),
                                      sense * -math.sin(theta)))
        rows.append([north, east, alt_m, yaw])
    return from_waypoints(rows, inner_half=inner_half, frame_width=frame_width)
