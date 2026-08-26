"""
Scene-pane builder (DESIGN.md §13): primitives + drone glyphs onto a pygame surface.

A builder, not a host: it draws onto whatever surface it is handed and owns no window —
the live viewer, the replay host and any export call the same function, so all hosts look
identical. The pane is a fixed wireframe projection (projection.py); primitives arrive
already bind-resolved as data (primitives.py); glyphs come straight from the plant rows.

The glyph is one visual language at every scale: a bright X-frame with four rotor rings
whose accent power arcs grow just inside each ring out of its arm point (a full circle
at full power), an accent heading line with an up-flag at the tip (body +z, 1/4 of the line:
it breaks roll symmetry), and a plumb line + ground ring for altitude (the focused world
only). Watched but unfocused worlds draw the fleet mark — a dot with the same heading
line + up-flag — at EVERY zoom; `show_glyphs` (the viewer's X key) promotes them to full
glyphs. The focused world always draws the full glyph (arm length floored for
readability). The whole-fleet scatter draws bare dots.
"""

import itertools
import math

import numpy as np

try:
    import pygame
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "skyflow.viz needs pygame — install the viz extra: "
        "`uv sync --extra viz` or `pip install 'skyflow[viz]'`"
    ) from e

from skyflow.viz import palette
from skyflow.viz.frame import ViewFrame, quat_to_rot
from skyflow.viz.primitives import (
    Box,
    Gate,
    Grid,
    Marker,
    Path,
    Scene,
    draw_fn_for,
    register_primitive,
)
from skyflow.viz.projection import Projection

__all__ = ["draw_scene"]

#: Rotor centres in the body frame (FLU, unit arm length): FL, FR, RR, RL.
_ROTORS = np.array(
    [[0.71, 0.71, 0.0], [0.71, -0.71, 0.0], [-0.71, -0.71, 0.0], [-0.71, 0.71, 0.0]]
)
_RING = np.stack(
    [
        np.array([math.cos(a), math.sin(a), 0.0])
        for a in np.linspace(0.0, 2.0 * math.pi, 13)
    ]
)
#: Box.corners() edge pairs (x-fastest corner ordering).
_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

#: Readability floor for a full glyph's arm length in pixels.
_GLYPH_MIN_PX = 12.0
#: Nominal glyph arm length in world metres (display size, not the physical airframe).
_GLYPH_M = 0.35


def _polyline(surface, pts2: np.ndarray, color, width: int = 1, closed: bool = False) -> None:
    if pts2.shape[0] < 2:
        return
    pts = [(float(p[0]), float(p[1])) for p in pts2]
    if width <= 1:
        pygame.draw.aalines(surface, color, closed, pts)
    else:
        pygame.draw.lines(surface, color, closed, pts, width)


def _dashed(surface, pts2: np.ndarray, color, dash: float = 7.0, gap: float = 6.0) -> None:
    """Screen-space dashes along a polyline (the arc-length cursor spans segments)."""
    s = 0.0
    for a, b in itertools.pairwise(pts2):
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1e-6:
            continue
        u = seg / length
        t = 0.0
        while t < length:
            cyc = s % (dash + gap)
            run = min((dash - cyc) if cyc < dash else (dash + gap - cyc), length - t)
            if cyc < dash:
                p0, p1 = a + u * t, a + u * (t + run)
                pygame.draw.aaline(surface, color, tuple(p0), tuple(p1))
            t += run
            s += run


def _draw_grid(surface, proj: Projection, g: Grid, color) -> None:
    hx, hy = g.half
    for x in np.arange(-hx, hx + 1e-6, g.step):
        p = proj.points(np.array([[x, -hy, 0.0], [x, hy, 0.0]]))
        pygame.draw.aaline(surface, color, tuple(p[0]), tuple(p[1]))
    for y in np.arange(-hy, hy + 1e-6, g.step):
        p = proj.points(np.array([[-hx, y, 0.0], [hx, y, 0.0]]))
        pygame.draw.aaline(surface, color, tuple(p[0]), tuple(p[1]))


def _draw_marker(surface, proj: Projection, m: Marker, color) -> None:
    x, y = proj.point(m.center)
    s = max(5.0, m.size * proj.ppm)
    pts = np.array([[x, y - s], [x + s, y], [x, y + s], [x - s, y]])
    _polyline(surface, pts, color, width=2, closed=True)
    if m.plumb and abs(m.center[2]) > 1e-6:
        fx, fy = proj.point((m.center[0], m.center[1], 0.0))
        pygame.draw.aaline(surface, palette.dim(color, 0.35), (x, y + s), (fx, fy))
        pygame.draw.line(surface, palette.dim(color, 0.6), (fx - 4, fy), (fx + 4, fy))


def _draw_path(surface, proj: Projection, p: Path, color) -> None:
    pts = np.asarray(p.points, np.float64)
    if p.closed:
        pts = np.concatenate([pts, pts[:1]], axis=0)
    pts2 = proj.points(pts)
    if p.dashed:
        _dashed(surface, pts2, color)
    else:
        _polyline(surface, pts2, color)


def _draw_gate(surface, proj: Projection, g: Gate, color) -> None:
    accent = g.style == "accent"
    _polyline(surface, proj.points(g.corners()), color, width=2 if accent else 1, closed=True)


def _draw_box(surface, proj: Projection, b: Box, color) -> None:
    corners2 = proj.points(b.corners())
    for i, j in _BOX_EDGES:
        pygame.draw.aaline(surface, color, tuple(corners2[i]), tuple(corners2[j]))


# The built-in five go through the same public registry as user primitives (§13).
register_primitive(Grid, _draw_grid)
register_primitive(Path, _draw_path)
register_primitive(Gate, _draw_gate)
register_primitive(Box, _draw_box)
register_primitive(Marker, _draw_marker)


def _draw_glyph(
    surface,
    proj: Projection,
    pos: np.ndarray,
    quat: np.ndarray,
    omg: np.ndarray,
    *,
    focus: bool,
    full: bool,
) -> None:
    alpha = 1.0 if focus else 0.4
    body = palette.dim(palette.BRIGHT, alpha)
    accent = palette.dim(palette.ACCENT, alpha)
    x, y = proj.point(pos)

    rot = quat_to_rot(quat)
    if not full:
        # fleet mark: dot + heading tick + an up-flag at the tip (body +z, 1/4 of the
        # tick) — without the flag the mark is roll-symmetric and a roll reads as nothing
        tip = pos + rot @ np.array([0.5, 0.0, 0.0])
        flag = tip + rot @ np.array([0.0, 0.0, 0.125])
        pts = proj.points(np.stack([pos, tip, flag]))
        pygame.draw.circle(surface, body, (x, y), 2.6)
        pygame.draw.aaline(surface, body, tuple(pts[0]), tuple(pts[1]))
        pygame.draw.aaline(surface, body, tuple(pts[1]), tuple(pts[2]))
        return

    arm_px = float(np.clip(_GLYPH_M * proj.ppm, _GLYPH_MIN_PX, 44.0))
    arm_m = arm_px / proj.ppm
    if focus:  # altitude cue: plumb line down to a ground ring
        gx, gy = proj.point((pos[0], pos[1], 0.0))
        pygame.draw.aaline(surface, palette.dim(palette.WIRE, 0.5), (x, y), (gx, gy))
        ring = proj.points(
            np.array([pos[0], pos[1], 0.0]) + _RING * (1.15 * arm_m)
        )
        _polyline(surface, ring, palette.dim(palette.WIRE, 0.5))

    rotors_w = pos + (_ROTORS * arm_m) @ rot.T  # [4,3] world rotor centres
    for rw in rotors_w:
        seg = proj.points(np.stack([pos, rw]))
        pygame.draw.line(surface, body, tuple(seg[0]), tuple(seg[1]), 2)
    for d, rw, o in zip(_ROTORS, rotors_w, np.clip(omg, 0.0, 1.0), strict=True):
        ring = proj.points(rw + (_RING * (0.55 * arm_m)) @ rot.T)
        _polyline(surface, ring, body)
        if float(o) > 0.02:
            # power arc just INSIDE the prop ring: it grows both ways out of the point
            # where the arm meets the ring, and closes into a full circle at full power
            a0 = math.atan2(-d[1], -d[0])
            half = float(o) * math.pi
            angles = np.linspace(a0 - half, a0 + half, max(3, int(o * 25)))
            arc_body = np.stack(
                [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1
            )
            arc = proj.points(rw + (arc_body * (0.36 * arm_m)) @ rot.T)
            _polyline(surface, arc, accent, width=2)
    # heading: the same forward line + up-flag the fleet mark carries (accent here)
    tip = pos + rot @ np.array([1.55 * arm_m, 0.0, 0.0])
    flag = tip + rot @ np.array([0.0, 0.0, 0.25 * 1.55 * arm_m])
    head = proj.points(np.stack([pos, tip, flag]))
    pygame.draw.line(surface, accent, tuple(head[0]), tuple(head[1]), 2)
    pygame.draw.line(surface, accent, tuple(head[1]), tuple(head[2]), 2)
    pygame.draw.circle(surface, body, (x, y), 3)


def draw_scene(
    surface,
    rect: tuple[int, int, int, int],
    proj: Projection,
    scene: Scene,
    frame: ViewFrame,
    *,
    trails: dict[int, list[np.ndarray]] | None = None,
    omega_max: float | None = None,
    label: str | None = None,
    font=None,
    show_glyphs: bool = False,
) -> None:
    """
    Draw the whole scene pane into `rect`: primitives (bind-resolved from `frame`,
    dispatched through the public registry), optional whole-fleet scatter, trails, then
    one glyph per watched world. `omega_max` normalises the rotor arcs (falls back to
    the frame's own max). `show_glyphs` draws full glyphs for every watched world;
    the default keeps unfocused worlds as fleet marks at every zoom. This function holds no task knowledge — accents and live
    geometry arrive already resolved on the primitives themselves.
    """
    surface.set_clip(pygame.Rect(rect))
    surface.fill(palette.BG, pygame.Rect(rect))
    try:
        for prim in scene.resolved(frame):
            color = palette.STYLES.get(prim.style, palette.WIRE)
            draw_fn_for(prim)(surface, proj, prim, color)

        if frame.positions is not None:  # whole-fleet scatter (subsampled, fleet marks)
            stride = max(1, frame.positions.shape[0] // 4000)
            for p in proj.points(frame.positions[::stride]):
                pygame.draw.circle(surface, palette.WIRE, (float(p[0]), float(p[1])), 2)

        for w, trail in (trails or {}).items():
            if len(trail) < 2:
                continue
            base = palette.BRIGHT if w == frame.focus else palette.WIRE
            pts2 = proj.points(np.asarray(trail))
            n = pts2.shape[0] - 1
            for i in range(n):
                a = 0.15 + 0.85 * (i + 1) / n
                pygame.draw.aaline(
                    surface, palette.dim(base, a), tuple(pts2[i]), tuple(pts2[i + 1])
                )

        omg = frame.rotor_speeds
        norm = omega_max if omega_max else max(float(np.abs(omg).max()), 1.0)
        for w in range(frame.plant.shape[0]):
            _draw_glyph(
                surface,
                proj,
                frame.pos[w],
                frame.quat[w],
                omg[w] / norm,
                focus=(w == frame.focus),
                full=(w == frame.focus) or show_glyphs,
            )
            if frame.done is not None and bool(frame.done[w]):
                x, y = proj.point(frame.pos[w])
                pygame.draw.line(surface, palette.BAD, (x - 6, y - 6), (x + 6, y + 6), 2)
                pygame.draw.line(surface, palette.BAD, (x - 6, y + 6), (x + 6, y - 6), 2)

        if label and font:
            surface.blit(font.render(label, True, palette.MUTED), (rect[0] + 10, rect[1] + 8))
    finally:
        surface.set_clip(None)
