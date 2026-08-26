"""
Instrument-strip builder (DESIGN.md §13) — vehicle truth plus user-selected channels.

Left to right: stick crosses (AETR in sticks mode, four action bars in motors mode), rotor
speed bars from plant[13:17], an attitude horizon and a heading compass from the
quaternion, a telemetry text block, then one small graph per named channel. The fixed instruments are vehicle truth —
valid for any quadrotor use case. Channels are whatever the caller traces (reward, goal
distance, estimator error, ...); this module knows no channel names and no task fields.
A builder, not a host: draws onto the given surface, owns no window.
"""

import math
from collections.abc import Sequence

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

__all__ = ["draw_hud"]


def _stick_box(surface, x: int, y: int, size: int, dx: float, dy: float) -> None:
    """One stick square with crosshair; (dx, dy) in [-1,1], +dy drawn up."""
    r = pygame.Rect(x, y, size, size)
    pygame.draw.rect(surface, palette.DIM, r, 1, border_radius=4)
    cx, cy = x + size / 2, y + size / 2
    pygame.draw.aaline(surface, palette.dim(palette.DIM, 0.6), (cx, y), (cx, y + size))
    pygame.draw.aaline(surface, palette.dim(palette.DIM, 0.6), (x, cy), (x + size, cy))
    px = cx + float(np.clip(dx, -1, 1)) * (size / 2 - 5)
    py = cy - float(np.clip(dy, -1, 1)) * (size / 2 - 5)
    pygame.draw.circle(surface, palette.BRIGHT, (px, py), 4)


def _bars(surface, x: int, y: int, h: int, values: np.ndarray, color) -> int:
    """Vertical bars for values in [0,1]; returns the x just past the group."""
    bw, gap = 9, 5
    pygame.draw.aaline(surface, palette.DIM, (x - 4, y), (x - 4, y + h))
    for i, v in enumerate(np.clip(values, 0.0, 1.0)):
        bh = max(1, round(float(v) * h))
        pygame.draw.rect(surface, color, pygame.Rect(x + i * (bw + gap), y + h - bh, bw, bh))
    return x + len(values) * (bw + gap)


def _horizon(surface, cx: int, cy: int, r: int, quat: np.ndarray) -> None:
    """Attitude ball: line rolled with the body, shifted by pitch."""
    rot = quat_to_rot(quat)
    # ZYX euler read-back from body→world R: roll about body x, pitch about body y
    pitch = -math.asin(float(np.clip(rot[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    pygame.draw.circle(surface, palette.DIM, (cx, cy), r, 1)
    dy = float(np.clip(pitch / (math.pi / 2.0), -1.0, 1.0)) * r * 0.8
    dxr, dyr = math.cos(-roll), math.sin(-roll)
    span = r * 0.85
    p0 = (cx - dxr * span, cy + dy - dyr * span)
    p1 = (cx + dxr * span, cy + dy + dyr * span)
    pygame.draw.aaline(surface, palette.BRIGHT, p0, p1)
    pygame.draw.aaline(surface, palette.MUTED, (cx, cy - r), (cx, cy - r + 5))


def _compass(surface, cx: int, cy: int, r: int, quat: np.ndarray, font=None) -> None:
    """Heading dial, north-up (north = world +x): needle is the body x-axis from above."""
    pygame.draw.circle(surface, palette.DIM, (cx, cy), r, 1)
    for i in range(4):  # cardinal ticks; north gets the horizon's reference treatment
        tx, ty = math.sin(i * math.pi / 2.0), -math.cos(i * math.pi / 2.0)
        color = palette.MUTED if i == 0 else palette.DIM
        pygame.draw.aaline(
            surface, color, (cx + tx * (r - 5), cy + ty * (r - 5)), (cx + tx * r, cy + ty * r)
        )
    fwd = quat_to_rot(quat)[:, 0]
    n = math.hypot(float(fwd[0]), float(fwd[1]))
    if n >= 1e-6:  # nose straight up/down leaves heading undefined: dial only
        # north-up top view of the z-up world: +x up, +y left on screen (right-handed)
        dx, dy = -float(fwd[1]) / n, -float(fwd[0]) / n
        p0 = (cx - dx * r * 0.35, cy - dy * r * 0.35)
        p1 = (cx + dx * r * 0.8, cy + dy * r * 0.8)
        pygame.draw.aaline(surface, palette.ACCENT, p0, p1)
    if font:
        img = font.render("N", True, palette.dim(palette.MUTED, 0.8))
        surface.blit(img, (cx - img.get_width() / 2, cy - r + 6))


def _graph(surface, x: int, top: int, w: int, h: int, name: str, values: Sequence[float],
           font=None) -> None:
    """One channel graph: auto-scaled trace, latest value printed in the corner."""
    pygame.draw.rect(surface, palette.DIM, pygame.Rect(x, top, w, h), 1, border_radius=4)
    arr = np.asarray(values, np.float64)
    if arr.shape[0] >= 2:
        lo, hi = float(arr.min()), float(arr.max())
        span = max(hi - lo, 1e-6)
        xs = x + 4 + np.linspace(0, w - 8, arr.shape[0])
        ys = top + h - 4 - (arr - lo) / span * (h - 8)
        pts = [(float(a), float(b)) for a, b in zip(xs, ys, strict=True)]
        pygame.draw.aalines(surface, palette.ACCENT, False, pts)
    if font and arr.shape[0]:
        img = font.render(f"{float(arr[-1]):+.3g}", True, palette.MUTED)
        surface.blit(img, (x + w - img.get_width() - 5, top + 3))


def draw_hud(
    surface,
    rect: tuple[int, int, int, int],
    frame: ViewFrame,
    *,
    control: str = "motors",
    omega_max: float | None = None,
    histories: dict[str, Sequence[float]] | None = None,
    armed: bool | None = None,
    font=None,
    small=None,
) -> None:
    """Draw the instrument strip for the focused world into `rect`."""
    surface.set_clip(pygame.Rect(rect))
    surface.fill(palette.BG, pygame.Rect(rect))
    x0, y0, _w, h = rect
    pygame.draw.aaline(surface, palette.DIM, (x0, y0), (x0 + _w, y0))
    top = y0 + 26
    box = min(56, h - 52)
    f = frame.focus

    def caption(x: int, text: str) -> None:
        if small:
            surface.blit(small.render(text, True, palette.dim(palette.MUTED, 0.8)), (x, y0 + h - 20))

    x = x0 + 14
    if frame.action is not None:
        a = frame.action[f]
        if control == "sticks":  # AETR, mode-2 sticks: left = yaw/throttle, right = roll/pitch
            _stick_box(surface, x, top, box, a[3], a[2])
            _stick_box(surface, x + box + 10, top, box, a[0], a[1])
            caption(x, "STICKS AETR" + ("  ARMED" if armed else ""))
            x += 2 * box + 34
        else:
            # advance from _bars' own return — the one home of the bar geometry
            end = _bars(surface, x + 4, top, box, (a + 1.0) / 2.0, palette.MUTED)
            caption(x, "ACTION")
            x = end + 30

    norm = omega_max if omega_max else max(float(np.abs(frame.rotor_speeds).max()), 1.0)
    end = _bars(surface, x + 4, top, box, frame.rotor_speeds[f] / norm, palette.MUTED)
    caption(x, "MOTORS Ω")
    x = end + 30

    r = box // 2
    _horizon(surface, x + r, top + r, r, frame.quat[f])
    caption(x, "ATTITUDE")
    x += 2 * r + 26

    _compass(surface, x + r, top + r, r, frame.quat[f], font=small)
    caption(x, "HEADING")
    x += 2 * r + 26

    if font:
        speed = float(np.linalg.norm(frame.vel[f]))
        alt = float(frame.pos[f][2])
        lines = [f"SPD {speed:5.2f} m/s", f"ALT {alt:5.2f} m"]
        for i, line in enumerate(lines):
            surface.blit(font.render(line, True, palette.MUTED), (x, top + i * 17))
        caption(x, "TELEMETRY")
        x += 150

    graph_w = 150
    for name, values in (histories or {}).items():
        if x + graph_w > rect[0] + rect[2] - 10:
            break  # no silent squeeze: extra channels wait for a wider window
        _graph(surface, x, top, graph_w, box, name, values, font=small)
        caption(x, name.upper())
        x += graph_w + 20
    surface.set_clip(None)
