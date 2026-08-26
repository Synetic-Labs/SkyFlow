"""
Instrument-strip builder (DESIGN.md §13) — vehicle truth plus user-selected channels.

Left to right: stick crosses (AETR in sticks mode, four action bars in motors mode) with
an arm lamp under them (when the caller passes `armed`), rotor speed bars from
plant[13:17], an attitude horizon (roll/pitch printed under it) and a heading compass
(heading printed under it), a speed dial and a cockpit-style climb dial (zero at the
left, needle up = climb), an episode-length bar chart (when the caller tracks one),
then one graph per named channel — reward drawn last. The fixed instruments are vehicle truth —
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


def _nice_ceil(v: float) -> float:
    """The smallest 1/2/5 x 10^k at or above v — gauge full-scale steps."""
    v = max(float(v), 1e-6)
    k = math.floor(math.log10(v))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10.0**k >= v * (1.0 - 1e-9):
            return m * 10.0**k
    return 10.0 ** (k + 1)  # unreachable: the m=10 rung always catches


def _dial(surface, cx: float, cy: float, r: int, value: float, full: float,
          signed: bool = False, font=None, small=None) -> None:
    """One round gauge with ONE printed number (the value, under the dial).
    Unsigned: 270° arc, gap at the bottom, needle sweeps 0..full clockwise. Signed
    (a cockpit vertical-speed dial): gap at the RIGHT, the needle rests sideways at
    the left for zero and tilts up for +value, down for -value."""
    t0 = 0.25 * math.pi if signed else 0.75 * math.pi
    sweep = 1.5 * math.pi
    arc = [
        (cx + math.cos(t0 + t * sweep) * r, cy + math.sin(t0 + t * sweep) * r)
        for t in np.linspace(0.0, 1.0, 25)
    ]
    pygame.draw.aalines(surface, palette.DIM, False, arc)
    marked = (0.5,) if signed else (0.0, 1.0)  # the reference tick(s)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        a = t0 + t * sweep
        ca, sa = math.cos(a), math.sin(a)
        color = palette.MUTED if t in marked else palette.DIM
        pygame.draw.aaline(surface, color, (cx + ca * (r - 5), cy + sa * (r - 5)),
                           (cx + ca * r, cy + sa * r))
    frac = value / max(full, 1e-9)
    frac = 0.5 + 0.5 * float(np.clip(frac, -1.0, 1.0)) if signed else float(np.clip(frac, 0.0, 1.0))
    a = t0 + frac * sweep
    pygame.draw.aaline(surface, palette.ACCENT, (cx, cy),
                       (cx + math.cos(a) * r * 0.85, cy + math.sin(a) * r * 0.85))
    pygame.draw.circle(surface, palette.MUTED, (cx, cy), 2)
    if font:
        img = font.render(f"{value:+.1f}" if signed else f"{value:.1f}", True, palette.MUTED)
        surface.blit(img, (cx - img.get_width() / 2, cy + r + 3))


def _ep_chart(surface, x: int, top: int, w: int, h: int, lengths, font=None) -> None:
    """Episode-length bars: one bar per finished episode (or per bin once the caller
    compresses), scaled so the WHOLE run always fits the panel width. The latest
    length prints in the corner."""
    pygame.draw.rect(surface, palette.DIM, pygame.Rect(x, top, w, h), 1, border_radius=4)
    arr = np.asarray(lengths, np.float64)
    if arr.shape[0] == 0:
        return
    peak = max(float(arr.max()), 1.0)
    bw = (w - 8) / arr.shape[0]
    for i, v in enumerate(arr):
        bh = max(1, round(float(v) / peak * (h - 8)))
        bx = round(x + 4 + i * bw)
        pygame.draw.rect(
            surface, palette.dim(palette.ACCENT, 0.8),
            pygame.Rect(bx, top + h - 4 - bh, max(1, math.floor(bw * 0.8)), bh),
        )
    if font:
        img = font.render(f"{int(arr[-1])}", True, palette.MUTED)
        surface.blit(img, (x + w - img.get_width() - 5, top + 3))


def _lamp(surface, x: float, y: float, w: float, on: bool, text: str, small=None) -> None:
    """A state pill: GOOD frame + text when on, dimmed when off."""
    color = palette.GOOD if on else palette.dim(palette.MUTED, 0.55)
    pygame.draw.rect(surface, color, pygame.Rect(x, y, w, 16), 1, border_radius=8)
    if small:
        img = small.render(text, True, color)
        surface.blit(img, (x + (w - img.get_width()) / 2, y + 2))


def draw_hud(
    surface,
    rect: tuple[int, int, int, int],
    frame: ViewFrame,
    *,
    control: str = "motors",
    omega_max: float | None = None,
    histories: dict[str, Sequence[float]] | None = None,
    armed: bool | None = None,
    ranges: dict[str, float] | None = None,
    episodes: Sequence[float] | None = None,
    font=None,
    small=None,
) -> None:
    """
    Draw the instrument strip for the focused world into `rect`.

    `armed` (when not None) lights a lamp under the stick/action group. `ranges` is a
    caller-OWNED dict of gauge full-scales: this function grows its entries in place
    (1/2/5 steps, never shrinking), so a persistent caller gets dials with a steady
    range instead of a per-frame autoscale. `episodes` is a sequence of finished
    episode lengths (steps) for the bar chart; None hides the panel.
    """
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

    lamp_y = top + box + 8  # under the stick/action group, above the caption row
    x = x0 + 14
    if frame.action is not None:
        a = frame.action[f]
        if control == "sticks":  # AETR, mode-2 sticks: left = yaw/throttle, right = roll/pitch
            _stick_box(surface, x, top, box, a[3], a[2])
            _stick_box(surface, x + box + 10, top, box, a[0], a[1])
            if armed is not None and lamp_y + 18 <= y0 + h - 22:
                _lamp(surface, x, lamp_y, 2 * box + 10,
                      bool(armed), "ARMED" if armed else "DISARMED", small=small)
            caption(x, "STICKS AETR")
            x += 2 * box + 34
        else:
            # advance from _bars' own return — the one home of the bar geometry
            end = _bars(surface, x + 4, top, box, (a + 1.0) / 2.0, palette.MUTED)
            if armed is not None and lamp_y + 18 <= y0 + h - 22:
                _lamp(surface, x, lamp_y, end - 5 - x,
                      bool(armed), "ARMED" if armed else "DISARMED", small=small)
            caption(x, "ACTION")
            x = end + 30

    norm = omega_max if omega_max else max(float(np.abs(frame.rotor_speeds).max()), 1.0)
    end = _bars(surface, x + 4, top, box, frame.rotor_speeds[f] / norm, palette.MUTED)
    caption(x, "MOTORS Ω")
    x = end + 30

    r = box // 2
    rot = quat_to_rot(frame.quat[f])
    _horizon(surface, x + r, top + r, r, frame.quat[f])
    if small:  # numeric read-back under the ball, same angles the ball draws
        pitch = -math.degrees(math.asin(float(np.clip(rot[2, 0], -1.0, 1.0))))
        roll = math.degrees(math.atan2(float(rot[2, 1]), float(rot[2, 2])))
        img = small.render(f"R{roll:+.0f} P{pitch:+.0f}", True, palette.MUTED)
        surface.blit(img, (x + r - img.get_width() / 2, top + 2 * r + 3))
    caption(x, "ATTITUDE")
    x += 2 * r + 26

    _compass(surface, x + r, top + r, r, frame.quat[f], font=small)
    if small:
        fwd = rot[:, 0]
        if math.hypot(float(fwd[0]), float(fwd[1])) >= 1e-6:
            hdg = (math.degrees(math.atan2(-float(fwd[1]), float(fwd[0]))) + 360.0) % 360.0
            img = small.render(f"{hdg:03.0f}°", True, palette.MUTED)
            surface.blit(img, (x + r - img.get_width() / 2, top + 2 * r + 3))
    caption(x, "HEADING")
    x += 2 * r + 26

    if ranges is None:
        ranges = {}
    speed = float(np.linalg.norm(frame.vel[f]))
    climb = float(frame.vel[f][2])
    for name, value, floor, signed in (
        ("SPD m/s", speed, 2.0, False),
        ("CLIMB m/s", climb, 1.0, True),
    ):
        full = max(ranges.get(name, 0.0), _nice_ceil(max(abs(value), floor)))
        ranges[name] = full  # grow-only: a persistent dict keeps the range steady
        _dial(surface, x + r, top + r, r, value, full, signed=signed,
              font=font, small=small)
        caption(x, name)
        x += 2 * r + 26

    if episodes is not None:
        _ep_chart(surface, x, top, 150, box, episodes, font=small)
        caption(x, "EP STEPS")
        x += 170

    named = dict(histories or {})
    if "reward" in named:
        named["reward"] = named.pop("reward")  # reorder only: reward draws last
    graph_w = 150
    for name, values in named.items():
        if x + graph_w > rect[0] + rect[2] - 10:
            break  # no silent squeeze: extra channels wait for a wider window
        _graph(surface, x, top, graph_w, box, name, values, font=small)
        caption(x, name.upper())
        x += graph_w + 20
    surface.set_clip(None)
