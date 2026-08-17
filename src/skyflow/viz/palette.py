"""
Shared viewer palette (DESIGN.md §13).

One dark ground and one accent — the mask orange the FPV composite already speaks — so
every pane, the recorded footage, and the FPV panes read as one instrument. Styles are the
names primitives carry through serde; panes resolve them here.
"""

Color = tuple[int, int, int]

BG: Color = (12, 14, 17)  # pane ground (near-black, cool)
GRID: Color = (32, 38, 46)  # floor grid lines
DIM: Color = (57, 65, 76)  # course lines, de-emphasised geometry
WIRE: Color = (85, 96, 110)  # default wireframe
MUTED: Color = (152, 160, 168)  # labels, secondary text
BRIGHT: Color = (223, 227, 232)  # glyph body, focused geometry
ACCENT: Color = (255, 74, 16)  # the mask orange: active gate, heading, goal
GOOD: Color = (70, 200, 120)  # pass pulse / success
BAD: Color = (200, 60, 40)  # crash marks

#: Style tag → color, the vocabulary `primitives.*.style` draws from.
STYLES: dict[str, Color] = {
    "grid": GRID,
    "dim": DIM,
    "wire": WIRE,
    "bright": BRIGHT,
    "accent": ACCENT,
    "good": GOOD,
    "bad": BAD,
}

# FPV composite classes (matches the analytic renderer's world: gates, floor, sky).
FPV_SKY: Color = (8, 9, 11)
FPV_FLOOR: Color = (56, 61, 68)
FPV_GATE: Color = (255, 64, 0)


def lerp(a: Color, b: Color, t: float) -> Color:
    """Linear blend a→b, t in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def dim(color: Color, alpha: float) -> Color:
    """Fake alpha on the dark ground: blend toward BG (cheaper than per-pixel alpha)."""
    return lerp(BG, color, alpha)
