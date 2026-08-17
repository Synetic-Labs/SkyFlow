"""
World → screen projection for the scene pane (DESIGN.md §13).

A fixed linear projection, fitted ONCE from the scene's AABB — no orbit camera to fight
during teleop, and every session over the same scene produces comparable footage. Three
kinds: "iso" (the default 30° isometric, z drawn up), "top" (plan view, +x right, +y up)
and "profile" (side view, +x right, +z up). Pure numpy.
"""

import math

import numpy as np

__all__ = ["KINDS", "Projection"]

KINDS = ("iso", "top", "profile")

_C30, _S30, _ZS = math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.95

#: 2x3 world→screen bases. Screen y grows DOWN, so world "up" carries a negative row-1
#: coefficient (iso: altitude lifts the point; top: +y is away from the viewer).
_BASES: dict[str, np.ndarray] = {
    "iso": np.array([[_C30, -_C30, 0.0], [_S30, _S30, -_ZS]]),
    "top": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    "profile": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
}


class Projection:
    """Affine world→pixel map: `points([N,3]) -> [N,2]`. Build with `fit`."""

    def __init__(self, kind: str, scale: float, offset: tuple[float, float]) -> None:
        if kind not in _BASES:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.scale = float(scale)
        self.offset = (float(offset[0]), float(offset[1]))
        self._basis = _BASES[kind]

    #: Pixels per world metre (approximate for iso, exact for top/profile) — the glyph
    #: level-of-detail decision reads this.
    @property
    def ppm(self) -> float:
        return self.scale

    def points(self, pts: np.ndarray) -> np.ndarray:
        """[N,3] world → [N,2] pixels (float)."""
        uv = np.asarray(pts, np.float64).reshape(-1, 3) @ self._basis.T
        return uv * self.scale + np.asarray(self.offset)

    def point(self, p: np.ndarray | tuple) -> tuple[float, float]:
        """One world point → (x, y) pixels."""
        q = self.points(np.asarray(p, np.float64).reshape(1, 3))[0]
        return float(q[0]), float(q[1])

    @classmethod
    def fit(
        cls,
        kind: str,
        lo: np.ndarray,
        hi: np.ndarray,
        rect: tuple[float, float, float, float],
        margin: float = 0.08,
    ) -> "Projection":
        """
        Projection fitted so the world AABB (lo, hi) fills `rect` = (x, y, w, h) pixels,
        centred, with `margin` of the rect kept clear on every side.
        """
        basis = _BASES[kind]
        lo = np.asarray(lo, np.float64)
        hi = np.asarray(hi, np.float64)
        corners = np.array(
            [[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]
        )
        uv = corners @ basis.T
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        span_u = max(umax - umin, 1e-6)
        span_v = max(vmax - vmin, 1e-6)
        x, y, w, h = rect
        avail_w = w * (1.0 - 2.0 * margin)
        avail_h = h * (1.0 - 2.0 * margin)
        scale = min(avail_w / span_u, avail_h / span_v)
        offset = (
            x + (w - (umin + umax) * scale) / 2.0,
            y + (h - (vmin + vmax) * scale) / 2.0,
        )
        return cls(kind, scale, offset)
