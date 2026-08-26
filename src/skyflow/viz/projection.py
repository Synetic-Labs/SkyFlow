"""
World → screen projection for the scene pane (DESIGN.md §13).

An orthographic camera around the z-up world: azimuth/elevation angles build the 2x3
basis, fitted ONCE from the scene's AABB so every session over the same scene starts
from the same framing. Three presets ("iso", "top", "profile" — the V key) plus full
user control through `orbit`, `pan` and `zoom_at` — the viewer maps the standard
mouse scheme onto them (left-drag orbit, right-drag pan, wheel zoom). Pure numpy.
"""

import math

import numpy as np

__all__ = ["KINDS", "Projection"]

KINDS = ("iso", "top", "profile")

#: Preset camera angles per kind: (azimuth°, elevation°). Elevation 90 looks straight
#: down (plan view); 0 sits on the horizon (side view).
_PRESETS: dict[str, tuple[float, float]] = {
    "iso": (-135.0, 30.0),
    "top": (-90.0, 90.0),
    "profile": (-90.0, 0.0),
}


def _camera_basis(azim: float, elev: float) -> np.ndarray:
    """2x3 world→screen rows for an orthographic camera at (azimuth°, elevation°)
    looking at the scene, z-up world, screen y growing DOWN."""
    az, el = math.radians(azim), math.radians(elev)
    sa, ca, se, ce = math.sin(az), math.cos(az), math.sin(el), math.cos(el)
    return np.array([[-sa, ca, 0.0], [se * ca, se * sa, -ce]])


class Projection:
    """Affine world→pixel map: `points([N,3]) -> [N,2]`. Build with `fit`."""

    def __init__(self, kind: str, scale: float, offset: tuple[float, float]) -> None:
        if kind not in _PRESETS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.scale = float(scale)
        self.offset = (float(offset[0]), float(offset[1]))
        self.azim, self.elev = _PRESETS[kind]
        self._home = (self.azim, self.elev)
        self._basis = _camera_basis(self.azim, self.elev)
        self._pivot = np.zeros(3)  # orbit centre; fit() sets the AABB centre

    @property
    def orbited(self) -> bool:
        """True once the camera left its preset angles (label hint for the viewer)."""
        return (self.azim, self.elev) != self._home

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

    def pan(self, dx: float, dy: float) -> None:
        """Shift the view by (dx, dy) pixels — right/middle mouse drag."""
        self.offset = (self.offset[0] + float(dx), self.offset[1] + float(dy))

    def orbit(self, dazim: float, delev: float, pivot: np.ndarray | tuple | None = None) -> None:
        """Rotate the camera by (dazim°, delev°) about `pivot` (default: the fit
        centre) — left mouse drag. The pivot's pixel stays put; elevation clamps to
        [-89°, 89°] so the basis never degenerates."""
        pv = np.asarray(self._pivot if pivot is None else pivot, np.float64)
        anchor = self.points(pv)[0]
        self.azim = (self.azim + float(dazim) + 180.0) % 360.0 - 180.0
        self.elev = float(np.clip(self.elev + float(delev), -89.0, 89.0))
        self._basis = _camera_basis(self.azim, self.elev)
        moved = self.points(pv)[0]
        self.offset = (self.offset[0] + float(anchor[0] - moved[0]),
                       self.offset[1] + float(anchor[1] - moved[1]))

    def zoom_at(self, px: float, py: float, factor: float) -> None:
        """Scale the view by `factor`, keeping the world point under pixel (px, py)
        fixed. The scale clamps to [0.05, 5000] px/m."""
        new = float(np.clip(self.scale * float(factor), 0.05, 5000.0))
        f = new / self.scale
        self.scale = new
        self.offset = (px + (self.offset[0] - px) * f, py + (self.offset[1] - py) * f)

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
        basis = _camera_basis(*_PRESETS[kind])
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
        proj = cls(kind, scale, offset)
        proj._pivot = (lo + hi) / 2.0
        return proj
