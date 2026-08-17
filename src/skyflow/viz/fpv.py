"""
FPV pixels (DESIGN.md §13): the composite and the pose re-renderer.

Two deliberately different panes share this module. The POLICY pane is the observation
verbatim — `obs_mask` slices the mask block out of the obs vector the policy received,
corruption included; `compose` only adds the display-only floor/sky behind it so a human
keeps orientation. The PILOT pane is `PilotCam`: a fresh analytic render at display
resolution from pose alone — which is also why FlightLog stores poses, never pixels: the
camera is a pure function, so any log replays at any resolution forever.

`compose`/`upscale`/`obs_mask` are pure numpy; `PilotCam` touches jax and the renderer,
lazily, and jits once per resolution.
"""

import dataclasses
from typing import Any

import numpy as np

from skyflow.viz import palette

__all__ = ["PilotCam", "compose", "obs_mask", "upscale"]


def compose(mask: np.ndarray | None, floor: np.ndarray | None = None) -> np.ndarray:
    """
    [H,W] coverage(s) in [0,1] → [H,W,3] uint8: gate orange over floor grey over sky.
    Either channel may be None; shapes must agree when both are given.
    """
    if mask is None and floor is None:
        raise ValueError("compose needs at least one of mask / floor")
    shape = mask.shape if mask is not None else floor.shape  # type: ignore[union-attr]
    img = np.empty((*shape, 3), np.float32)
    img[:] = palette.FPV_SKY
    if floor is not None:
        f = np.clip(np.asarray(floor, np.float32), 0.0, 1.0)[..., None]
        img = img * (1.0 - f) + np.asarray(palette.FPV_FLOOR, np.float32) * f
    if mask is not None:
        m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)[..., None]
        img = img * (1.0 - m) + np.asarray(palette.FPV_GATE, np.float32) * m
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def upscale(img: np.ndarray, k: int) -> np.ndarray:
    """Nearest-neighbour xk — deliberately blocky: the policy pane shows real pixels."""
    if k <= 1:
        return img
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def obs_mask(
    obs_row: np.ndarray,
    image_shape: tuple[int, int, int],
    layout: dict[str, slice],
    term: str = "mask",
) -> np.ndarray:
    """
    The [H,W] image block sliced VERBATIM from one flat obs row (honesty rule, §13).
    `term` names the block in the obs layout — "mask" for the shipped vision task; a
    custom task passes its own name (e.g. Viewer.for_env(env, image_term="depth")).
    """
    h, w, _c = image_shape
    if term not in layout:
        raise KeyError(f"obs layout has no term {term!r}; terms: {sorted(layout)}")
    return np.asarray(obs_row[layout[term]], np.float32).reshape(h, w)


class PilotCam:
    """
    Display-resolution re-render of the analytic camera from pose alone.

    Keeps the given camera's lens and mount but swaps resolution (supersample 1 — the
    pilot pane is large enough not to alias). With `gates=None` it renders the floor/sky
    horizon only, which is the pilot view for tasks with no gate geometry at all.
    """

    def __init__(
        self,
        camera: Any = None,
        gates: Any = None,
        *,
        height: int = 192,
        width: int = 256,
        floor_half: float | None = None,
    ) -> None:
        from skyflow.vision.camera import CameraModel

        base = camera if camera is not None else CameraModel()
        self.camera = dataclasses.replace(base, height=height, width=width, supersample=1)
        self.gates = gates
        self.floor_half = floor_half
        self._fn: Any = None  # jitted on first render

    def channels(
        self, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """One pose (world z-up, wxyz) → ([H,W] gate coverage or None, [H,W] floor)."""
        if self._fn is None:
            import jax

            from skyflow.vision.renderer import render_floor, render_masks

            cam, gates, floor_half = self.camera, self.gates, self.floor_half

            def _render(p, q):
                floor = render_floor(cam, p, q, half_extent=floor_half)
                if gates is None:
                    return None, floor
                return render_masks(cam, gates, p, q), floor

            self._fn = jax.jit(_render)
        p = np.asarray(pos, np.float32).reshape(1, 3)
        q = np.asarray(quat, np.float32).reshape(1, 4)
        mask, floor = self._fn(p, q)
        return (None if mask is None else np.asarray(mask)[0]), np.asarray(floor)[0]

    def render(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """One pose (world z-up, wxyz) → [H,W,3] uint8 composite."""
        return compose(*self.channels(pos, quat))
