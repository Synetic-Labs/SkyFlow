"""
Pinhole camera model for the analytic mask renderer (DESIGN.md §2).

Public frame contract (DESIGN.md §3): the mount is stated in body FLU — x forward, y left,
z up. The ray-cast internals are ported NED/FRD math and consume the private
``_R_frd_from_cam`` / ``_offset_frd`` twins, both derived from the same FLU fields by the
(x, -y, -z) flip, so a mount is authored in exactly one frame. The camera's own image
frame is the standard pinhole convention — x right, y down, z forward along the optical
axis — which is camera-frame, not NED/FRD, and appears in public docstrings unchanged.

Defaults model the BetaFPV C03 as flown: 64x64 policy view; 99° x 79.8° fields of view
(the real sensor undistorted then squashed to a square frame has fx ≠ fy, so the focal
lengths are tracked separately); 25° up-tilt (``mount_pitch_deg = -25``); lens 2 cm
forward and 2 cm above the body origin.

Ported from the nav-train gate renderer (validated there against MuJoCo segmentation
renders); the intrinsics and ray-grid math are unchanged.
"""

import math
from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CameraModel:
    """
    Pinhole camera mounted on the drone body (FLU).

    The optical axis is body +x pitched by ``mount_pitch_deg`` about the body lateral
    axis — positive tilts it DOWN, so the default -25° is the racing 25° up-tilt.
    ``fov_x_deg`` / ``fov_y_deg`` are the horizontal / vertical fields of view in degrees
    and may differ. ``offset_body`` is the lens position relative to the body origin in
    FLU metres (small; it mostly matters at very close range). The image frame is
    x-right / y-down / z-forward.

    Anti-aliasing: ``supersample`` casts supersample² rays per output pixel and averages
    their hits into a soft coverage in [0, 1]. >1 stops a distant gate's thin frame band
    from aliasing out of a low-res mask and yields soft edges that match a real
    downsampled binary mask. Cost scales as supersample².
    """

    height: int = 64
    width: int = 64
    fov_x_deg: float = 99.0  # horizontal FOV (BetaFPV C03, undistorted)
    fov_y_deg: float = 79.8  # vertical FOV
    mount_pitch_deg: float = -25.0  # + = optical axis down; -25 = 25° up-tilt
    offset_body: tuple[float, float, float] = (0.02, 0.0, 0.02)  # FLU m: 2 cm fwd, 2 cm up
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
    def _offset_frd(self) -> tuple[float, float, float]:
        """``offset_body`` re-expressed in body FRD — the ported renderer's frame."""
        x, y, z = self.offset_body
        return (x, -y, -z)

    @property
    def _R_frd_from_cam(self) -> jax.Array:
        """
        3x3 rotation mapping a camera-frame vector into body FRD — the ported internal
        mount math (DESIGN.md §3a). Camera axes expressed in FRD, with downward pitch θ:

          right   (cam +x) = body +y                  = [0, 1, 0]
          down    (cam +y) = forward x right          = [-sinθ, 0, cosθ]
          forward (cam +z) = body +x tilted down by θ = [cosθ, 0, sinθ]

        The columns are these body-frame axis vectors.
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
    def R_body_from_cam(self) -> jax.Array:
        """
        3x3 rotation mapping a camera-frame vector (x right, y down, z forward) into body
        FLU — the public statement of the mount. Derived from the ported FRD matrix by the
        (x, -y, -z) row flip, so both frames read one definition. At zero pitch the
        optical axis is exactly body +x; the default -25° pitch gives (cos 25°, 0, sin 25°).
        """
        return self._R_frd_from_cam * jnp.array([[1.0], [-1.0], [-1.0]], jnp.float32)

    @property
    def ray_dirs_cam(self) -> jax.Array:
        """
        Ray directions in the camera frame at the supersampled grid, shape
        [H·ss, W·ss, 3] (un-normalised; z = 1). Sub-pixel j maps to base-pixel coordinate
        (j + 0.5)/ss and through the pinhole as [(u-cx)/f, (v-cy)/f, 1]; cx/cy/f are in
        base-pixel units. Sampling at sub-pixel centres keeps the exact left/right +
        up/down symmetry about (cx, cy).
        """
        cx, cy = self.principal_point
        fx, fy = self.focal
        ss = self.supersample
        us = ((jnp.arange(self.width * ss, dtype=jnp.float32) + 0.5) / ss - cx) / fx
        vs = ((jnp.arange(self.height * ss, dtype=jnp.float32) + 0.5) / ss - cy) / fy
        uu, vv = jnp.meshgrid(us, vs, indexing="xy")  # [H·ss, W·ss]
        return jnp.stack([uu, vv, jnp.ones_like(uu)], axis=-1)
