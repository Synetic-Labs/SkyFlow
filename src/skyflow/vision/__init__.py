"""
Vision — analytic gate-mask rendering for camera-based tasks (DESIGN.md §2).

The package renders what a deployed gate-detection front-end would output — a coverage
mask of the gate frames — analytically, by per-pixel ray-cast against the gates' solid
frames, entirely in JAX so it runs inside jitted rollouts. mask_noise then makes the
perfect render look like a real HSV pipeline's output, persistently per world.

Frame contract (DESIGN.md §3): every public entry point speaks world z-up FLU — poses are
(pos [F, 3], quat [F, 4] wxyz body→world) and gate definitions are z-up world coordinates.
The ported NED/FRD renderer math stays behind these entry points (§3a).
"""

from skyflow.vision.camera import CameraModel
from skyflow.vision.gates import (
    GateSet,
    circle,
    classify_crossings,
    figure_eight,
    from_waypoints,
    line,
)
from skyflow.vision.mask_noise import (
    GROW_FAMILY,
    N_FAMILIES_GROW,
    NOISE_FAMILIES,
    corrupt_mask,
    erasure_at,
    fresh_noise_keys,
    grow_from_keys,
    noise_state_init,
    noise_state_step,
)
from skyflow.vision.renderer import render_floor, render_masks, render_masks_perworld

__all__ = [
    "GROW_FAMILY",
    "NOISE_FAMILIES",
    "N_FAMILIES_GROW",
    "CameraModel",
    "GateSet",
    "circle",
    "classify_crossings",
    "corrupt_mask",
    "erasure_at",
    "figure_eight",
    "fresh_noise_keys",
    "from_waypoints",
    "grow_from_keys",
    "line",
    "noise_state_init",
    "noise_state_step",
    "render_floor",
    "render_masks",
    "render_masks_perworld",
]
