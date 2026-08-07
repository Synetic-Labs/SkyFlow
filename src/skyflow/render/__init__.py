"""Analytic rendering — gate geometry to a coverage mask, in pure JAX.

No rasterizer and no render pass: :mod:`gatenet_renderer` computes per-pixel gate
coverage directly from pose and camera model, batched over the whole fleet, so the
camera is just more arithmetic inside the rollout scan.

* :mod:`gatenet_renderer` — the camera model and the fleet-batched mask render.
* :mod:`mask_noise` — persistent mask corruption (dropouts, blobs, ghosts) as
  domain randomization; artifacts survive across frames rather than flickering
  i.i.d., which is what a real perception front-end's failures look like.
* :mod:`courses` — gate layouts as data (``from_waypoints``) or standard shapes
  (``line``, ``circle``).
"""

from .courses import circle, from_waypoints, line
from .gatenet_renderer import CameraModel, GateSet

__all__ = ["CameraModel", "GateSet", "circle", "from_waypoints", "line"]
