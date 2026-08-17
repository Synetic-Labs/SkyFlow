"""
skyflow.viz — the optional viewer (DESIGN.md §13). Install with the `viz` extra.

Lazy exports: the pure pieces (Scene + primitives, FlightLog, the FPV composite,
ViewFrame) import without pygame; `Viewer` and `replay` pull it in on first touch with
install guidance if it is missing. Core skyflow modules never import this package.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Box",
    "FlightLog",
    "Gate",
    "Grid",
    "Marker",
    "Path",
    "PilotCam",
    "Projection",
    "ReplayLog",
    "Scene",
    "ViewFrame",
    "Viewer",
    "compose",
    "obs_mask",
    "register_primitive",
    "replay",
    "snapshot",
    "upscale",
]

_HOME: dict[str, str] = {
    "Scene": "skyflow.viz.primitives",
    "Grid": "skyflow.viz.primitives",
    "Path": "skyflow.viz.primitives",
    "Gate": "skyflow.viz.primitives",
    "Box": "skyflow.viz.primitives",
    "Marker": "skyflow.viz.primitives",
    "register_primitive": "skyflow.viz.primitives",
    "ViewFrame": "skyflow.viz.frame",
    "snapshot": "skyflow.viz.frame",
    "Projection": "skyflow.viz.projection",
    "compose": "skyflow.viz.fpv",
    "obs_mask": "skyflow.viz.fpv",
    "upscale": "skyflow.viz.fpv",
    "PilotCam": "skyflow.viz.fpv",
    "FlightLog": "skyflow.viz.record",
    "ReplayLog": "skyflow.viz.record",
    "Viewer": "skyflow.viz.viewer",
    "replay": "skyflow.viz.replay",
}


def __getattr__(name: str) -> Any:
    home = _HOME.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(home), name)


def __dir__() -> list[str]:
    return sorted(__all__)
