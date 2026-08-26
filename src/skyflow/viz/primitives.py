"""
Scene model — objects are data (DESIGN.md §13).

A `Scene` is a flat list of five primitive dataclasses: `Grid`, `Path`, `Gate`, `Box`,
`Marker`. Each primitive is world z-up FLU geometry plus a style tag from
`palette.STYLES`; each has exactly one draw function (scenepane.py), so a new primitive is
one dataclass plus one function. Gates hold no special status here — they are one
primitive among five. Drone glyphs are NOT primitives: they come from the plant state and
are always drawn.

`bind` makes a primitive live: a dotted ViewFrame attribute path (`"task_state.goal"`) or
any callable of the ViewFrame, re-evaluated once per displayed frame. A callable may
return a dict of field overrides or None to hide the primitive. Any other bound value is
interpreted by the primitive itself (`_apply_bind`): a Marker or Box moves its `center`, a
Path replaces its `points`, a Gate reads an active index and turns accent when it matches
its own `index`. Fleet-leading `[W, ...]` values take the focused world's row first. The
generic code never inspects task fields by name — a task names its own state in the bind.

Extension is public: `register_primitive(cls, draw_fn)` adds a user-defined primitive to
the serde registry and the scene pane's draw table — the same registry idiom as
`register_task` and `register_airframe`. `draw_fn(surface, projection, prim, color)` is
only ever called by a pygame host, so registering costs no pygame import here.

Serde: `Scene.to_dicts()`/`from_dicts()` round-trip primitives as plain dicts (JSON-safe);
string binds survive, callable binds are live-only and drop. The dict form is also the
duck-typed task hook contract — `task.viz_scene() -> list[dict]` — so tasks contribute
default scenes without importing skyflow.viz. This module is pure numpy: no pygame, no jax.
"""

import dataclasses
import json
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

__all__ = ["Box", "Gate", "Grid", "Marker", "Path", "Scene", "register_primitive", "resolve"]

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Bind = Callable[[Any], Any] | str | None


def _vec(x: Any, n: int) -> tuple[float, ...]:
    a = np.asarray(x, np.float64).reshape(-1)
    if a.shape[0] != n:
        raise ValueError(f"expected {n} components, got shape {np.asarray(x).shape}")
    return tuple(float(v) for v in a)


def _points3(x: Any) -> tuple[Vec3, ...]:
    """[N,3] (or [N,2], z padded to 0) → tuple of 3-tuples."""
    a = np.asarray(x, np.float64)
    if a.ndim != 2 or a.shape[1] not in (2, 3):
        raise ValueError(f"points must be [N,2] or [N,3], got shape {a.shape}")
    if a.shape[1] == 2:
        a = np.concatenate([a, np.zeros((a.shape[0], 1))], axis=1)
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in a)


_REGISTRY: dict[str, type] = {}
_DRAW: dict[type, Callable] = {}


def _register(cls: type) -> type:
    _REGISTRY[cls.__name__.lower()] = cls
    return cls


def register_primitive(cls: type, draw_fn: Callable) -> type:
    """
    Register a primitive class and its draw function (DESIGN.md §13).

    `cls` must be a frozen dataclass with JSON-safe fields (and an optional `bind` field
    for live geometry). `draw_fn(surface, projection, prim, color)` draws one resolved
    instance onto a pygame surface; `color` is the resolved style color. After this call,
    `Scene(cls(...))` draws, and `{"type": "<clsname>"}` dicts round-trip through serde.
    Returns `cls`, so it also works as a decorator partner. Registration is idempotent.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass primitive")
    _REGISTRY[cls.__name__.lower()] = cls
    _DRAW[cls] = draw_fn
    return cls


def draw_fn_for(prim: Any) -> Callable:
    """The registered draw function for a primitive instance (scene pane dispatch)."""
    fn = _DRAW.get(type(prim))
    if fn is None:
        raise ValueError(
            f"no draw function registered for {type(prim).__name__!r} — "
            "add it with skyflow.viz.register_primitive(cls, draw_fn)"
        )
    return fn


@_register
@dataclasses.dataclass(frozen=True)
class Grid:
    """Floor grid on world z = 0, spanning ±half metres about the origin."""

    half: Vec2 = (5.0, 5.0)
    step: float = 1.0
    style: str = "grid"
    bind: Bind = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "half", _vec(self.half, 2))


@_register
@dataclasses.dataclass(frozen=True)
class Path:
    """Polyline through world points — a course line, a reference route, a plan."""

    points: tuple[Vec3, ...]
    dashed: bool = True
    closed: bool = False
    style: str = "dim"
    bind: Bind = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _points3(self.points))

    def _apply_bind(self, value: Any) -> "Path":
        return dataclasses.replace(self, points=_points3(value))


@_register
@dataclasses.dataclass(frozen=True)
class Gate:
    """
    Rectangular frame outline: centre, in-plane lateral/vertical axes, outer half-extents.
    A bound value is read as the ACTIVE INDEX: the gate turns accent when it equals its
    own `index` — so `bind="task_state.active_gate"` on each gate is the entire
    active-gate mechanism, declared by the task, with no field names in the generic code.
    """

    center: Vec3
    lateral: Vec3
    vertical: Vec3 = (0.0, 0.0, 1.0)
    half_w: float = 0.5
    half_h: float = 0.5
    index: int | None = None
    style: str = "wire"
    bind: Bind = None

    # bound values are per-world scalars ([W] rows pick the focus row, see resolve())
    _BIND_SCALAR = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec(self.center, 3))
        object.__setattr__(self, "lateral", _vec(self.lateral, 3))
        object.__setattr__(self, "vertical", _vec(self.vertical, 3))

    @classmethod
    def from_gateset(cls, gates: Any) -> tuple["Gate", ...]:
        """One display Gate per GateSet gate, from its public z-up world read-back."""
        centers = np.asarray(gates.centers_world, np.float64)
        laterals = np.asarray(gates.laterals_world, np.float64)
        verticals = np.asarray(gates.verticals_world, np.float64)
        outer = np.asarray(gates.outer_half, np.float64)
        return tuple(
            cls(
                center=tuple(centers[g]),
                lateral=tuple(laterals[g]),
                vertical=tuple(verticals[g]),
                half_w=float(outer[g, 0]),
                half_h=float(outer[g, 1]),
                index=g,
            )
            for g in range(centers.shape[0])
        )

    def corners(self) -> np.ndarray:
        """[4,3] outer-corner loop (world), counter-clockwise in the gate plane."""
        c = np.asarray(self.center)
        lat = np.asarray(self.lateral) * self.half_w
        ver = np.asarray(self.vertical) * self.half_h
        return np.stack([c - lat - ver, c + lat - ver, c + lat + ver, c - lat + ver])

    def _apply_bind(self, value: Any) -> "Gate":
        if self.index is None:
            return self
        active = int(np.asarray(value).reshape(-1)[0])
        return dataclasses.replace(self, style="accent") if active == self.index else self


@_register
@dataclasses.dataclass(frozen=True)
class Box:
    """Axis-aligned wireframe box: props, safe volumes, obstacles, pads."""

    center: Vec3
    half: Vec3
    style: str = "wire"
    bind: Bind = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec(self.center, 3))
        object.__setattr__(self, "half", _vec(self.half, 3))

    def corners(self) -> np.ndarray:
        """[8,3] corners, x-fastest ordering (edge table in scenepane pairs them)."""
        c, h = np.asarray(self.center), np.asarray(self.half)
        signs = np.array(
            [[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)],
            np.float64,
        )
        return c + signs * h

    def _apply_bind(self, value: Any) -> "Box":
        return dataclasses.replace(self, center=_vec(value, 3))


@_register
@dataclasses.dataclass(frozen=True)
class Marker:
    """Point of interest — a goal, a setpoint, a waypoint. Diamond + optional plumb."""

    center: Vec3 = (0.0, 0.0, 0.0)
    size: float = 0.18  # metres, display half-height
    plumb: bool = True
    style: str = "accent"
    bind: Bind = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec(self.center, 3))

    def _apply_bind(self, value: Any) -> "Marker":
        return dataclasses.replace(self, center=_vec(value, 3))


#: sentinel: the bind PATH did not resolve (an attribute was absent) — distinct
#: from a bind that resolved to a legitimate None (which means "hide me").
_MISSING = object()

#: bind paths already warned about (once per path per process, any frame source)
_WARNED_PATHS: set[str] = set()


def _lookup(frame: Any, path: str) -> Any:
    obj = frame
    for part in path.split("."):
        if not hasattr(obj, part):
            return _MISSING
        obj = getattr(obj, part)
        if obj is None:
            return None
    return obj


def warn_missing_bind(path: str, root: Any) -> None:
    """One warning per bind path per process: a path that does not resolve used to
    hide its primitive with no signal — the wrong-pytree failure mode."""
    if path in _WARNED_PATHS:
        return
    _WARNED_PATHS.add(path)
    warnings.warn(
        f"scene bind {path!r} does not resolve on {type(root).__name__} — the bound "
        "primitive is HIDDEN. A typo, a renamed task-state field, or a raw sticks-mode "
        "SimState.task_state (use env.task_state(state)) all land here.",
        RuntimeWarning,
        stacklevel=3,
    )


def resolve(prim: Any, frame: Any) -> Any | None:
    """
    The primitive to draw this frame: `prim` itself when unbound, else the bind result
    applied — None hides, a dict overrides fields, any other value goes to the
    primitive's own `_apply_bind`. Fleet-leading `[W, ...]` values take the focused
    world's row first. An unresolvable PATH also hides, but warns once per path —
    silence here shipped a six-vanishing-gates bug.
    """
    bind = prim.bind
    if bind is None:
        return prim
    val = bind(frame) if callable(bind) else _lookup(frame, bind)
    if val is _MISSING:
        warn_missing_bind(str(bind), frame)
        return None
    if val is None:
        return None
    if isinstance(val, dict):
        return dataclasses.replace(prim, **val)
    arr = np.asarray(val, np.float64)
    if frame is not None and arr.ndim >= 1 and arr.shape[0] == frame.plant.shape[0]:
        # [W, ...] rows pick the focus row; a bare 1-D vector (e.g. one xyz) stays whole
        # unless the primitive expects per-world scalars (_BIND_SCALAR, e.g. Gate).
        if arr.ndim >= 2 or getattr(prim, "_BIND_SCALAR", False):
            arr = arr[frame.focus]
    apply = getattr(prim, "_apply_bind", None)
    return prim if apply is None else apply(arr)


class Scene:
    """Ordered, flat collection of primitives — the display world the scene pane draws."""

    def __init__(self, *prims: Any) -> None:
        self._prims: list[Any] = list(prims)

    def add(self, *prims: Any) -> "Scene":
        self._prims.extend(prims)
        return self

    def __iter__(self) -> Iterator[Any]:
        return iter(self._prims)

    def __len__(self) -> int:
        return len(self._prims)

    def resolved(self, frame: Any) -> list[Any]:
        """Concrete primitives for this frame (binds applied, hidden ones dropped)."""
        out = []
        for p in self._prims:
            r = resolve(p, frame)
            if r is not None:
                out.append(r)
        return out

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """
        (lo, hi) world bounds of the static geometry — the fixed camera framing.
        Duck-typed so registered user primitives count too: `corners()` wins, then
        `points`, then `center`; Grid contributes its floor extent.
        """
        pts: list[np.ndarray] = []
        for p in self._prims:
            if isinstance(p, Grid):
                hx, hy = p.half
                pts.append(np.array([[-hx, -hy, 0.0], [hx, hy, 0.0]]))
            elif hasattr(p, "corners"):
                pts.append(np.asarray(p.corners(), np.float64).reshape(-1, 3))
            elif hasattr(p, "points"):
                pts.append(np.asarray(p.points, np.float64).reshape(-1, 3))
            elif hasattr(p, "center"):
                pts.append(np.asarray([p.center], np.float64))
        if not pts:
            return np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 3.0])
        allp = np.concatenate(pts, axis=0)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        # give flat scenes headroom so glyphs at altitude stay in frame
        hi[2] = max(hi[2], lo[2] + 2.5)
        return lo, hi

    # -- serde ---------------------------------------------------------------------

    def to_dicts(self) -> list[dict]:
        """JSON-safe primitive dicts; callable binds are live-only and drop here."""
        out = []
        for p in self._prims:
            d: dict[str, Any] = {"type": type(p).__name__.lower()}
            for f in dataclasses.fields(p):
                v = getattr(p, f.name)
                if f.name == "bind" and (v is None or callable(v)):
                    continue
                d[f.name] = v
            out.append(d)
        return out

    @classmethod
    def from_dicts(cls, dicts: list[dict]) -> "Scene":
        """Rebuild from the serde form (also the `task.viz_scene()` hook contract)."""
        prims = []
        for d in dicts:
            d = dict(d)
            kind = d.pop("type")
            prim_cls = _REGISTRY.get(kind)
            if prim_cls is None:
                raise ValueError(f"unknown primitive type {kind!r}; known: {sorted(_REGISTRY)}")
            prims.append(prim_cls(**d))
        return cls(*prims)

    def to_json(self) -> str:
        return json.dumps(self.to_dicts())

    @classmethod
    def from_json(cls, s: str) -> "Scene":
        return cls.from_dicts(json.loads(s))
