"""
Task registry (DESIGN.md §2): the name → builder map behind `SimConfig.task`.

SkyFlow ships two reference tasks — `hover` and `figure_eight` — registered here at
import; research tasks live in consuming repos and register against the same protocol
(`skyflow.types.Task`), which makes them first-class: the env reaches every task through
`build_task`, and nothing special-cases the built-ins. Builders receive
`SimConfig.task_kwargs` unmodified; on top of those the env forwards the env-owned
quantities `spawn_dr_scale` and `control_hz` to builders that name them (the Task
protocol carries no clock — see `SkyFlowEnv._build_task`).
"""

from collections.abc import Callable

from skyflow.tasks.gate_course import GateCourseTask
from skyflow.tasks.hover import HoverTask
from skyflow.types import Task

__all__ = [
    "GateCourseTask",
    "HoverTask",
    "build_task",
    "get_builder",
    "register_task",
]

_REGISTRY: dict[str, Callable[..., Task]] = {}


def register_task(name: str, builder: Callable[..., Task]) -> None:
    """
    Add a task builder — usually the task class itself — under `name`. Refuses name
    collisions, matching `params.register_airframe`: silently shadowing a registered
    task is never what anyone wants; a variant registers under its own name.
    """
    if name in _REGISTRY:
        raise ValueError(f"task {name!r} is already registered")
    _REGISTRY[name] = builder


def get_builder(name: str) -> Callable[..., Task]:
    """
    The registered builder itself, unbuilt — for callers that inspect before building
    (the env reads its signature to decide which env-owned kwargs to forward).
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(_REGISTRY)}") from None


def build_task(name: str, **kwargs) -> Task:
    """Build a registered task. kwargs pass through to the builder unmodified."""
    return get_builder(name)(**kwargs)


register_task("hover", HoverTask)
register_task("figure_eight", GateCourseTask)
