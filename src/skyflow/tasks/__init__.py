"""SkyFlow tasks — objectives that plug into the shared ``SkyFlowEnv`` platform.

Select with ``env.task=<name>``. The platform (plant + DR + disturbances + latency +
auto-reset) is task-agnostic; each task owns its spawn, observation, reward,
terminal/success events and metrics. See :mod:`.base` for the contract.

``hover`` ships with the package. To fly your own objective, implement the
:class:`Task` protocol and register it — the env reaches every task through the
protocol, so a registered task is a first-class citizen:

    from skyflow.tasks import register_task
    register_task("my_task", MyTask)
"""

from __future__ import annotations

from .base import Task, TaskEval, build_task, register_task

__all__ = ["Task", "TaskEval", "build_task", "register_task"]
