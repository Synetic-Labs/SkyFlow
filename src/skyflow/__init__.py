"""
SkyFlow — fleet-batched quadrotor simulator in pure JAX.

Physics is generated from the SkyFlow-Dynamics symbolic spec; SkyFlow is the harness:
stepping, disturbances, randomization, sensors, vision, tasks (DESIGN.md §1). The public
surface is the platform (`SkyFlowEnv` + `SimConfig`), the two registries (`register_task`,
`register_airframe`), and the shared types consuming repos implement against (DESIGN.md §2).
"""

from skyflow.env import SimConfig, SkyFlowEnv
from skyflow.params import Airframe, register_airframe
from skyflow.tasks import build_task, register_task
from skyflow.types import (
    Array,
    FirmwareFleet,
    ObsSpec,
    ObsTerm,
    PlantState,
    SimState,
    StepInfo,
    Task,
    TaskEval,
)

__version__ = "0.2.0"

__all__ = [
    "Airframe",
    "Array",
    "FirmwareFleet",
    "ObsSpec",
    "ObsTerm",
    "PlantState",
    "SimConfig",
    "SimState",
    "SkyFlowEnv",
    "StepInfo",
    "Task",
    "TaskEval",
    "__version__",
    "build_task",
    "register_airframe",
    "register_task",
]
