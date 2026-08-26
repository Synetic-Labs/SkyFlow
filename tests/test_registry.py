"""
Task registry contract (DESIGN.md §2) — register_task/build_task with the two shipped
tasks registered — plus the env's builder-aware forwarding of the env-owned
spawn_dr_scale and control_hz (§7): forwarded to builders that name them, overridden by
explicit task_kwargs, and never pushed on builders that don't (GateCourseTask).
"""

import pytest

from skyflow import build_task, register_task, tasks
from skyflow.env import DomainRand, SimConfig, SkyFlowEnv
from skyflow.tasks.gate_course import GateCourseTask
from skyflow.tasks.hover import HoverTask


def test_shipped_tasks_are_registered():
    assert isinstance(build_task("hover"), HoverTask)
    assert isinstance(build_task("figure_eight"), GateCourseTask)


def test_build_task_passes_kwargs_through_unmodified():
    task = build_task("hover", goal_hold_s=1.0, control_hz=50.0)
    assert isinstance(task, HoverTask)
    assert task.hold_steps == 50


def test_unknown_task_raises_with_registered_names():
    with pytest.raises(KeyError, match="hover"):
        build_task("wingsuit")


def test_register_refuses_collisions():
    name = "_registry_test_hover"
    register_task(name, HoverTask)
    try:
        assert isinstance(build_task(name), HoverTask)
        with pytest.raises(ValueError, match="already registered"):
            register_task(name, HoverTask)
    finally:
        del tasks._REGISTRY[name]  # keep the module-global registry test-clean


def test_env_forwards_env_owned_kwargs_to_naming_builders():
    env = SkyFlowEnv(
        SimConfig(
            num_envs=2,
            task="hover",
            task_kwargs={"goal_hold_s": 1.0},
            control_hz=50.0,
            physics_hz=1000.0,
            dr=DomainRand(spawn_scale=0.25),
        )
    )
    task = env.task
    assert isinstance(task, HoverTask)
    assert task.hold_steps == 50  # counted at the platform's 50 Hz, not the default
    assert task.spawn_dr_scale == 0.25


def test_explicit_task_kwargs_beat_forwarding():
    env = SkyFlowEnv(
        SimConfig(
            num_envs=2,
            task="hover",
            task_kwargs={"goal_hold_s": 1.0, "control_hz": 200.0},
        )
    )
    task = env.task
    assert isinstance(task, HoverTask)
    assert task.hold_steps == 200


def test_builders_not_naming_the_kwargs_build_untouched():
    env = SkyFlowEnv(SimConfig(num_envs=2, task="figure_eight", dr=DomainRand(spawn_scale=3.0)))
    assert isinstance(env.task, GateCourseTask)
