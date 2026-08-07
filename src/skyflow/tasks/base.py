"""Task seam for the SkyFlow env — the objective a policy is trained toward.

``SkyFlowEnv`` (``..env``) owns the flight *platform*: the Betaflight firmware
fleet, the SkyDreamer analytic plant, domain randomization, in-flight
disturbances, transport latency, the fused ``lax.scan`` rollout, the generic
crash set (disarm / flyaway / ground-collision) and the in-jit auto-reset. A
:class:`Task` owns the *objective*: where the drone spawns, what it observes,
the reward, the task-specific terminal/success events, and the diagnostic
metrics. One platform, many tasks — select with ``env.task=<name>`` (see
:func:`build_task`), exactly as the ``crazyflow`` env selects ``hover|gate``.

The contract is deliberately small. The env calls, per policy step:

* :meth:`Task.spawn`    — sample initial plant states (also on auto-reset),
* :meth:`Task.init`     — the task's carried sub-state (mask history, setpoint…),
* :meth:`Task.observe`  — build the obs vector, advancing any frame history,
* :meth:`Task.evaluate` — reward + per-step success/task-crash events,
* :meth:`Task.scalar_metrics` / :meth:`Task.fpv` — diagnostics / viz.

The task's carried state is an opaque pytree (``state.task``); the env blends it
on auto-reset with :func:`skyflow.env._tree_where`, so the
env never needs to know its shape. Everything the task needs from the platform
(control frequency, the action-history width, the frame-stack depth) is passed
in at construction, so the task stays a plain static object closed over by the
jitted env methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from .. import plant
from ..obs import ObsSpec

# The trainer applies MASKED running-normalization to the vision obs: the
# measurement tail (gyro/accel/motors/action) is standardized, the [0,1] mask
# block is left raw (per-pixel standardizing a mask is non-stationary,
# CNN-hostile). This pre-scale keeps the obs ~unit even if running-norm is
# disabled: mask [0,1], motors/action [-1,1], body rates → ~[-1,1] via
# OBS_RATE_SCALE (rad/s), and body specific force → ~[-1,1] via OBS_ACCEL_SCALE
# (m/s²; ~g at rest → ~1.0, aggressive flight to a few g).
OBS_RATE_SCALE = 10.0
OBS_ACCEL_SCALE = 10.0
OBS_CLIP = 100.0


class TaskEval(NamedTuple):
    """Per-step task outcome. ``reward`` is the raw reward *before* the env's
    "zero the reward on any terminal" rule; ``success``/``task_crash`` are the
    task's own per-agent events (the env folds them into ``finished``/``crashed``
    — whether ``success`` ends the episode is :attr:`Task.success_terminates`).
    ``info`` carries any extra per-agent flags for the step ``info`` dict.

    ``task_state`` lets ``evaluate`` hand back an UPDATED carried state when the
    reward computation itself advances it — e.g. the gate task advancing the
    per-world active-gate index on a pass (``evaluate`` is the only task method
    given ``prev_pos``, so the crossing that drives the advance is detected
    there). The env adopts it before the post-step ``observe``, so the next
    obs reflects the advance; ``None`` leaves the carried state untouched."""

    reward: jax.Array        # [F] float32
    success: jax.Array       # [F] bool — task objective met this step
    task_crash: jax.Array    # [F] bool — task-specific terminal (e.g. gate frame hit)
    info: dict[str, jax.Array]
    task_state: Any = None   # updated carried state, or None to leave it unchanged


@runtime_checkable
class Task(Protocol):
    """The objective plugged into :class:`~skyflow.env.SkyFlowEnv`."""

    # -- static obs contract (read by the trainer) --
    vision: bool
    OBS_LAYOUT: ObsSpec
    obs_dim: int
    act_dim: int
    image_shape: tuple[int, int, int] | None   # (H, W, C=frame-stack); None in state mode
    # privileged (fully-observed) state — the vision=False actor obs AND the
    # asymmetric critic's input. ``priv_layout`` is the ordered term spec.
    priv_layout: ObsSpec
    priv_dim: int
    # names for the [task_crash, success] EMA slots in the env's term_ema metric
    term_labels: tuple[str, str]
    # does meeting the objective end the episode? gate: True (a pass); hover: False (station-keep)
    success_terminates: bool

    def spawn(self, key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
        """Sample ``n`` initial plant states. Returns (plant_state [n,17], pos_ned [n,3])."""
        ...

    def init(self, key: jax.Array, n: int, plant_state: jax.Array,
             pos_ned: jax.Array) -> Any:
        """The task's carried sub-state at (re)spawn (an opaque pytree)."""
        ...

    def observe(self, plant_state: jax.Array, task_state: Any, act_buf: jax.Array,
                key: jax.Array, params: jax.Array, *,
                fresh_spawn: bool = False,
                substeps: jax.Array | None = None) -> tuple[jax.Array, Any]:
        """Build the obs vector; return (obs [F,obs_dim], task_state') with any
        frame history advanced for the next step. ``params`` is the per-world DR'd
        plant-coefficient row [F,35] — a task needs it to synthesise IMU signals
        (e.g. the accelerometer specific force via ``plant.specific_force_body``).

        ``fresh_spawn`` marks the RE-OBSERVE call in ``jax_step``'s auto-reset, whose
        result is kept only for worlds that just respawned. A task may use it to skip
        per-step integrators that have nothing to integrate at t=0 — see
        ``GateTask.observe``, where it is worth ~half the filter block's cost. Ignoring
        it is always correct; acting on it must ONLY change what a reset world sees.

        ``substeps`` is ``[F, decimation, 17]`` — the plant's INTERMEDIATE 1 kHz states from
        this control step's rollout — supplied only when the task declared
        ``substep_imu_slots > 0``, and ``None`` on ``jax_reset`` and the ``fresh_spawn`` call
        (where no rollout produced them, or where they belong to the pre-reset trajectory).
        A task that models a sensor faster than the control loop reads them; one that does
        not never sees them, and the rollout scan never stacks them.
        """
        ...

    def privileged_state(self, plant_state: jax.Array, task_state: Any) -> jax.Array:
        """Fully-observed ground-truth state [F, priv_dim] (order = ``priv_layout``)
        — the state-mode actor obs and the asymmetric critic's input. Raw; the env
        applies the same clip as the actor obs."""
        ...

    def evaluate(self, prev_pos_ned: jax.Array, plant_state: jax.Array,
                 task_state: Any) -> TaskEval:
        """Reward + per-step success / task-crash events for the ``prev→cur`` step."""
        ...

    def scalar_metrics(self, plant_state: jax.Array, task_state: Any,
                       ep_reach: jax.Array | None = None) -> dict[str, jax.Array]:
        """Scalar diagnostics (means over the fleet). ``ep_reach`` (optional) is the
        env's per-world latch of gates-cleared in each world's last completed episode,
        for per-episode gate metrics; tasks that don't use it ignore it."""
        ...

    def fpv(self, plant_state: jax.Array, task_state: Any, n: int) -> jax.Array:
        """A [n,H,W] first-person render for the viz clip. Vision tasks only;
        state tasks return a tiny zeros array (the env gates the viz on
        ``vision``)."""
        ...


# -- shared observation helpers (used by every task's ``observe``) ------------

def world_to_body(quat_wxyz: jax.Array, v: jax.Array) -> jax.Array:
    """R(q)^T v — rotate a world vector into the body frame, batched [F,3]."""
    R = plant.rot_matrix(quat_wxyz)                     # body->world [F,3,3]
    return jnp.einsum("fji,fj->fi", R, v)


def orientation(quat_ned: jax.Array) -> jax.Array:
    """World axes expressed in the body frame (R^T flattened to [F, 9])."""
    f = quat_ned.shape[0]
    ex = world_to_body(quat_ned, jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (f, 3)))
    ey = world_to_body(quat_ned, jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), (f, 3)))
    ez = world_to_body(quat_ned, jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), (f, 3)))
    return jnp.concatenate([ex, ey, ez], axis=1)


def finalize_obs(obs: jax.Array) -> jax.Array:
    """NaN-scrub, cast to f32 and clip to the obs rail — the last step of every
    task's ``observe`` so the policy never sees a NaN or an out-of-range spike."""
    return jnp.clip(jnp.nan_to_num(obs.astype(jnp.float32), nan=0.0), -OBS_CLIP, OBS_CLIP)


# Task registry. Built-in tasks self-register below; downstream projects add their own
# with :func:`register_task`, which is the supported way to fly a task SkyFlow does not
# ship (a research reward, a course format, a filter-in-the-loop observation) without
# forking the env. The env only ever reaches a task through the :class:`Task` protocol,
# so a registered task is a first-class citizen — nothing here special-cases the built-ins.
_TASKS: dict[str, Callable[..., Task]] = {}


def register_task(name: str, factory: Callable[..., Task]) -> None:
    """Register ``factory`` under ``name`` so ``env.task=<name>`` can build it.

    ``factory`` is called with the task keyword arguments assembled by
    :func:`~skyflow.make.make_skyflow` and must return a :class:`Task`. Re-registering
    a name replaces it, which is what makes overriding a built-in possible.
    """
    _TASKS[name] = factory


def _hover_task(**kw: Any) -> Task:
    from .hover import HoverTask

    return HoverTask(**kw)


register_task("hover", _hover_task)


def build_task(name: str, **kw: Any) -> Task:
    """Construct the task selected by ``env.task``. Platform-derived scalars
    (``control_freq``, ``act_hist``, ``stack``) and the shared camera/obs knobs
    are forwarded by :func:`~skyflow.make.make_skyflow`."""
    try:
        factory = _TASKS[name]
    except KeyError:
        known = ", ".join(repr(k) for k in sorted(_TASKS)) or "none registered"
        raise ValueError(
            f"unknown skyflow task {name!r} (registered: {known}). Register a custom "
            f"task with skyflow.tasks.register_task({name!r}, factory).") from None
    return factory(**kw)
