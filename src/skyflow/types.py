"""
Shared types — the vocabulary every SkyFlow module speaks (DESIGN.md §4).

Frames and layout match SkyFlow-Dynamics exactly (DESIGN.md §3): world right-handed z-up,
body FLU, quaternions wxyz scalar-first Hamilton body→world, SI units with rotor speeds in
rad/s. Every batched array leads with the fleet axis [F, ...]. The env creates float32
leaves only; precision inside the dynamics follows the ambient JAX config.

This module holds structure, not behavior: Task and FirmwareFleet are the seams consuming
repos implement against, and SimState is the single pytree the env scans.
"""

import dataclasses
from typing import Any, NamedTuple, Protocol, TypedDict

import jax

Array = jax.Array

#: Flat plant state in the spec layout: x_W(3), v_W(3), q_wxyz(4), ω_B(3), Ω(4) — [..., 17].
PlantState = Array


class ObsTerm(NamedTuple):
    """One named block of the flat observation vector."""

    name: str
    dim: int


class ObsSpec(tuple[ObsTerm, ...]):
    """
    Ordered observation layout. Concatenation order is tuple order; `dim` is the total
    width and `layout` maps each term name to its slice of the flat obs vector.
    """

    @property
    def dim(self) -> int:
        return sum(t.dim for t in self)

    @property
    def layout(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        offset = 0
        for t in self:
            out[t.name] = slice(offset, offset + t.dim)
            offset += t.dim
        return out


class TaskEval(NamedTuple):
    """Task verdict on one transition (prev_plant → plant); all arrays [F]."""

    reward: Array  # [F] f32
    success: Array  # [F] bool
    crash: Array  # [F] bool — task-specific fatal condition
    info: dict[str, Array]  # scalarizable diagnostics, all [F]
    task_state: Any  # updated opaque task pytree


class StepInfo(TypedDict):
    """
    Info returned by env.step. The fixed keys are the PRE-auto-reset flags and
    observation; the task's `evaluate` info merges in as additional [F] entries.
    """

    terminated: Array  # [F] bool — crash ∨ task crash ∨ (success ∧ success_terminates)
    truncated: Array  # [F] bool — episode-length / stuck cutoff
    final_obs: Array  # [F, obs_dim] f32 — observation before any auto-reset blend


class Task(Protocol):
    """
    A task decides what the vehicle attempts and senses (DESIGN.md §4, §9). SkyFlow ships
    `hover` and `gate_course` as examples; research tasks live in consuming repos and
    register against this protocol. All methods are pure and jit/vmap-safe.
    """

    obs_spec: ObsSpec
    image_shape: tuple[int, int, int] | None  # (H, W, C) when vision obs present, else None
    success_terminates: bool

    def spawn(self, key: Array, n: int, params: Array) -> tuple[Array, Any]:
        """Fresh plant rows [n,17] f32 (spec layout) and a fresh task_state pytree."""
        ...

    def observe(
        self,
        plant: Array,
        task_state: Any,
        imu: tuple[Array, Array],
        last_action: Array,
        key: Array,
        fresh_spawn: bool,
    ) -> tuple[Array, Any]:
        """Obs rows [n, obs_spec.dim] f32 and the (possibly advanced) task_state."""
        ...

    def evaluate(self, prev_plant: Array, plant: Array, task_state: Any) -> TaskEval:
        """Reward/success/crash on the transition prev_plant → plant."""
        ...

    def metrics(self, task_state: Any) -> dict[str, Array]:
        """Scalarizable task diagnostics, all [F]."""
        ...


class FirmwareFleet(Protocol):
    """
    Betaflight-fleet seam for control="sticks" (DESIGN.md §10). Ticked at 1 kHz. Frames
    at this boundary only are NED/FRD: sensor rows f32 [F,7] = gyro_FRD rad/s (3),
    specific force FRD m/s² (3) (level hover ⇒ az = −9.81), baro Pa (1); sticks f32 [F,4]
    AETR in [−1,1]; motors f32 [F,4] in [0,1] QUADX order; armed u8 [F]. `blob` is the
    implementation's opaque device/host handle, `fwstate` its per-world pytree.
    """

    act_dim: int

    def fresh_firmware_state(self) -> tuple[Any, Any]:
        """New (blob, fwstate) for the whole fleet, disarmed."""
        ...

    def fw_step(
        self, blob: Any, fwstate: Any, sticks: Array, sensors: Array
    ) -> tuple[Any, Any, Array, Array]:
        """One 1 kHz firmware tick → (blob, fwstate, motors [F,4] in [0,1], armed u8 [F])."""
        ...

    def reset(self, blob: Any, fwstate: Any, mask: Array) -> tuple[Any, Any]:
        """Re-initialize the worlds selected by mask (u8 [F])."""
        ...

    def close(self) -> None:
        """Release host/device resources; the fleet is unusable afterwards."""
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SimState:
    """
    Full simulator state — one registered pytree, carried through env.step and scanned
    in-jit. All leaves lead with the fleet axis [F, ...] and are float32 unless noted.
    """

    plant: Array  # [F,17] spec layout: x(3) v(3) q_wxyz(4) ω(3) Ω(4) rad/s
    params: Array  # [F,P] per-world randomized flat spec params (pack_params order)
    key: Array  # jax PRNG key (env-owned; split every step)
    wind_vel: Array  # [F,3] OU wind velocity state, world frame (statedot's v_wind)
    act_buf: Array  # [F,D+1,4] transport-delay ring, newest first
    delay_idx: Array  # [F] int32 per-world delay draw
    last_action: Array  # [F,4]
    steps: Array  # [F] int32
    airborne: Array  # [F] bool
    ep_return: Array  # [F]
    ep_len: Array  # [F] int32
    task_state: Any  # opaque task pytree

    def replace(self, **updates: Any) -> "SimState":
        """New SimState with the given leaves swapped (dataclasses.replace)."""
        return dataclasses.replace(self, **updates)
