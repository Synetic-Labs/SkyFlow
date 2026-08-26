"""
Shared types — the vocabulary every SkyFlow module speaks (DESIGN.md §4).

Frames and layout match SkyFlow-Dynamics exactly (DESIGN.md §3): world right-handed z-up,
body FLU, quaternions wxyz scalar-first Hamilton body→world, SI units with rotor speeds in
rad/s. Every batched array leads with the fleet axis [F, ...]. The env creates float32
leaves only; precision inside the dynamics follows the ambient JAX config.

This module holds structure, not behavior: Task and FirmwareFleet are the seams
downstream projects implement against, and SimState is the single pytree the env scans.
"""

import dataclasses
from typing import Any, NamedTuple, Protocol, Self, TypedDict

import jax

Array = jax.Array

#: Flat plant state in the spec layout: x_W(3), v_W(3), q_wxyz(4), ω_B(3), Ω(4) — [..., 17].
PlantState = Array


class ObsTerm(NamedTuple):
    """One named block of the flat observation vector.

    ``units`` declares what the numbers ARE — units and frame in one short canonical
    string ("m world z-up", "m/s body FLU", "[-1,1]"). It defaults to "" so every
    existing 2-field call site and consumer stays valid, but tasks SHOULD declare it:
    downstream training contracts hash the string to catch silent re-scales and frame
    changes at an unchanged width, so once declared, treat it as part of the layout —
    a units change is a layout change.
    """

    name: str
    dim: int
    units: str = ""
    # True for an image block (a rendered mask): env-side corruption that assumes
    # metric numbers (dr.obs_noise) must skip it. An explicit flag, not a units
    # string convention — a task that forgets units still gets the right behavior.
    image: bool = False


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


class DRState(NamedTuple):
    """
    Per-world trait draws of the DomainRand block (DESIGN.md §7): quantities drawn once
    per episode and constant within it, redrawn for done worlds at the auto-reset
    respawn. New per-episode traits are added here, never as new SimState leaves.
    """

    wind_mean: Array  # [F,3] f32 steady wind velocity, world frame (z component is 0)
    imu_bias: Array  # [F,6] f32 additive IMU bias: accel(3) m/s², gyro(3) rad/s
    w_max: Array  # [F] f32 per-world rotor-speed ceiling, rad/s (battery-sag trait)
    est_bias: Array  # [F,12] f32 estimator-error bias trait: pos(3) m, vel(3) m/s,
    # att rotation-vector(3) rad, rate(3) rad/s (errors.py channel groups)


class TaskEval(NamedTuple):
    """Task verdict on one transition (prev_plant → plant); all arrays [F]."""

    reward: Array  # [F] f32
    success: Array  # [F] bool
    crash: Array  # [F] bool — task-specific fatal condition
    info: dict[str, Array]  # scalarizable diagnostics, all [F]
    task_state: Any  # updated opaque task pytree


class StepInfo(TypedDict):
    """
    Info returned by env.step. The fixed keys are the PRE-auto-reset flags, observation
    and episode bookkeeping; the task's `evaluate` info merges in as additional [F]
    entries (open by design — TypedDict cannot express the extra keys, so the env casts
    at the merge site).
    """

    terminated: Array  # [F] bool — crash or task crash or (success and success_terminates)
    truncated: Array  # [F] bool — episode-length / stuck cutoff
    final_obs: Array  # [F, obs_dim] f32 — observation before any auto-reset blend
    poke_active: Array  # [F] bool — this step's poke Bernoulli draw
    ep_return: Array  # [F] f32 — pre-reset episode return, valid on done rows
    ep_len: Array  # [F] int32 — pre-reset episode length, valid on done rows


class Task(Protocol):
    """
    A task decides what the vehicle attempts and senses (DESIGN.md §4, §9). SkyFlow ships
    `hover` and `figure_eight` as examples; research tasks live in downstream projects
    and register against this protocol. All methods are pure and jit/vmap-safe.
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
        """Obs rows [n, obs_spec.dim] f32 and the (possibly advanced) task_state.

        `plant` is the ESTIMATOR's state (corrupted under dr.obs_error). Tasks with
        `image_shape` set additionally receive the keyword `true_plant` (the real
        pose) and must render their image from it — a camera images from where the
        vehicle really is. State-only tasks never see the keyword.
        """
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
    specific force FRD m/s² (3) (level hover ⇒ az = -9.81), baro Pa (1); sticks f32 [F,4]
    AETR in [-1,1]; motors f32 [F,4] in [0,1] QUADX order; armed u8 [F]. `blob` is the
    implementation's opaque device/host handle, `fwstate` its per-world pytree.
    """

    act_dim: int

    def fresh_firmware_state(self) -> tuple[Any, Any]:
        """New (blob, fwstate) for the whole fleet, from the ARMED-ON-GROUND snapshot.

        Arming lifecycle (THE normative statement — implementations and docs defer
        here): instances arm during construction (settle → arm → snapshot), so
        `armed` is truthy from the first tick after any fresh state or `reset`.
        A mid-episode disarm (failsafe, runaway-takeoff) persists until the next
        reset restores the snapshot; Betaflight re-arms only on LOW throttle, so
        an open-loop feeder that never lowers throttle cannot re-arm a world.
        """
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


class FirmwareCarry(NamedTuple):
    """
    Sticks-mode `SimState.task_carry` wrapper: the task's own pytree plus the
    value-threaded firmware pair of `FirmwareFleet`. The env wraps/unwraps it around
    every task call, so tasks never see it; motors mode stores the task pytree bare.
    (SimState has no firmware slot, so the pair rides in the one opaque slot the env
    owns end to end.) Read task fields through ``env.task_state(state)`` — it unwraps
    this carry in sticks mode and is the identity in motors mode.
    """

    task: Any
    blob: Any
    fwstate: Any


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
    wind_vel: Array  # [F,3] OU gust velocity state, world frame, zero-mean deviation
    dr_state: "DRState"  # per-episode trait draws (steady wind, IMU bias)
    act_buf: Array  # [F,D+1,4] transport-delay ring, newest first
    delay_idx: Array  # [F] int32 per-world delay draw
    cmd_prev: Array  # [F,4] the last APPLIED (post-delay/drop) command — what the
    # link holds when a packet drops (dr.cmd_drop_prob)
    last_action: Array  # [F,4]
    # -- estimator-error process state (errors.py; inert zeros when obs_error off) ----
    est_ou: Array  # [F,12] f32 OU drift state, errors.py channel groups
    est_hold: Array  # [F] int32 dropout hold steps remaining (0 = tracking)
    est_held: Array  # [F,17] f32 the estimate emitted during a dropout hold
    steps: Array  # [F] int32
    airborne: Array  # [F] bool
    armed: Array  # [F] bool — sticks: the firmware's arm flag after the last substep
    # (a failsafe/runaway disarm shows here and in metrics armed_frac); motors: all True
    ep_return: Array  # [F]
    ep_len: Array  # [F] int32
    # -- §7 step 10 episode-bookkeeping EMAs (0-d f32, fleet-global by design): updated
    # from the pre-reset done rows each step, decayed per completed episode, start at 0.
    crash_frac: Array  # 0-d f32 — EMA fraction of completed episodes ending in a crash
    success_frac: Array  # 0-d f32 — EMA fraction whose final step evaluated success
    trunc_frac: Array  # 0-d f32 — EMA fraction ending by pure truncation (no termination)
    ep_return_ema: Array  # 0-d f32 — EMA of completed-episode return
    ep_len_ema: Array  # 0-d f32 — EMA of completed-episode length, control steps
    # opaque task pytree (motors) or FirmwareCarry (sticks) — read task fields
    # through env.task_state(state), never off this field directly
    task_carry: Any

    @property
    def task_state(self) -> Any:
        """The task's own pytree — motors mode only.

        In sticks mode `task_carry` holds the firmware carry, and a raw read of a
        task field off it compiles fine but binds to the wrong pytree — the shipped
        vanishing-gates bug. This property therefore RAISES on a carry instead of
        returning it: read through ``env.task_state(state)`` (unwraps in both
        modes), or take the carry itself from ``state.task_carry``.
        """
        if isinstance(self.task_carry, FirmwareCarry):
            raise TypeError(
                "SimState.task_state is the firmware carry in sticks mode — read "
                "the task pytree through env.task_state(state), or the raw carry "
                "through state.task_carry"
            )
        return self.task_carry

    def replace(self, **updates: Any) -> Self:
        """New SimState with the given leaves swapped (dataclasses.replace)."""
        return dataclasses.replace(self, **updates)
