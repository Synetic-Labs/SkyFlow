"""
Free-flying playground, registered as `freefly` — no objective, no task crash.

A large open field with a loose ring of racing gates to fly through for fun: straight
gates, an angled gate, a split-S double gate (two openings stacked in one frame: in
through the top, half-loop down, back out through the bottom the opposite way), a high
tower gate, a horizontal dive gate and a three-gate tunnel. Gates are display and
counting geometry only — the sim has no gate collision physics, so a frame hit is
counted, never fatal. The task never crashes or succeeds; an episode ends only by the
env's own rules (flyaway bounds, time limit, ground crash).

For hand flying, pair it with the env settings in `FREEFLY_ENV`: ground crash off,
no practical time limit, bounds that match the field. `examples/fly_teleop.py
--task freefly` applies them.

Frames: world z-up, body FLU, quaternion wxyz Hamilton body→world (DESIGN.md §3).

Observations (state mode): [pos(3) world z-up, vel_body(3), rot_matrix(9),
last_action(4)] = 19. Reward: +1 per clean gate pass in either direction, so the
viewer's EP REWARD bars count gates.
"""

import math
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from skyflow.tasks.base import finalize_obs, quat_to_rot, world_to_body
from skyflow.types import Array, ObsSpec, ObsTerm, TaskEval
from skyflow.vision.gates import GateSet, classify_crossings

if TYPE_CHECKING:
    from skyflow.vision.camera import CameraModel

__all__ = ["FREEFLY_ENV", "FreeflyTask", "FreeflyTaskState", "playground"]

#: Field half-size, metres: the grid, the FPV floor and the suggested flyaway bounds.
FIELD_HALF_M = 60.0

#: SimConfig overrides for a hand-flying session (`dataclasses.replace(cfg, **FREEFLY_ENV)`
#: or SimConfig(**FREEFLY_ENV, ...)): touchdowns never reset, no practical time limit,
#: flyaway bounds at the field edge. The speed/rate guards stay as divergence guards.
FREEFLY_ENV: dict = {
    "ground_crash": False,
    "max_episode_steps": 2**30,
    "stuck_steps": 2**30,
    "bounds_xy_m": FIELD_HALF_M,
    "bounds_z_m": 40.0,
    "max_speed_mps": 60.0,
}

_DEG = math.pi / 180.0
#: Gate centre height for a gate on the ground: bottom bar 0.6 m up.
_Z0 = 1.5


def playground() -> GateSet:
    """
    The freefly gate layout: 11 gates of one design (1.5 m square opening, 0.15 m bar —
    the common 5-inch racing size), world z-up, yaw about +z (0 = +x). The start pad is
    the origin, facing +x; gate 0 is straight ahead.
    """
    inner = 0.75
    bar = 0.15
    split_dz = 2.0 * (inner + bar)  # stack the two openings frame on frame
    gates = [
        # (centre, yaw°, pitch°)
        ((12.0, 0.0, _Z0), 0.0, 0.0),  # 0  straight ahead of the pad
        ((28.0, 8.0, _Z0), 30.0, 0.0),  # 1  angled gate
        ((44.0, 16.0, _Z0 + split_dz), 0.0, 0.0),  # 2  split-S: top opening, in along +x
        ((44.0, 16.0, _Z0), 180.0, 0.0),  # 3  split-S: bottom opening, out along -x
        ((30.0, 32.0, 6.0), 180.0, 0.0),  # 4  tower gate, 6 m up
        ((12.0, 32.0, 8.0), 180.0, -90.0),  # 5  dive gate: horizontal, flown downward
        ((-8.0, 22.0, _Z0), -120.0, 0.0),  # 6  angled gate
        ((-14.0, -14.0, _Z0), 0.0, 0.0),  # 7  tunnel 1/3
        ((-7.0, -14.0, _Z0), 0.0, 0.0),  # 8  tunnel 2/3
        ((0.0, -14.0, _Z0), 0.0, 0.0),  # 9  tunnel 3/3
        ((20.0, -24.0, 3.0), 45.0, 0.0),  # 10 raised angled gate
    ]
    return GateSet.build(
        centers=[g[0] for g in gates],
        yaws=[g[1] * _DEG for g in gates],
        pitches=[g[2] * _DEG for g in gates],
        inner_half=(inner, inner),
        frame_width=bar,
    )


class FreeflyTaskState(NamedTuple):
    """Per-world fun counters (a pytree carried through SimState.task_state)."""

    passes: Array  # [F] int32 — clean gate passes this episode, either direction
    hits: Array  # [F] int32 — gate frame touches this episode (counted, not fatal)


class FreeflyTask:
    """Open-field free flight among gates: never crashes, never succeeds."""

    success_terminates = False
    image_shape: tuple[int, int, int] | None = None

    def __init__(self, gates: GateSet | None = None, *, body_radius_m: float = 0.1):
        """
        Args:
          gates: the gate layout; None builds `playground()`.
          body_radius_m: drone-as-sphere radius for pass/hit counting
            (`classify_crossings`): a pass needs this much clearance from every bar.
        """
        self.gates = gates if gates is not None else playground()
        self.body_radius_m = float(body_radius_m)
        self.obs_spec = ObsSpec(
            (
                ObsTerm("pos", 3, "m world z-up"),
                ObsTerm("vel_body", 3, "m/s body FLU"),
                ObsTerm("rot_matrix", 9, "R body FLU -> world z-up row-major"),
                ObsTerm("last_action", 4, "[-1,1]"),
            )
        )

    # -- Task protocol -------------------------------------------------------------

    def spawn(self, key: Array, n: int, params: Array) -> tuple[Array, FreeflyTaskState]:
        """Every world rests on the start pad at the origin, level, facing +x (gate 0).
        No randomization: a free-flying session starts the same way every time."""
        del key, params
        quat = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (n, 1))
        plant = jnp.concatenate(
            [jnp.zeros((n, 6)), quat, jnp.zeros((n, 7))], axis=-1
        ).astype(jnp.float32)
        zeros = jnp.zeros((n,), jnp.int32)
        return plant, FreeflyTaskState(passes=zeros, hits=zeros)

    def observe(
        self,
        plant: Array,
        task_state: FreeflyTaskState,
        imu: tuple[Array, Array],
        last_action: Array,
        key: Array,
        fresh_spawn: bool,
    ) -> tuple[Array, FreeflyTaskState]:
        """Obs rows [n, 19] f32; task_state passes through unchanged."""
        del imu, key, fresh_spawn
        n = plant.shape[0]
        quat = plant[:, 6:10]
        obs = jnp.concatenate(
            [
                plant[:, 0:3],
                world_to_body(quat, plant[:, 3:6]),
                quat_to_rot(quat).reshape(n, 9),
                last_action,
            ],
            axis=-1,
        )
        return finalize_obs(obs), task_state

    def evaluate(
        self, prev_plant: Array, plant: Array, task_state: FreeflyTaskState
    ) -> TaskEval:
        """Count gate passes (reward +1 each) and frame hits. Never a crash or success."""
        fwd, bwd, hit = classify_crossings(
            prev_plant[:, 0:3], plant[:, 0:3], self.gates, self.body_radius_m
        )
        passed = jnp.sum((fwd | bwd).astype(jnp.int32), axis=-1)
        hits = jnp.sum(hit.astype(jnp.int32), axis=-1)
        never = jnp.zeros(plant.shape[0], bool)
        return TaskEval(
            reward=passed.astype(jnp.float32),
            success=never,
            crash=never,
            info={
                "gate_passed": passed.astype(jnp.float32),
                "gate_hit": hits.astype(jnp.float32),
            },
            task_state=FreeflyTaskState(
                passes=task_state.passes + passed, hits=task_state.hits + hits
            ),
        )

    def metrics(self, task_state: FreeflyTaskState) -> dict[str, Array]:
        """Fun counters, all [F] float32."""
        return {
            "freefly/passes": task_state.passes.astype(jnp.float32),
            "freefly/hits": task_state.hits.astype(jnp.float32),
        }

    @property
    def camera(self) -> "CameraModel | None":
        """No sensor camera (state obs); the viewer's pilot cam uses its default lens."""
        return None

    def viz_scene(self) -> list[dict]:
        """Viewer scene: the field grid, every gate, and the start pad."""
        import numpy as np  # host-side hook: called at viewer build time, never in-jit

        centers = np.asarray(self.gates.centers_world, np.float64)
        laterals = np.asarray(self.gates.laterals_world, np.float64)
        verticals = np.asarray(self.gates.verticals_world, np.float64)
        outer = np.asarray(self.gates.outer_half, np.float64)
        scene: list[dict] = [{"type": "grid", "half": (FIELD_HALF_M, FIELD_HALF_M), "step": 5.0}]
        scene += [
            {
                "type": "gate",
                "center": tuple(centers[g]),
                "lateral": tuple(laterals[g]),
                "vertical": tuple(verticals[g]),
                "half_w": float(outer[g, 0]),
                "half_h": float(outer[g, 1]),
                "index": g,
            }
            for g in range(len(self.gates))
        ]
        scene.append(
            {"type": "marker", "center": (0.0, 0.0, 0.0), "style": "bright", "size": 0.3,
             "plumb": False}
        )
        return scene
