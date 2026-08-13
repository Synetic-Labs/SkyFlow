"""
Hover task — fly to a setpoint and stay there (DESIGN.md §9).

Worlds spawn resting on a ground pad with jittered XY and chase a goal position drawn
uniformly in a box; the goal resamples every `goal_hold_s` seconds of held time, so one
episode exercises several approach-and-hold segments. Success is proximity
(d < success radius) and does NOT terminate — holding is the point; the task crash is
leaving the safe box, which bounds exploration long before the env's flyaway set (§7).

Reward per control step (shipped defaults in the constructor; no reward code elsewhere):

    w_pos·exp(−3·d) + w_hold·exp(−50·d) − w_vel·|v| − w_rate·|ω| + w_prog·(d_prev² − d²)

The two exponentials pay for coarse approach and precise hold; the progress term
telescopes over a held goal (both endpoints measure against the same goal, so a goal
switch never pays), and the speed/rate penalties damp the equilibrium.

Frames (DESIGN.md §3): observations are world z-up — `rel_pos` = goal − x and `vel` are
world-frame rows, with the body→world rotation matrix flattened alongside, matching the
§9 layout [rel_pos(3), vel(3), rot_matrix(9), last_action(4)] = 19. All outputs are
float32 with the fleet axis leading.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from skyflow.tasks.base import finalize_obs, quat_to_rot
from skyflow.types import Array, ObsSpec, ObsTerm, TaskEval

__all__ = ["HoverTask", "HoverTaskState"]


class HoverTaskState(NamedTuple):
    """Per-world goal state (a pytree carried through SimState.task_state)."""

    goal: Array  # [F,3] f32 world-frame setpoint
    hold: Array  # [F] int32 control steps the current goal has been held
    dist: Array  # [F] f32 latest |goal − x| m (metrics)


class HoverTask:
    """
    Reach-and-hold reference task (DESIGN.md §9). Pure and jit/vmap-safe throughout;
    reward constants live here with the shipped defaults — env.py has no reward code.
    """

    success_terminates = False
    image_shape: tuple[int, int, int] | None = None
    obs_spec = ObsSpec(
        (
            ObsTerm("rel_pos", 3),
            ObsTerm("vel", 3),
            ObsTerm("rot_matrix", 9),
            ObsTerm("last_action", 4),
        )
    )

    def __init__(
        self,
        *,
        spawn_xy_m: float = 1.0,
        spawn_dr_scale: float = 1.0,
        goal_xy_m: float = 1.5,
        goal_z_min_m: float = 0.8,
        goal_z_max_m: float = 2.0,
        goal_hold_s: float = 4.0,
        control_hz: float = 100.0,
        safe_xy_m: float = 4.0,
        safe_z_m: float = 4.0,
        success_radius_m: float = 0.1,
        obs_noise: float = 0.0,
        w_pos: float = 1.0,
        w_hold: float = 0.5,
        w_vel: float = 0.05,
        w_rate: float = 0.01,
        w_prog: float = 1.0,
    ):
        """
        Args:
          spawn_xy_m: pad spawn XY jitter half-width, metres.
          spawn_dr_scale: multiplier on the spawn jitter (SimConfig.spawn_dr_scale —
            the env forwards it when building this task from the registry).
          goal_xy_m / goal_z_min_m / goal_z_max_m: goal box — XY half-width and the
            altitude band, metres.
          goal_hold_s: seconds a goal is held before resampling.
          control_hz: control rate the hold time is counted at (the Task protocol
            carries no clock — like spawn_dr_scale, the env forwards
            SimConfig.control_hz when building this task from the registry).
          safe_xy_m / safe_z_m: safe box — leaving it is the task crash.
          success_radius_m: |goal − x| below this is success (never terminates).
          obs_noise: half-width of additive uniform observation noise; 0 disables.
          w_pos / w_hold / w_vel / w_rate / w_prog: reward weights (module docstring).
        """
        if goal_z_min_m <= 0.0 or goal_z_max_m < goal_z_min_m:
            raise ValueError(
                f"need 0 < goal_z_min_m <= goal_z_max_m, got ({goal_z_min_m}, {goal_z_max_m})"
            )
        if goal_xy_m >= safe_xy_m or goal_z_max_m >= safe_z_m:
            raise ValueError(
                "the goal box must sit strictly inside the safe box: "
                f"xy {goal_xy_m} vs {safe_xy_m}, z {goal_z_max_m} vs {safe_z_m}"
            )
        if spawn_dr_scale * spawn_xy_m >= safe_xy_m:
            raise ValueError(
                f"spawn jitter {spawn_dr_scale * spawn_xy_m} m reaches outside the "
                f"{safe_xy_m} m safe box"
            )
        self.spawn_xy_m = float(spawn_xy_m)
        self.spawn_dr_scale = float(spawn_dr_scale)
        self.goal_xy_m = float(goal_xy_m)
        self.goal_z_min_m = float(goal_z_min_m)
        self.goal_z_max_m = float(goal_z_max_m)
        self.hold_steps = max(1, round(float(goal_hold_s) * float(control_hz)))
        self.safe_xy_m = float(safe_xy_m)
        self.safe_z_m = float(safe_z_m)
        self.success_radius_m = float(success_radius_m)
        self.obs_noise = float(obs_noise)
        self.w_pos = float(w_pos)
        self.w_hold = float(w_hold)
        self.w_vel = float(w_vel)
        self.w_rate = float(w_rate)
        self.w_prog = float(w_prog)

    def _draw_goal(self, key: Array, n: int) -> Array:
        """[n,3] f32 world goals uniform in the goal box."""
        k_xy, k_z = jax.random.split(key)
        xy = jax.random.uniform(k_xy, (n, 2), jnp.float32, -self.goal_xy_m, self.goal_xy_m)
        z = jax.random.uniform(k_z, (n, 1), jnp.float32, self.goal_z_min_m, self.goal_z_max_m)
        return jnp.concatenate([xy, z], axis=-1)

    # -- Task protocol -----------------------------------------------------------------

    def spawn(self, key: Array, n: int, params: Array) -> tuple[Array, HoverTaskState]:
        """Fresh plant rows [n,17] f32 resting on the pad, plus fresh goals.

        Level attitude, zero velocity and body rates, XY jittered by
        spawn_dr_scale·spawn_xy_m, z = 0. Rotors spawn at rest ("motors near idle"):
        the env's post-step clip lifts them to the airframe's idle floor on the first
        substep. `params` is unused — the pad spawn needs no equilibrium solve.
        """
        del params
        k_xy, k_goal = jax.random.split(key)
        xy = self.spawn_dr_scale * self.spawn_xy_m * jax.random.uniform(
            k_xy, (n, 2), jnp.float32, -1.0, 1.0
        )
        pos = jnp.concatenate([xy, jnp.zeros((n, 1))], axis=-1)
        quat = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (n, 1))
        plant = jnp.concatenate(
            [pos, jnp.zeros((n, 3)), quat, jnp.zeros((n, 3)), jnp.zeros((n, 4))], axis=-1
        ).astype(jnp.float32)
        goal = self._draw_goal(k_goal, n)
        state = HoverTaskState(
            goal=goal,
            hold=jnp.zeros((n,), jnp.int32),
            dist=jnp.linalg.norm(goal - pos, axis=-1).astype(jnp.float32),
        )
        return plant, state

    def observe(
        self,
        plant: Array,
        task_state: HoverTaskState,
        imu: tuple[Array, Array],
        last_action: Array,
        key: Array,
        fresh_spawn: bool,
    ) -> tuple[Array, HoverTaskState]:
        """Obs rows [n,19] f32; advances the goal-hold clock and resamples due goals.

        The hold clock ticks once per regular observation and the resample happens
        BEFORE the obs is built, so the policy always sees the goal the next transition
        is rewarded against. `fresh_spawn=True` (spawn-time and auto-reset re-observe)
        neither ticks nor resamples — spawn just drew these goals. `imu` is unused:
        this task observes exact state (IMU packaging stays in sensors.py).
        """
        del imu
        k_goal, k_noise = jax.random.split(key)
        goal, hold = task_state.goal, task_state.hold
        if not fresh_spawn:
            hold = hold + 1
            renew = hold >= self.hold_steps
            goal = jnp.where(renew[:, None], self._draw_goal(k_goal, plant.shape[0]), goal)
            hold = jnp.where(renew, 0, hold)

        rot = quat_to_rot(plant[:, 6:10]).reshape(-1, 9)
        obs = jnp.concatenate([goal - plant[:, 0:3], plant[:, 3:6], rot, last_action], axis=-1)
        if self.obs_noise > 0.0:
            obs = obs + jax.random.uniform(
                k_noise, obs.shape, obs.dtype, -self.obs_noise, self.obs_noise
            )
        return finalize_obs(obs), HoverTaskState(goal=goal, hold=hold, dist=task_state.dist)

    def evaluate(
        self, prev_plant: Array, plant: Array, task_state: HoverTaskState
    ) -> TaskEval:
        """Reward/success/crash on the transition prev_plant → plant (module docstring).

        Both distances measure against the CURRENT goal, so the progress term
        telescopes cleanly across a held goal and a resample never pays a phantom jump.
        """
        goal = task_state.goal
        d_prev = jnp.linalg.norm(goal - prev_plant[:, 0:3], axis=-1)
        d = jnp.linalg.norm(goal - plant[:, 0:3], axis=-1)
        speed = jnp.linalg.norm(plant[:, 3:6], axis=-1)
        rate = jnp.linalg.norm(plant[:, 10:13], axis=-1)
        reward = (
            self.w_pos * jnp.exp(-3.0 * d)
            + self.w_hold * jnp.exp(-50.0 * d)
            - self.w_vel * speed
            - self.w_rate * rate
            + self.w_prog * (d_prev**2 - d**2)
        )
        success = d < self.success_radius_m
        crash = (
            (jnp.abs(plant[:, 0]) > self.safe_xy_m)
            | (jnp.abs(plant[:, 1]) > self.safe_xy_m)
            | (plant[:, 2] > self.safe_z_m)
        )
        d = d.astype(jnp.float32)
        info = {"hover/dist": d, "hover/success": success.astype(jnp.float32)}
        return TaskEval(
            reward=reward.astype(jnp.float32),
            success=success,
            crash=crash,
            info=info,
            task_state=HoverTaskState(goal=goal, hold=task_state.hold, dist=d),
        )

    def metrics(self, task_state: HoverTaskState) -> dict[str, Array]:
        """Goal-tracking diagnostics, all [F] float32."""
        return {
            "hover/dist": task_state.dist,
            "hover/goal_hold": task_state.hold.astype(jnp.float32),
        }
