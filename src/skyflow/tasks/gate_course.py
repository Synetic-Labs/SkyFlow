"""
Gate-course task, registered as `figure_eight` — ordered racing through a GateSet course
(DESIGN.md §9).

The course is a `skyflow.vision.gates.GateSet` (default: the shipped figure-eight
lemniscate builder). Each world chases one ACTIVE gate at a time: reward is distance
progress toward the active gate's pre-gate point, plus a centering-weighted pass credit
when `classify_crossings` reports a clean forward transit, minus a body-rate penalty.
Touching any gate's frame solid (frame hit) or carrying the active gate's forward plane
crossing outside its opening (miss — wide fly-arounds included) is a task crash; passing
the last gate is success and terminates the episode.

The reward shape follows SkyDreamer (Diermayr et al. 2025, arXiv 2510.14783),
simplified. The centering measure is SkyDreamer's Chebyshev miss at the centre-plane
crossing point — here normalized per axis so rectangular openings score the same as
square ones — giving centering ∈ (0, 1] on every clean pass. The depth ratchet,
back-half cash-out, and slow-speed gain of the full SkyDreamer shape are deliberately
out of the shipped reward (DESIGN.md §9).

Frames: world z-up, body FLU, quaternion wxyz Hamilton body→world (DESIGN.md §3). The
vision modules convert to their internal NED at their own public boundaries, so every
array crossing this module is z-up FLU.

Observations (state mode, gate blocks and velocity in body FLU): [gate_rel(3),
gate_normal(3), next_gate_rel(3), vel_body(3), rot_matrix(9), last_action(4)] = 25.
Vision mode replaces the three gate blocks with the rendered coverage mask [H, W, 1]
(flattened, leading the obs vector) and keeps the same proprio tail. All outputs are
float32 with the fleet axis leading.
"""

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from skyflow.tasks.base import finalize_obs, quat_to_rot
from skyflow.types import Array, ObsSpec, ObsTerm, TaskEval
from skyflow.vision.gates import (
    GateSet,
    classify_crossings,
    crossing_offsets,
    figure_eight,
)

if TYPE_CHECKING:
    from skyflow.vision.camera import CameraModel

__all__ = ["GateCourseTask", "GateTaskState"]

_UP = (0.0, 0.0, 1.0)  # world z-up; altitude jitter uses this, not a gate's tilted vertical


class GateTaskState(NamedTuple):
    """Per-world course progress (a pytree carried through SimState.task_state)."""

    active_gate: Array  # [F] int32 — index of the gate to pass next (clips at the last)
    passes: Array  # [F] int32 — clean active-gate passes this episode (metric)


def _world_to_body(rot: Array, vec: Array) -> Array:
    """Rotate world rows [F,3] into body FLU with R(q)ᵀ: (Rᵀ v)_b = Σ_a R_ab v_a."""
    return jnp.einsum("fab,fa->fb", rot, vec)


class GateCourseTask:
    """Ordered gate racing over a shared course, state or vision observations.

    One `GateSet` serves the whole fleet (worlds are physically independent and the
    reward is gate-relative, so shared absolute placement costs nothing — and lets the
    analytic renderer serve every world in vision mode). Both obs variants share this
    class behind the `vision` flag; `image_shape` is (H, W, 1) in vision mode, else None.
    Reward constants live in the constructor with the shipped defaults (DESIGN.md §9);
    no reward code exists anywhere else.
    """

    success_terminates = True

    def __init__(
        self,
        gates: GateSet | None = None,
        *,
        vision: bool = False,
        camera: "CameraModel | None" = None,
        body_radius_m: float = 0.0,
        w_prog: float = 1.0,
        w_gate: float = 10.0,
        w_rate: float = 0.01,
        pre_gate_offset_m: float = 0.5,
        spawn_mode: str = "podium",
        podium_pos_m: tuple[float, float] | None = None,
        spawn_dist_m: float = 1.5,
        podium_height_m: float = 0.0,
        spawn_lateral_m: float = 0.4,
        spawn_alt_jitter_m: float = 0.3,
        spawn_yaw_jitter_rad: float = 0.3,
    ):
        """
        Args:
          gates: course geometry. None builds the shipped default figure-eight
            (`figure_eight(3)`: the nav-jax FigureEight map — six gates, 20 x 6 m,
            1.5 m altitude).
          vision: observe the rendered gate mask instead of privileged gate geometry.
          camera: vision-mode CameraModel; None takes the renderer's nominal camera.
          body_radius_m: drone-as-sphere radius for `classify_crossings` — a pass needs
            this much clearance from every bar, grazing within it is a frame hit.
          w_prog: reward per metre of progress toward the active pre-gate point.
          w_gate: pass credit scale; a pass pays w_gate·centering, centering ∈ (0, 1].
          w_rate: penalty per rad/s of body-rate magnitude, each control step.
          pre_gate_offset_m: pre-gate point distance in front of the plane (-normal
            side). Progress pulls to the approach side only; the pass credit, not the
            progress term, pays for committing through the opening.
          spawn_mode: "podium" — every world starts on the podium pad at podium
            height, facing gate 0; "spread" — each world starts behind a uniformly
            drawn gate at its altitude (curriculum knob: seeds every course segment
            from step one).
          podium_pos_m: podium pad world (x, y). None pads `spawn_dist_m` behind
            gate 0 — except on the SHIPPED default course, which pads the centre of
            the gate cluster opposite gate 0, the canonical figure-eight start
            (DESIGN.md §9).
          spawn_dist_m / podium_height_m / spawn_lateral_m / spawn_alt_jitter_m /
          spawn_yaw_jitter_rad: spawn geometry (metres, radians). podium_height_m
            defaults to 0: worlds start RESTING on the ground with the env's airborne
            latch cold, so the arm-idle → spool → lift sequence of a real drone cannot
            trip the ground-impact terminal (DESIGN.md §7 step 6). Raise it only for a
            physical podium — a raised pad free-falls during spool-up.
        """
        if spawn_mode not in ("podium", "spread"):
            raise ValueError(f"spawn_mode must be 'podium' or 'spread', got {spawn_mode!r}")
        self.gates = gates if gates is not None else figure_eight(3)
        self.body_radius_m = float(body_radius_m)
        min_inner = float(jnp.min(self.gates.inner_half))
        if self.body_radius_m >= min_inner:
            raise ValueError(
                f"body_radius_m={self.body_radius_m} closes a "
                f"{2 * min_inner:.3f} m opening entirely"
            )
        self.w_prog = float(w_prog)
        self.w_gate = float(w_gate)
        self.w_rate = float(w_rate)
        self.pre_gate_offset_m = float(pre_gate_offset_m)
        self.spawn_mode = spawn_mode
        self.spawn_dist_m = float(spawn_dist_m)
        self.podium_height_m = float(podium_height_m)
        self.spawn_lateral_m = float(spawn_lateral_m)
        self.spawn_alt_jitter_m = float(spawn_alt_jitter_m)
        self.spawn_yaw_jitter_rad = float(spawn_yaw_jitter_rad)

        # Podium pad: an explicit (x, y), or — on the shipped default course only — the
        # centre of the gate cluster opposite gate 0 (the second course half: the other
        # lobe), the canonical figure-eight start. Custom courses keep the legacy
        # behind-gate-0 pad unless podium_pos_m says otherwise.
        self._podium_xy: jax.Array | None = None
        if podium_pos_m is not None:
            self._podium_xy = jnp.asarray(podium_pos_m, jnp.float32).reshape(2)
        elif gates is None and self.num_gates >= 2:
            self._podium_xy = jnp.mean(self._centers[self.num_gates // 2 :, :2], axis=0)

        self.vision = bool(vision)
        tail = (
            ObsTerm("vel_body", 3, "m/s body FLU"),
            ObsTerm("rot_matrix", 9, "R body FLU -> world z-up row-major"),
            ObsTerm("last_action", 4, "[-1,1]"),
        )
        if self.vision:
            # Imported here, not at module top: the state-only variant has no reason to
            # pull the ray-cast machinery in, and stays usable without it.
            from skyflow.vision.camera import CameraModel
            from skyflow.vision.renderer import render_masks

            self._camera = camera if camera is not None else CameraModel()
            self._render_masks = render_masks
            h, w = int(self._camera.height), int(self._camera.width)
            self.image_shape: tuple[int, int, int] | None = (h, w, 1)
            self.obs_spec = ObsSpec((ObsTerm("mask", h * w, "{0,1} HxW row-major"), *tail))
        else:
            self._camera = None
            self.image_shape = None
            self.obs_spec = ObsSpec(
                (
                    ObsTerm("gate_rel", 3, "m body FLU"),
                    ObsTerm("gate_normal", 3, "unit body FLU"),
                    ObsTerm("next_gate_rel", 3, "m body FLU"),
                    *tail,
                )
            )

    @property
    def gates(self):
        """The course. Reassigning it re-derives ALL cached geometry below, so the
        reward geometry and the collision geometry can never come from two different
        courses (a subclass that swapped `gates` used to get a chimera: cached
        reward constants from the old course, live crossings from the new)."""
        return self._gates

    @gates.setter
    def gates(self, gates) -> None:
        # Course geometry as fixed world-frame constants, read through the GateSet's
        # public z-up properties (the raw fields are the renderer's internal NED —
        # DESIGN.md §3a). They ride inside jitted programs as baked-in values, indexed
        # per world by the active gate. Lateral/vertical signs are internal-convention;
        # the centering measure only takes magnitudes along them.
        self._gates = gates
        self.num_gates = len(gates)
        self._centers = gates.centers_world  # [G, 3]
        self._normals = gates.normals_world  # [G, 3] unit, facing flight direction
        self._laterals = gates.laterals_world  # [G, 3] unit, in-plane horizontal
        self._verticals = gates.verticals_world  # [G, 3] unit, in-plane vertical
        self._inner = gates.inner_half  # [G, 2] (lateral, vertical) half-extents

    # -- Task protocol -------------------------------------------------------------

    def spawn(self, key: Array, n: int, params: Array) -> tuple[Array, GateTaskState]:
        """Fresh plant rows [n,17] f32 facing the start gate, on the approach side.

        Podium mode places every world on the podium pad at podium height with lateral
        and yaw jitter, facing gate 0 — the pad is the course's fixed podium when one is
        defined (the shipped figure-eight pads the opposite lobe's centre), else
        `spawn_dist_m` behind gate 0. Spread mode draws the start gate uniformly and
        spawns near its altitude (active_gate starts there). Velocity and body rates are
        zero and rotors are at rest — the env's post-step clip lifts them to the
        airframe's idle floor on the first substep. `params` is unused: the spawn is
        course-relative geometry, no equilibrium solve is needed.
        """
        del params
        k_gate, k_lat, k_alt, k_yaw = jax.random.split(key, 4)
        if self.spawn_mode == "spread":
            active = jax.random.randint(k_gate, (n,), 0, self.num_gates, dtype=jnp.int32)
        else:
            active = jnp.zeros((n,), jnp.int32)
        center = self._centers[active]
        normal = self._normals[active]
        lateral = self._laterals[active]

        lat_off = jax.random.uniform(k_lat, (n, 1), jnp.float32, -1.0, 1.0)
        if self.spawn_mode == "podium" and self._podium_xy is not None:
            # fixed pad: face gate 0's centre, jitter across the takeoff line
            to_gate = self._centers[0, :2] - self._podium_xy
            heading = jnp.arctan2(to_gate[1], to_gate[0])
            perp = jnp.stack([-jnp.sin(heading), jnp.cos(heading)])
            xy = self._podium_xy + self.spawn_lateral_m * lat_off * perp
            pos = jnp.concatenate(
                [xy, jnp.full((n, 1), self.podium_height_m, jnp.float32)], axis=-1
            )
            bearing = jnp.broadcast_to(heading, (n,))
        else:
            pos = center - self.spawn_dist_m * normal + self.spawn_lateral_m * lat_off * lateral
            if self.spawn_mode == "podium":
                pos = pos.at[:, 2].set(self.podium_height_m)
            else:
                alt_off = jax.random.uniform(k_alt, (n, 1), jnp.float32, -1.0, 1.0)
                pos = pos + self.spawn_alt_jitter_m * alt_off * jnp.asarray(_UP, jnp.float32)
                pos = pos.at[:, 2].set(jnp.maximum(pos[:, 2], 0.05))  # never below ground
            bearing = jnp.arctan2(normal[:, 1], normal[:, 0])

        # Heading toward the gate, plus yaw jitter, level.
        yaw_off = jax.random.uniform(k_yaw, (n,), jnp.float32, -1.0, 1.0)
        half = 0.5 * (bearing + self.spawn_yaw_jitter_rad * yaw_off)
        zeros = jnp.zeros_like(half)
        quat = jnp.stack([jnp.cos(half), zeros, zeros, jnp.sin(half)], axis=-1)

        plant = jnp.concatenate(
            [pos, jnp.zeros((n, 3)), quat, jnp.zeros((n, 3)), jnp.zeros((n, 4))], axis=-1
        ).astype(jnp.float32)
        return plant, GateTaskState(active_gate=active, passes=jnp.zeros((n,), jnp.int32))

    def observe(
        self,
        plant: Array,
        task_state: GateTaskState,
        imu: tuple[Array, Array],
        last_action: Array,
        key: Array,
        fresh_spawn: bool,
    ) -> tuple[Array, GateTaskState]:
        """Obs rows [n, obs_spec.dim] f32; task_state passes through unchanged.

        `imu`, `key` and `fresh_spawn` are unused: this task observes exact state (the
        IMU packaging stays in sensors.py), and the persistent mask-corruption families
        for vision obs are deferred work (DESIGN.md §12). rot_matrix flattens row-major
        (tasks.base.quat_to_rot).
        """
        del imu, key, fresh_spawn
        pos = plant[:, 0:3]
        vel = plant[:, 3:6]
        quat = plant[:, 6:10]
        rot = quat_to_rot(quat)
        vel_body = _world_to_body(rot, vel)
        rot_flat = rot.reshape(pos.shape[0], 9)

        if self.vision:
            camera = self._camera
            assert camera is not None  # __init__ always builds one in vision mode
            mask = self._render_masks(camera, self.gates, pos, quat)
            head = [mask.reshape(pos.shape[0], -1)]
        else:
            # clip, don't trust: JAX gathers clamp OOB indices silently, and a bad
            # index (mismatched checkpoint, subclass with fewer gates) would stick.
            active = jnp.clip(task_state.active_gate, 0, self.num_gates - 1)
            nxt = jnp.minimum(active + 1, self.num_gates - 1)
            head = [
                _world_to_body(rot, self._centers[active] - pos),
                _world_to_body(rot, jnp.broadcast_to(self._normals[active], pos.shape)),
                _world_to_body(rot, self._centers[nxt] - pos),
            ]
        obs = jnp.concatenate([*head, vel_body, rot_flat, last_action], axis=-1)
        return finalize_obs(obs), task_state

    def evaluate(
        self, prev_plant: Array, plant: Array, task_state: GateTaskState
    ) -> TaskEval:
        """Reward/success/crash on the transition prev_plant → plant (DESIGN.md §9).

        reward = w_prog·(d_prev - d) toward the active pre-gate point
               + w_gate·centering on a clean forward pass of the active gate
               - w_rate·‖ω‖.
        Crash: the segment touched ANY gate's frame solid (frame hit, from
        `classify_crossings`), or crossed the active gate's centre plane forward
        without a clean pass (miss — which also catches wide fly-arounds beyond the
        outer edge; pass and miss partition every forward transit). Success: clean pass of the last gate. On a pass the active gate
        advances (clipping at the last); progress this step is measured against the
        pre-advance gate for both endpoints, so the term stays telescoping.
        """
        pos_prev = prev_plant[:, 0:3]
        pos = plant[:, 0:3]
        omega = plant[:, 10:13]
        # Clip once so a bad stored index (mismatched checkpoint, subclass with a
        # shorter course) self-heals through new_active below, instead of riding
        # JAX's silent gather-clamp forever with `success` permanently unreachable.
        active = jnp.clip(task_state.active_gate, 0, self.num_gates - 1)
        fleet = pos.shape[0]

        center = self._centers[active]
        normal = self._normals[active]

        pre_point = center - self.pre_gate_offset_m * normal
        d_prev = jnp.linalg.norm(pre_point - pos_prev, axis=-1)
        d = jnp.linalg.norm(pre_point - pos, axis=-1)
        reward = self.w_prog * (d_prev - d)

        fwd, _bwd, hit = classify_crossings(pos_prev, pos, self.gates, self.body_radius_m)
        passed = fwd[jnp.arange(fleet), active]

        # Centering at the centre-plane crossing point: SkyDreamer's Chebyshev miss,
        # normalized per axis (rectangular openings score like square ones). On a pass
        # the crossing point is strictly inside the opening (classify_crossings), so
        # centering ∈ (0, 1]; elsewhere it is zeroed and pays nothing. The solve is
        # crossing_offsets — the SAME one classify_crossings uses for its pass
        # predicate, so reward math cannot drift from collision math (TECH_DEBT T15).
        crossed_g, forward_g, lat_g, vert_g = crossing_offsets(pos_prev, pos, self.gates)
        rows = jnp.arange(fleet)
        lat = lat_g[rows, active] / self._inner[active, 0]
        vert = vert_g[rows, active] / self._inner[active, 1]
        centering = jnp.where(passed, 1.0 - jnp.maximum(lat, vert), 0.0)
        reward = reward + self.w_gate * centering

        # Miss: forward crossing of the active centre plane that is not a clean pass —
        # the same crossing predicate classify_crossings applies, so pass and miss
        # partition every forward transit. Catches wide fly-arounds beyond the outer
        # edge, which touch no solid but end the attempt.
        missed = crossed_g[rows, active] & forward_g[rows, active] & ~passed
        crash = jnp.any(hit, axis=-1) | missed

        reward = reward - self.w_rate * jnp.linalg.norm(omega, axis=-1)

        success = passed & (active == self.num_gates - 1)
        new_active = jnp.where(
            passed, jnp.minimum(active + 1, self.num_gates - 1), active
        ).astype(jnp.int32)
        new_state = GateTaskState(
            active_gate=new_active, passes=task_state.passes + passed.astype(jnp.int32)
        )
        info = {
            "gate_passed": passed.astype(jnp.float32),
            "gate_missed": missed.astype(jnp.float32),
            "gate_centering": centering.astype(jnp.float32),
            "gate_active": new_active.astype(jnp.float32),
        }
        return TaskEval(
            reward=reward.astype(jnp.float32),
            success=success,
            crash=crash,
            info=info,
            task_state=new_state,
        )

    def metrics(self, task_state: GateTaskState) -> dict[str, Array]:
        """Course-progress diagnostics, all [F] float32."""
        return {
            "gate/active_idx": task_state.active_gate.astype(jnp.float32),
            "gate/passes": task_state.passes.astype(jnp.float32),
        }

    @property
    def camera(self) -> "CameraModel | None":
        """The vision-mode camera (None in state mode) — read by the optional viewer."""
        return self._camera

    def viz_scene(self) -> list[dict]:
        """
        Default display scene for the OPTIONAL viewer (DESIGN.md §13): every gate as an
        indexed frame (the viewer accents the active one), a dashed centre line through
        the course, and a grid sized to it. Plain dicts in the skyflow.viz serde form —
        the duck-typed hook keeps the core free of viz imports.
        """
        import numpy as np  # host-side hook: called at viewer build time, never in-jit

        centers = np.asarray(self._centers, np.float64)
        laterals = np.asarray(self._laterals, np.float64)
        verticals = np.asarray(self._verticals, np.float64)
        outer = np.asarray(self.gates.outer_half, np.float64)
        hx = float(np.abs(centers[:, 0]).max() + outer[:, 0].max() + 1.0)
        hy = float(np.abs(centers[:, 1]).max() + outer[:, 0].max() + 1.0)
        scene: list[dict] = [{"type": "grid", "half": (hx, hy)}]
        scene += [
            {
                "type": "gate",
                "center": tuple(centers[g]),
                "lateral": tuple(laterals[g]),
                "vertical": tuple(verticals[g]),
                "half_w": float(outer[g, 0]),
                "half_h": float(outer[g, 1]),
                "index": g,
                # THIS task names its own state; the viewer knows no task fields. A Gate
                # reads a bound value as the active index and turns accent on a match.
                "bind": "task_state.active_gate",
            }
            for g in range(self.num_gates)
        ]
        scene.append(
            {"type": "path", "points": [tuple(c) for c in centers], "closed": True,
             "dashed": True, "style": "dim"}
        )
        if self._podium_xy is not None:
            px, py = (float(v) for v in np.asarray(self._podium_xy))
            scene.append(
                {"type": "marker", "center": (px, py, self.podium_height_m),
                 "style": "bright", "size": 0.12, "plumb": False}
            )
        return scene
