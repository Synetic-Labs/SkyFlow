"""
PD hover demo — fly a small fleet from the pad to the task's goal setpoint.

A ~20-line cascaded PD controller runs on the hover task's own observations: position
error → desired acceleration → desired thrust direction → attitude error → per-motor
throttle through the plain geometric mixer of the built-in airframe's rotor layout.
No policy, no training, no learning code — just enough control to show the env flying
(examples are demos, not package code, DESIGN.md §2). Body rates come from the sim
state for damping; that is a demo convenience, not part of the task observation.

Run from the repo root:

    uv run python examples/fly_hover.py
    uv run python examples/fly_hover.py --seconds 5 --fleet 8
    uv run python examples/fly_hover.py --seconds 30 --view   # live viewer (skyflow[viz])
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from skyflow import SimConfig, SkyFlowEnv
from skyflow.params import AIRFRAMES

# Position PD (world frame): error [m] → desired acceleration [m/s²].
KP_POS, KD_POS = 3.0, 3.0
ACC_XY_MAX, ACC_Z_MIN, ACC_Z_MAX = 4.0, -4.0, 6.0
# Attitude PD: tilt error (sin of the angle) / body rate [rad/s] → throttle differential.
KP_ATT, KD_ATT = 0.09, 0.015
# Rotor layout signs for the built-in airframes' order (FL, FR, RR, RL), body FLU.
X_SIGN = np.array([1.0, 1.0, -1.0, -1.0], np.float32)
Y_SIGN = np.array([1.0, -1.0, -1.0, 1.0], np.float32)
SPIN = np.array([-1.0, 1.0, -1.0, 1.0], np.float32)


def hover_throttle(airframe) -> tuple[float, float]:
    """(u_hover, g): throttle that balances weight, from the airframe's spec row.

    Ω_h = √(m·g / Σct2); the Crazyflie command map is linear (throttle_k = 1), so
    u = (Ω_h - Ω_min)/(Ω_max - Ω_min)."""
    v = airframe.values
    g = float(v["grav"])
    omega_h = float(np.sqrt(v["mass"] * g / sum(v["ct2"])))
    u = (omega_h - airframe.rotor_speed_min) / (
        airframe.rotor_speed_max - airframe.rotor_speed_min
    )
    return u, g


def pd_action(obs: np.ndarray, omega: np.ndarray, u_hover: float, g: float) -> np.ndarray:
    """[F,4] motor actions in [-1,1] from hover obs [rel_pos, vel, rot_matrix, ...]."""
    rel, vel = obs[:, 0:3], obs[:, 3:6]
    rot = obs[:, 6:15].reshape(-1, 3, 3)  # body→world

    acc = KP_POS * rel - KD_POS * vel
    acc[:, 0:2] = np.clip(acc[:, 0:2], -ACC_XY_MAX, ACC_XY_MAX)
    acc[:, 2] = np.clip(acc[:, 2], ACC_Z_MIN, ACC_Z_MAX)
    thrust_vec = acc + np.array([0.0, 0.0, g], np.float32)
    a_norm = np.linalg.norm(thrust_vec, axis=-1, keepdims=True)
    t_dir = thrust_vec / a_norm

    body_z = rot[:, :, 2]
    err_world = np.cross(body_z, t_dir)  # rotation axis, |err| = sin(tilt error)
    err_body = np.einsum("fij,fi->fj", rot, err_world)  # Rᵀ · err
    att = KP_ATT * err_body - KD_ATT * omega

    # T ∝ u² on the linear Crazyflie map, so u scales with √(demanded/hover accel).
    u_coll = u_hover * np.sqrt(a_norm[:, 0] / g)
    mix = (
        Y_SIGN[None, :] * att[:, 0:1]  # +τx: raise the +y (left) rotors
        - X_SIGN[None, :] * att[:, 1:2]  # +τy: raise the -x (rear) rotors
        - SPIN[None, :] * att[:, 2:3]  # +τz: raise the CW (spin -1) rotors
    )
    u = np.clip(u_coll[:, None] + mix, 0.0, 1.0)
    return 2.0 * u - 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("--seconds", type=float, default=3.0, help="flight time, s")
    ap.add_argument("--fleet", type=int, default=4, help="number of worlds")
    ap.add_argument("--seed", type=int, default=0, help="reset PRNG seed")
    ap.add_argument("--view", action="store_true",
                    help="open the live viewer, wall-clock paced (needs skyflow[viz])")
    ap.add_argument("--headless", action="store_true", help="viewer on the SDL dummy driver")
    ap.add_argument("--frames", type=int, default=None, help="close the viewer after N frames")
    ap.add_argument("--shot", default=None, help="screenshot path saved when --frames ends")
    args = ap.parse_args()

    cfg = SimConfig(
        num_envs=args.fleet,
        task="hover",
        # Goal held longer than the flight so the setpoint never moves mid-demo. (The
        # env also accepts a pre-built instance: SkyFlowEnv(cfg, task=HoverTask(...)).)
        task_kwargs={"goal_hold_s": 60.0},
        physics_dr_scale=0.0,  # nominal vehicle: this demo shows the sim, not DR
    )
    env = SkyFlowEnv(cfg)
    u_hover, g = hover_throttle(AIRFRAMES["crazyflie"])

    obs, state = env.reset(jax.random.PRNGKey(args.seed))
    goal = np.asarray(state.task_state.goal)
    print(f"fleet of {args.fleet}: pad → goal, {args.seconds:.1f} s at {cfg.control_hz:.0f} Hz")
    print(f"  goal[0] = {goal[0].round(2)}, initial error {np.linalg.norm(np.asarray(obs)[:, 0:3], axis=-1).mean():.2f} m")

    viewer = None
    if args.view:
        from skyflow.viz import Viewer  # optional extra — imported only on request

        viewer = Viewer.for_env(
            env, watch=tuple(range(min(4, args.fleet))),
            headless=args.headless, frames=args.frames, shot=args.shot,
        )

    jstep = jax.jit(env.step)
    n_steps = round(args.seconds * cfg.control_hz)
    for t in range(1, n_steps + 1):
        t0 = time.perf_counter()
        action = pd_action(
            np.asarray(obs), np.asarray(state.plant[:, 10:13]), u_hover, g
        )
        obs, state, reward, done, info = jstep(state, jnp.asarray(action))
        if viewer is not None:
            viewer.frame(state, obs=obs, action=action, reward=reward, done=done, info=info)
            if viewer.take_reset():
                obs, state = env.reset(jax.random.PRNGKey(args.seed + t))
            while viewer.paused and viewer.open:
                viewer.idle()
            if not viewer.open:
                break
            # the demo runs faster than real time; pace it so the flight reads naturally
            time.sleep(max(0.0, 1.0 / cfg.control_hz - (time.perf_counter() - t0)))
        elif bool(done.any()):
            raise SystemExit(f"episode ended early at step {t} — controller diverged")
        if t % int(cfg.control_hz / 2) == 0:
            err = np.linalg.norm(np.asarray(obs)[:, 0:3], axis=-1)
            print(f"  t={t / cfg.control_hz:4.1f} s  |error| mean {err.mean():.3f} m  max {err.max():.3f} m")

    err = np.linalg.norm(np.asarray(obs)[:, 0:3], axis=-1)
    print(f"final position error after {args.seconds:.1f} s: "
          f"mean {err.mean():.3f} m, max {err.max():.3f} m")


if __name__ == "__main__":
    main()
