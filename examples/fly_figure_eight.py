"""
Scripted figure-eight fly-through — no policy, no training, no dynamics.

A kinematic tracer follows pre/post-gate waypoints around the shipped figure-eight
course at constant speed and feeds each transition to `GateCourseTask.evaluate` — the
same pass/crash/centering machinery the env applies to real flight. Prints every gate
pass with its centering and the final tally.

`--save-masks N` additionally renders N camera coverage masks along the tour (the
analytic ray-cast renderer the vision-obs variant uses) and writes them as PNGs when
matplotlib is available, `.npy` arrays otherwise.

Run from the repo root:

    uv run python examples/fly_figure_eight.py
    uv run python examples/fly_figure_eight.py --save-masks 6 --outdir masks
    uv run python examples/fly_figure_eight.py --view      # live viewer (skyflow[viz])

Examples are demos, not package code (DESIGN.md §2).
"""

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.tasks.gate_course import GateCourseTask, GateTaskState
from skyflow.vision.gates import figure_eight


def waypoint_path(gates, step_m: float) -> np.ndarray:
    """[T,3] constant-speed samples along pre→post waypoints of every gate in order.

    Asymmetric offsets (0.63 m before, 0.47 m past each plane) keep samples off the
    exact gate planes, where a strict sign-change crossing test would see nothing.
    """
    waypoints = []
    for g in range(len(gates)):
        c = np.asarray(gates.centers_world[g], np.float64)
        n = np.asarray(gates.normals_world[g], np.float64)
        waypoints.append(c - 0.63 * n)
        waypoints.append(c + 0.47 * n)
    path = [waypoints[0]]
    for wp in waypoints[1:]:
        start = path[-1]
        leg = wp - start
        steps = max(2, math.ceil(np.linalg.norm(leg) / step_m) + 1)
        for t in np.linspace(0.0, 1.0, steps)[1:]:
            path.append(start + t * leg)
    return np.asarray(path, np.float32)


def plant_rows(pos: np.ndarray, vel: np.ndarray, yaw: float) -> jax.Array:
    """One [1,17] f32 spec-layout row: given pose, level attitude at the given yaw."""
    quat = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], np.float32)
    row = np.concatenate([pos, vel, quat, np.zeros(3, np.float32), np.zeros(4, np.float32)])
    return jnp.asarray(row, jnp.float32)[None]


def save_masks(gates, path: np.ndarray, count: int, outdir: Path) -> None:
    """Render `count` masks at evenly spaced tour poses; PNG via matplotlib, else .npy."""
    from skyflow.vision.camera import CameraModel
    from skyflow.vision.renderer import render_masks

    try:
        from matplotlib.image import imsave  # pyright: ignore[reportMissingImports]
    except ImportError:
        imsave = None
        print("matplotlib not installed — saving raw .npy masks instead of PNGs")

    cam = CameraModel()
    outdir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(1, path.shape[0] - 1, count).astype(int)
    for j, i in enumerate(idx):
        seg = path[i] - path[i - 1]
        yaw = math.atan2(float(seg[1]), float(seg[0]))
        quat = jnp.asarray(
            [[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]], jnp.float32
        )
        mask = np.asarray(render_masks(cam, gates, jnp.asarray(path[i][None]), quat)[0])
        target = outdir / f"mask_{j:02d}.png"
        if imsave is None:
            np.save(target.with_suffix(".npy"), mask)
        else:
            imsave(target, mask, cmap="gray", vmin=0.0, vmax=1.0)
        print(f"  wrote {target if imsave else target.with_suffix('.npy')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("--speed", type=float, default=2.0, help="tracer speed, m/s")
    ap.add_argument("--dt", type=float, default=0.02, help="control step, s")
    ap.add_argument("--save-masks", type=int, default=0, metavar="N",
                    help="render N coverage masks along the tour")
    ap.add_argument("--outdir", type=Path, default=Path("figure_eight_masks"),
                    help="directory for --save-masks output")
    ap.add_argument("--gates-per-lobe", type=int, default=3,
                    help="k gates per lemniscate lobe (course has 2k gates)")
    ap.add_argument("--view", action="store_true",
                    help="open the live viewer, wall-clock paced (needs skyflow[viz])")
    ap.add_argument("--headless", action="store_true", help="viewer on the SDL dummy driver")
    ap.add_argument("--frames", type=int, default=None, help="close the viewer after N frames")
    ap.add_argument("--shot", default=None, help="screenshot path saved when --frames ends")
    args = ap.parse_args()

    gates = figure_eight(args.gates_per_lobe)
    task = GateCourseTask(gates)
    path = waypoint_path(gates, step_m=args.speed * args.dt)
    print(f"course: {len(gates)} gates, tour of {path.shape[0]} steps "
          f"at {args.speed:.1f} m/s ({args.dt * 1e3:.0f} ms steps)")

    viewer = None
    if args.view:
        from skyflow.viz import Scene, Viewer, ViewFrame  # optional extra

        viewer = Viewer(
            Scene.from_dicts(task.viz_scene()),
            gates=gates,
            watch=(0,),
            control="motors",
            dt=args.dt,
            title="SkyFlow Viz — figure_eight · tracer",
            headless=args.headless, frames=args.frames, shot=args.shot,
        )

    evaluate = jax.jit(task.evaluate)
    state = GateTaskState(
        active_gate=jnp.zeros((1,), jnp.int32), passes=jnp.zeros((1,), jnp.int32)
    )
    crashes = 0
    success = False
    for i in range(1, path.shape[0]):
        t0 = time.perf_counter()
        seg = (path[i] - path[i - 1]) / args.dt
        yaw = math.atan2(float(seg[1]), float(seg[0]))
        plant = plant_rows(path[i], seg.astype(np.float32), yaw)
        ev = evaluate(
            plant_rows(path[i - 1], seg.astype(np.float32), yaw),
            plant,
            state,
        )
        if float(ev.info["gate_passed"][0]) > 0.0:
            gate = int(state.active_gate[0])
            print(f"  gate {gate}: pass, centering {float(ev.info['gate_centering'][0]):.3f}")
        crashes += int(bool(ev.crash[0]))
        success = success or bool(ev.success[0])
        state = ev.task_state
        if viewer is not None:
            viewer.push(ViewFrame(
                plant=np.asarray(plant), step=i, t=i * args.dt,
                channels={"reward": np.asarray(ev.reward)},
                task_state=jax.device_get(state),
            ))
            while viewer.paused and viewer.open:
                viewer.idle()
            if not viewer.open:
                break
            time.sleep(max(0.0, args.dt - (time.perf_counter() - t0)))

    print(f"gates passed: {int(state.passes[0])}/{len(gates)}"
          f" — success={success}, miss/frame-hit steps={crashes}")

    if args.save_masks > 0:
        save_masks(gates, path, args.save_masks, args.outdir)


if __name__ == "__main__":
    main()
