"""
Replay host — the DVR (DESIGN.md §13): `python -m skyflow.viz.replay flight.npz`.

The same window fed from a file: scrub (←/→ while paused), pause, 0.125-8x speed, and a
pilot cam re-rendered from poses at whatever resolution you ask for — footage quality is
chosen at watch time, not record time. Geometry replays exactly (the renderer is pure);
the policy pane stays hidden because observations are not logged. `--mp4` exports through
imageio when importable (soft dependency, like matplotlib in the examples).
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from skyflow.viz.frame import ViewFrame
from skyflow.viz.primitives import Grid, Scene
from skyflow.viz.record import FlightLog, ReplayLog, gateset_from_dict

__all__ = ["main", "replay", "viewer_for_log"]


def viewer_for_log(log: ReplayLog, *, pilot: tuple[int, int] | None = None, **kw: Any):
    """A Viewer rebuilt purely from a log's header — no env, no task code."""
    from skyflow.viz.viewer import Viewer

    header = log.header
    scene = Scene.from_dicts(header.get("scene", []))
    if not any(isinstance(p, Grid) for p in scene):
        scene.add(Grid())
    camera = None
    if header.get("camera"):
        from skyflow.vision.camera import CameraModel

        camera = CameraModel(**header["camera"])
    gates = gateset_from_dict(header["gateset"]) if header.get("gateset") else None
    n_watch = log.plant.shape[1]
    viewer = Viewer(
        scene,
        camera=camera,
        gates=gates,
        watch=tuple(header.get("watch", range(n_watch)))[:n_watch],
        omega_max=header.get("omega_max"),
        control=header.get("control", "motors"),
        dt=log.dt,
        title=f"SkyFlow Viz — replay · {header.get('task', '?')}",
        **kw,
    )
    if pilot is not None:
        from skyflow.viz.fpv import PilotCam

        viewer._pilot = PilotCam(camera, gates, height=pilot[0], width=pilot[1])
    return viewer


def _namespace_from_binds(binds: dict[str, np.ndarray], i: int) -> SimpleNamespace | None:
    """Rebuild the object tree the scene's string binds walk (row i of each logged path)."""
    if not binds:
        return None
    root = SimpleNamespace()
    for path, arr in binds.items():
        node = root
        parts = path.split(".")
        for part in parts[:-1]:
            child = getattr(node, part, None)
            if child is None:
                child = SimpleNamespace()
                setattr(node, part, child)
            node = child
        setattr(node, parts[-1], arr[i])
    return root


def _frame_at(log: ReplayLog, i: int) -> ViewFrame:
    # binds resolve against the ViewFrame itself; logged paths start at "task_state."
    ns = _namespace_from_binds(log.binds, i)
    return ViewFrame(
        plant=log.plant[i],
        step=i * int(log.header.get("every", 1)),
        t=i * log.dt,
        action=None if log.action is None else log.action[i],
        channels={name: arr[i] for name, arr in log.channels.items()},
        done=None if log.done is None else log.done[i],
        task_state=getattr(ns, "task_state", None),
    )


def replay(
    path: str | Path,
    *,
    pilot: tuple[int, int] | None = None,
    speed: float = 1.0,
    start: int = 0,
    headless: bool = False,
    frames: int | None = None,
    shot: str | None = None,
    mp4: str | None = None,
) -> None:
    """Scrub/replay a flight.npz; `mp4` renders every logged row offscreen instead."""
    log = FlightLog.load(path)
    total = len(log)

    if mp4 is not None:
        try:
            import imageio.v3 as iio  # pyright: ignore[reportMissingImports] — soft dep: only the export path needs it
        except ImportError as e:
            raise ImportError("mp4 export needs imageio: pip install 'imageio[ffmpeg]'") from e
        viewer = viewer_for_log(log, pilot=pilot, headless=True, threaded=False)
        grabs = []
        for i in range(total):
            viewer.push(_frame_at(log, i), force=True)
            if not viewer.open:
                break
            grabs.append(viewer.grab())
        viewer.close()
        iio.imwrite(mp4, np.stack(grabs), fps=round(1.0 / log.dt), codec="libx264")
        print(f"wrote {mp4}: {len(grabs)} frames at {round(1.0 / log.dt)} fps")
        return

    # synchronous: the scrub cursor needs every push drawn exactly once, in this thread
    viewer = viewer_for_log(
        log, pilot=pilot, headless=headless, frames=frames, shot=shot, threaded=False
    )
    viewer.speed = float(speed)
    i = int(np.clip(start, 0, total - 1))
    while viewer.open:
        t0 = time.perf_counter()
        viewer.push(_frame_at(log, i), force=True)
        i += viewer.take_seek()
        if not viewer.paused:
            i += 1
        if i >= total:  # hold on the last frame rather than exiting: it's a DVR
            i = total - 1
            viewer.paused = True
        i = max(0, i)
        time.sleep(max(0.0, log.dt / viewer.speed - (time.perf_counter() - t0)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("path", type=Path, help="flight.npz written by FlightLog.save")
    ap.add_argument("--pilot-cam", default=None, metavar="WxH",
                    help="pilot cam resolution, e.g. 384x288 (default 256x192)")
    ap.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    ap.add_argument("--start", type=int, default=0, help="first logged row to show")
    ap.add_argument("--headless", action="store_true", help="SDL dummy driver (CI)")
    ap.add_argument("--frames", type=int, default=None, help="auto-close after N frames")
    ap.add_argument("--shot", default=None, help="screenshot path saved when --frames ends")
    ap.add_argument("--mp4", default=None, help="export the whole log to this mp4 instead")
    args = ap.parse_args(argv)
    pilot = None
    if args.pilot_cam:
        w, h = (int(v) for v in args.pilot_cam.lower().split("x"))
        pilot = (h, w)
    replay(
        args.path,
        pilot=pilot,
        speed=args.speed,
        start=args.start,
        headless=args.headless,
        frames=args.frames,
        shot=args.shot,
        mp4=args.mp4,
    )


if __name__ == "__main__":
    main()
