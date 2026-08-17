"""
Hand-fly SkyFlow through the real Betaflight SITL — sticks in, viewer out.

The loop is deliberately thin: read AETR sticks, `env.step` with control="sticks" (the
same 1 kHz firmware path a stick-level policy trains against), show the viewer, pace to
the wall clock. Runs the CPU firmware fleet pinned to the JAX CPU backend (fleet of 1;
DESIGN.md §10). Needs the `viz` and `firmware` extras.

Run from the repo root:

    uv run python examples/fly_teleop.py                        # keyboard sticks
    uv run python examples/fly_teleop.py --sticks joystick      # local gamepad (pygame)
    uv run python examples/fly_teleop.py --sticks udp:9111      # UDP AETR datagrams
    uv run python examples/fly_teleop.py --task hover --record dvr/

Keyboard: W/S throttle · A/D yaw · arrows roll/pitch (window must have focus).
UDP protocol: 20-byte little-endian `<ffffI` = roll, pitch, yaw, throttle in [-1,1] plus a
button bitmask (bit0 reset, bit1 save lap); latest datagram wins, any sender works.
Recording (--record DIR): R / button-0 resets the world and DISCARDS the lap; K /
button-1 saves it; a lap still open at quit is saved too. Replay with
`python -m skyflow.viz.replay DIR/lap_*.npz`.

Examples are demos, not package code (DESIGN.md §2).
"""

import argparse
import os
import socket
import struct
import time
from pathlib import Path

# The CPU SITL is the teleop backend (GPU firmware fleets want n >= 3, §10); pin the
# platform before jax initializes.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from skyflow import SimConfig, SkyFlowEnv

_UDP_FMT = "<ffffI"  # roll, pitch, yaw, throttle, buttons
_BTN_RESET, _BTN_SAVE = 1, 2


class KeyboardSticks:
    """Spring-centered roll/pitch/yaw on keys; throttle integrates on W/S."""

    def __init__(self) -> None:
        self.throttle = -1.0  # stick-low rest state so the firmware can arm

    def read(self, pygame, dt: float) -> tuple[np.ndarray, int]:
        keys = pygame.key.get_pressed()
        roll = 0.5 * (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]))
        pitch = 0.5 * (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN]))
        yaw = 0.5 * (int(keys[pygame.K_d]) - int(keys[pygame.K_a]))
        rate = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        self.throttle = float(np.clip(self.throttle + 1.2 * rate * dt, -1.0, 1.0))
        buttons = _BTN_SAVE if keys[pygame.K_k] else 0
        return np.array([[roll, pitch, self.throttle, yaw]], np.float32), buttons


class JoystickSticks:
    """First local gamepad via pygame; axis order roll,pitch,throttle,yaw (--axes)."""

    def __init__(self, pygame, axes: tuple[int, int, int, int]) -> None:
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise SystemExit("no joystick found — plug one in or use --sticks keyboard/udp")
        self._js = pygame.joystick.Joystick(0)
        self._js.init()
        self._axes = axes
        print(f"joystick: {self._js.get_name()} ({self._js.get_numaxes()} axes)")

    def read(self, pygame, dt: float) -> tuple[np.ndarray, int]:
        r, p, t, y = (float(self._js.get_axis(a)) for a in self._axes)
        # SDL sticks read +1 down; AETR wants +1 forward/up
        aetr = np.array([[r, -p, -t, y]], np.float32)
        buttons = 0
        if self._js.get_numbuttons() > 0 and self._js.get_button(0):
            buttons |= _BTN_RESET
        if self._js.get_numbuttons() > 1 and self._js.get_button(1):
            buttons |= _BTN_SAVE
        return aetr, buttons


class UdpSticks:
    """Latest-wins AETR datagrams — the radio can live on another machine entirely."""

    def __init__(self, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.setblocking(False)
        self._last = np.array([[0.0, 0.0, -1.0, 0.0]], np.float32)  # rest: throttle low
        self._buttons = 0
        print(f"listening for {_UDP_FMT!r} stick datagrams on udp:{port}")

    def read(self, pygame, dt: float) -> tuple[np.ndarray, int]:
        packet = None
        while True:  # drain the socket; only the newest packet matters
            try:
                packet, _ = self._sock.recvfrom(64)
            except BlockingIOError:
                break
        if packet is not None and len(packet) >= struct.calcsize(_UDP_FMT):
            roll, pitch, yaw, throttle, buttons = struct.unpack_from(_UDP_FMT, packet)
            # senders transmit the raw SDL pitch axis (stick forward = negative);
            # the receiver negates — the established wire convention
            self._last = np.array([[roll, -pitch, throttle, yaw]], np.float32)
            self._buttons = int(buttons)
        return self._last, self._buttons


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("--sticks", default="keyboard", metavar="SRC",
                    help='"keyboard", "joystick", or "udp:PORT"')
    ap.add_argument("--axes", default="0,1,2,3",
                    help="joystick axis indices for roll,pitch,throttle,yaw")
    ap.add_argument("--task", default="figure_eight", help="registered task to fly in")
    ap.add_argument("--control-hz", type=float, default=100.0, help="control rate")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-quit after this long (0 = fly)")
    ap.add_argument("--seed", type=int, default=0, help="reset PRNG seed")
    ap.add_argument("--record", type=Path, default=None, metavar="DIR",
                    help="record laps as DIR/lap_NN.npz (R discards, K saves)")
    ap.add_argument("--headless", action="store_true", help="viewer on the SDL dummy driver")
    ap.add_argument("--frames", type=int, default=None, help="close the viewer after N frames")
    ap.add_argument("--shot", default=None, help="screenshot path saved when --frames ends")
    args = ap.parse_args()

    import pygame  # via the viz extra

    from skyflow.viz import FlightLog, Viewer  # raises with install guidance if absent

    cfg = SimConfig(
        num_envs=1,
        task=args.task,
        control="sticks",  # SkyFlowEnv raises with install guidance if cudaflight is absent
        control_hz=args.control_hz,
        physics_hz=1000.0,
        physics_dr_scale=0.0,
        # hand flying wanders more than a policy: hold the goal still and give room
        task_kwargs={"goal_hold_s": 3600.0} if args.task == "hover" else {},
    )
    env = SkyFlowEnv(cfg)
    viewer = Viewer.for_env(
        env, watch=(0,), headless=args.headless, frames=args.frames, shot=args.shot,
        title=f"SkyFlow Viz — teleop · {args.task} · Betaflight sticks",
    )

    lap = 0

    def fresh_log() -> "FlightLog | None":
        return FlightLog.for_env(env, watch=(0,)) if args.record else None

    def save_lap(log: "FlightLog | None") -> None:
        nonlocal lap
        if log is not None and len(log) > 0:
            path = log.save(args.record / f"lap_{lap:02d}.npz")
            print(f"saved {path} ({len(log)} steps)")
            lap += 1

    log = fresh_log()
    key = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(key)
    step = jax.jit(env.step)
    dt = env.dt_control

    if args.sticks == "keyboard":
        sticks = KeyboardSticks()
    elif args.sticks == "joystick":
        sticks = JoystickSticks(pygame, tuple(int(a) for a in args.axes.split(",")))
    elif args.sticks.startswith("udp:"):
        sticks = UdpSticks(int(args.sticks.split(":", 1)[1]))
    else:
        raise SystemExit(f"unknown --sticks {args.sticks!r}")

    print(f"flying {args.task} at {args.control_hz:.0f} Hz — throttle low to arm, Esc quits")
    n, last_buttons, t_start = 0, 0, time.perf_counter()
    while viewer.open:
        t0 = time.perf_counter()
        aetr, buttons = sticks.read(pygame, dt)
        obs, state, reward, done, info = step(state, jnp.asarray(aetr))
        if log is not None:
            log.capture(state, action=aetr, reward=reward, done=done)
        viewer.frame(state, obs=obs, action=aetr, reward=reward, done=done, info=info)

        rising = buttons & ~last_buttons
        last_buttons = buttons
        if viewer.take_reset() or rising & _BTN_RESET:
            n += 1
            obs, state = env.reset(jax.random.fold_in(key, n))
            log = fresh_log()  # a reset discards the open lap
        if rising & _BTN_SAVE:
            save_lap(log)
            log = fresh_log()
        while viewer.paused and viewer.open:
            viewer.idle()
        if args.seconds and time.perf_counter() - t_start > args.seconds:
            break
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    save_lap(log)  # a lap still open at quit is kept
    viewer.close()


if __name__ == "__main__":
    main()
