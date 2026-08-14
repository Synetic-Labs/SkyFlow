"""
Fleet-throughput benchmark — time SkyFlow across fleet sizes, in two modes.

`env` times the full jitted env.step: RK4 substeps, wind/poke draws, transport delay,
observation, reward, termination, and in-jit auto-reset — what a training loop pays.
`dynamics` times bare RK4 substeps of the plant (skyflow.dynamics.substep), nothing
else — the raw-integrator unit that simulator "steps/s" headlines usually quote.
Default runs both.

Dynamics mode scans --substeps (default 100) substeps inside each timed call and
divides the metrics by the count. One device dispatch then covers 100 integration
steps, so per-call submission latency (~1.3 ms under WSL2, enough to cap a
one-step-per-call loop at ~190 M steps/s regardless of compute) amortizes away and the
measurement becomes device-bound. --substeps 1 times one dispatch per step instead.

Methodology: 1000 timed calls per fleet size, every call synchronized with
jax.block_until_ready, and a 2-step probe skipping sizes projected past the per-size
budget. Rows append to benchmark/data/skyflow_<timestamp>.csv; benchmark/compare.py
tabulates any CSVs in this schema side by side.

Compare simulators on real_time_factor (simulated seconds per wall-clock second) — it
normalizes away step-rate differences. fps counts timed calls x worlds: control steps
in env mode, physics steps in dynamics mode.

Run from the repo root:

    uv run python benchmark/main.py --device gpu --worlds 64,1024,16384,65536,262144
    uv run python benchmark/main.py --device cpu --mode env --worlds 16,256 --n-steps 200
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# JAX initializes its CUDA backend at import: on a machine with CUDA jaxlib installed,
# even a --device cpu run would preallocate GPU memory. Decide before importing jax.
if "--device=cpu" in sys.argv or (
    "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1 :][:1] == ["cpu"]
):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from skyflow import SimConfig, SkyFlowEnv
from skyflow.dynamics import substep
from skyflow.params import AIRFRAMES

# Keep this schema stable: compare.py tabulates any CSVs that share it, so results from
# different simulators, machines, and runs merge as-is.
CSV_COLUMNS = [
    "test_type",
    "n_drones",
    "n_worlds",
    "n_steps",
    "total_time_s",
    "avg_step_time_s",
    "fps",
    "real_time_factor",
    "device",
]


def timed_run(advance, carry, n_steps: int, budget_s: float):
    """
    Drive `advance` (carry → carry, fully synchronized inside) → (per-call wall times,
    compile seconds), or None when the post-warmup probe projects past `budget_s`.
    """
    tstart = time.perf_counter()
    carry = advance(carry)
    compile_s = time.perf_counter() - tstart

    probe = []
    for _ in range(2):
        tstart = time.perf_counter()
        carry = advance(carry)
        probe.append(time.perf_counter() - tstart)
    if probe[1] * n_steps > budget_s:
        return None

    times = []
    for _ in range(n_steps):
        tstart = time.perf_counter()
        carry = advance(carry)
        times.append(time.perf_counter() - tstart)
    return times, compile_s


def bench_env(cfg: SimConfig, n_steps: int, budget_s: float):
    """Full env.step. Blocks on the whole (obs, state, reward, done, info) tuple so no
    part of the pipeline is dead-code-eliminated out of the measurement."""
    env = SkyFlowEnv(cfg)
    step = jax.jit(env.step)
    obs, state = env.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((env.fleet, env.act_dim), jnp.float32)
    jax.block_until_ready((obs, state))

    def advance(state):
        out = jax.block_until_ready(step(state, action))
        return out[1]

    return timed_run(advance, state, n_steps, budget_s)


def bench_dynamics(cfg: SimConfig, n_steps: int, budget_s: float, substeps: int):
    """Bare RK4 substeps — plant integration only, constant hover-ish rotor command, no
    wind/obs/reward/reset. Each timed call scans `substeps` steps in one dispatch; the
    caller divides the metrics back down."""
    env = SkyFlowEnv(cfg)  # reused only to spawn plant rows and per-world param packs
    _, state = env.reset(jax.random.PRNGKey(0))
    af = AIRFRAMES[cfg.airframe]
    w_min, w_max = af.rotor_speed_min, af.rotor_speed_max
    omega_cmd = jnp.full((env.fleet, 4), 0.5 * (w_min + w_max), jnp.float32)
    zeros3 = jnp.zeros((env.fleet, 3), jnp.float32)
    dt = 1.0 / cfg.physics_hz
    params = state.params

    @jax.jit
    def scan_substeps(plant):
        def body(p, _):
            return substep(p, omega_cmd, zeros3, zeros3, zeros3, params, dt, w_min, w_max), None

        return jax.lax.scan(body, plant, length=substeps)[0]

    def advance(plant):
        return jax.block_until_ready(scan_substeps(plant))

    return timed_run(advance, state.plant, n_steps, budget_s)


def summarize(times: list[float], n_worlds: int, hz: float, steps_per_call: int = 1) -> dict:
    """Metrics for one size; `hz` is the simulated step rate (control_hz in env mode,
    physics_hz in dynamics mode) and `steps_per_call` how many such steps one timed call
    advanced. All step-denominated fields are per simulated step, not per call. Warns
    when jit leaked into the timed region."""
    tmin, tmax = float(np.min(times)), float(np.max(times))
    if tmax / tmin > 10:
        print(
            f"  Warning: step time varies {tmax / tmin:.0f}x "
            f"(max {tmax:.2e}s, min {tmin:.2e}s). Is JIT compiling during the benchmark?"
        )
    total = float(np.sum(times))
    n_steps = len(times) * steps_per_call
    return {
        "n_steps": n_steps,
        "total_time_s": total,
        "avg_step_time_s": float(np.mean(times)) / steps_per_call,
        "fps": n_steps * n_worlds / total,
        "real_time_factor": (n_steps / hz) * n_worlds / total,
    }


MODES = {
    # mode → (bench fn, CSV test_type, rate of one timed call)
    "env": (bench_env, "skyflow_env", lambda cfg: cfg.control_hz),
    "dynamics": (bench_dynamics, "skyflow_dynamics", lambda cfg: cfg.physics_hz),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("--device", default="auto", help='"cpu", "gpu", or "auto" (default)')
    ap.add_argument("--mode", default="both", choices=["env", "dynamics", "both"])
    ap.add_argument(
        "--worlds", default=None,
        help="comma-separated fleet sizes (e.g. 64,1024,16384); default 10^0..10^max-exp",
    )
    ap.add_argument("--max-exp", type=int, default=6, help="sweep n_worlds = 10^0 .. 10^max_exp")
    ap.add_argument("--n-steps", type=int, default=1000, help="timed calls per fleet size")
    ap.add_argument(
        "--substeps", type=int, default=100,
        help="dynamics mode: physics steps scanned per timed call; 1 = one dispatch "
        "per step (dispatch-latency-bound on WSL2)",
    )
    ap.add_argument("--budget", type=float, default=60.0, help="per-size time budget, s")
    ap.add_argument("--task", default="hover", help="registered task name")
    ap.add_argument("--control", default="motors", help='"motors" | "sticks"')
    ap.add_argument("--control-hz", type=float, default=100.0)
    ap.add_argument("--physics-hz", type=float, default=1000.0)
    args = ap.parse_args()

    try:
        device = jax.devices()[0] if args.device == "auto" else jax.devices(args.device)[0]
    except RuntimeError as e:
        raise SystemExit(f"no {args.device!r} device available to JAX: {e}") from e
    print(f"device: {device} ({device.platform}), jax {jax.__version__}")

    if args.worlds:
        world_sizes = [int(w) for w in args.worlds.split(",")]
    else:
        world_sizes = [10**i for i in range(args.max_exp + 1)]
    modes = ["env", "dynamics"] if args.mode == "both" else [args.mode]

    csv_file = (
        Path(__file__).parent
        / "data"
        / f"skyflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    csv_file.parent.mkdir(exist_ok=True)
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)

    for mode in modes:
        bench, test_type, rate = MODES[mode]
        steps_per_call = args.substeps if mode == "dynamics" else 1
        per_call = f", {steps_per_call} substeps/call" if steps_per_call > 1 else ""
        print(f"\n[{mode}] sweeping fleet sizes {world_sizes}, {args.n_steps} calls{per_call}...")
        for n_worlds in world_sizes:
            print("-" * 80)
            cfg = SimConfig(
                num_envs=n_worlds,
                task=args.task,
                control=args.control,
                control_hz=args.control_hz,
                physics_hz=args.physics_hz,
            )
            with jax.default_device(device):
                if mode == "dynamics":
                    result = bench(cfg, args.n_steps, args.budget, args.substeps)
                else:
                    result = bench(cfg, args.n_steps, args.budget)
            if result is None:
                print(f"Skipping {n_worlds} worlds and higher — projected > {args.budget:.0f}s")
                break
            times, compile_s = result
            m = summarize(times, n_worlds, rate(cfg), steps_per_call)
            print(
                f"[{mode}] {n_worlds} worlds: compile {compile_s:.2f}s, "
                f"avg step {m['avg_step_time_s']:.2e}s, "
                f"FPS {m['fps']:.3e}, real time factor {m['real_time_factor']:.2e}"
            )
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        test_type,
                        1,
                        n_worlds,
                        m["n_steps"],
                        m["total_time_s"],
                        m["avg_step_time_s"],
                        m["fps"],
                        m["real_time_factor"],
                        device.platform,
                    ]
                )

    print("-" * 80)
    print(f"results: {csv_file}")


if __name__ == "__main__":
    main()
