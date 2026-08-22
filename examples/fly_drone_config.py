"""
Drone-config demo — boot the sticks firmware from a Betaflight CLI dump and fly it.

The config source of truth is CLI text: the drone's own `dump all` (Configurator →
CLI tab → `dump all` → save). `SimConfig.eeprom` points at that file; the env renders
it into the boot eeprom at construction through cudaflight's version-gated strict
round-trip, so a dump from another firmware release fails at construction instead of
silently flying stock defaults (examples/configs/README.md). This demo flies the
repo's stock dump — the pinned firmware's own defaults — so it is self-contained;
point --dump at a real drone's dump (plus its sim_overrides.txt) to fly that config.

Flight plan: open-loop AETR sticks on the nominal plant (no DR, no policy). The fleet
arms during creation (armed snapshot), so throttle is live from the first step: climb
one second, then ease toward hover and read the altitude.

Run from the repo root (CPU firmware — no GPU needed):

    uv run python examples/fly_drone_config.py
    uv run python examples/fly_drone_config.py --dump my_drone_dump.txt \
        --overrides my_sim_overrides.txt
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from skyflow import DomainRand, SimConfig, SkyFlowEnv

CLIMB_STICKS = (0.0, 0.0, 0.60, 0.0)  # AETR: level, throttle above hover
HOLD_STICKS = (0.0, 0.0, 0.45, 0.0)  # AETR: level, near-hover throttle
CLIMB_S = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    default_dump = Path(__file__).parent / "configs" / "stock_dump.txt"
    ap.add_argument("--dump", type=Path, default=default_dump,
                    help="Betaflight CLI `dump all` file (default: the stock dump)")
    ap.add_argument("--overrides", type=Path, default=None,
                    help="optional sim-only CLI lines appended after the dump")
    ap.add_argument("--fleet", type=int, default=1,
                    help="CPU SITL instances (each adds ~7 s boot)")
    ap.add_argument("--seconds", type=float, default=2.0, help="flight time, s")
    ap.add_argument("--seed", type=int, default=0, help="reset PRNG seed")
    args = ap.parse_args()

    cfg = SimConfig(
        num_envs=args.fleet,
        task="hover",
        task_kwargs={"goal_hold_s": 60.0},
        control="sticks",
        firmware="cpu",  # self-contained: any fleet size, no CUDA needed
        eeprom=str(args.dump),
        eeprom_overrides=str(args.overrides) if args.overrides else None,
        dr=DomainRand().off(),  # nominal vehicle: this demo shows the config seam
    )
    print(f"rendering {args.dump.name} (version-gated) and booting {args.fleet} instance(s)…")
    env = SkyFlowEnv(cfg)
    print(f"  version gate passed — boot image: {env.eeprom_image}")

    _obs, state = env.reset(jax.random.PRNGKey(args.seed))
    jstep = jax.jit(env.step)
    n_steps = round(args.seconds * cfg.control_hz)
    n_climb = round(CLIMB_S * cfg.control_hz)
    climb = jnp.tile(jnp.asarray(CLIMB_STICKS, jnp.float32), (args.fleet, 1))
    hold = jnp.tile(jnp.asarray(HOLD_STICKS, jnp.float32), (args.fleet, 1))

    z_max = 0.0
    for t in range(1, n_steps + 1):
        _obs, state, _reward, done, _info = jstep(state, climb if t <= n_climb else hold)
        z = np.asarray(state.plant[:, 2])
        z_max = max(z_max, float(z.max()))
        if bool(done.any()):
            # after an auto-reset the firmware re-arms only on LOW throttle — this
            # open-loop plan never lowers it, so stop instead of flying a dead world
            print(f"  done fired at step {t} — stopping (episode cull or task end)")
            break
        if t % int(cfg.control_hz / 2) == 0:
            print(f"  t={t / cfg.control_hz:4.1f} s  z mean {z.mean():.2f} m  max {z.max():.2f} m")

    print(f"peak altitude: {z_max:.2f} m")
    if z_max < 0.05:
        raise SystemExit("firmware never lifted — check the dump and overrides")
    print("the fleet flew the rendered config — seam OK")


if __name__ == "__main__":
    main()
