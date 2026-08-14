# Benchmarks

Fleet-throughput benchmarks: 1000 timed calls per fleet size, every call synchronized
with `block_until_ready`, probe-and-skip past a per-size time budget, and CSV output
with a stable schema so runs from different simulators, machines, and configs tabulate
together.

## Run

```bash
uv run python benchmark/main.py --device gpu --worlds 64,1024,16384,65536,262144
uv run python benchmark/main.py --device cpu --mode env --worlds 16,256 --n-steps 200   # quick check
```

`start-bench.sh` wraps the GPU sweep plus the comparison table.

Two modes per run (`--mode both` is the default):

- **`env`** — the full jitted `env.step`: 10 RK4 substeps, wind/poke draws, transport
  delay, observation, reward, termination, in-jit auto-reset. What a training loop pays.
- **`dynamics`** — bare RK4 substeps of the plant, nothing else: the raw-integrator
  unit that simulator "steps/s" headlines usually quote. Each timed call scans
  `--substeps` (default 100) steps in one dispatch and the metrics divide back down:
  per-call submission latency (~1.3 ms under WSL2, vs ~50 µs native Linux) otherwise
  caps throughput at `n_worlds / latency` no matter how fast the GPU is.
  `--substeps 1` times one dispatch per step instead.

Each size is a fresh env + jit; compile time is printed but kept out of the timed
region. Results append to `benchmark/data/skyflow_<timestamp>.csv`.

## Compare

```bash
uv run python benchmark/compare.py benchmark/data/*.csv
```

`compare.py` tabulates any CSVs in the shared schema side by side, keyed by
`test_type`. `benchmark/data/` is gitignored, so baseline CSVs transcribed from other
simulators' published tables (or measured on other machines) stay local.

## Reading the numbers

- **Match workloads.** `dynamics` rows time bare plant integration; `env` rows carry
  the whole RL pipeline. Quoting one against the other conflates simulator speed with
  workload.
- **Mind the step units.** `fps` counts physics steps in dynamics mode and control
  steps in env mode (each control step hides `physics_hz / control_hz` = 10 substeps
  by default). `real_time_factor` (simulated seconds per wall-clock second) is the
  rate-independent column — but a simulator with a larger physics dt needs fewer steps
  for the same simulated time, so quote steps/s for integrator comparisons and RTF for
  wall-clock claims.
- **Watch for dispatch-bound rows.** If per-call time stays flat as `n_worlds` grows,
  the row measures call-submission latency, not compute — batch more work per dispatch
  (`--substeps`) before drawing conclusions. This matters especially under WSL2.
