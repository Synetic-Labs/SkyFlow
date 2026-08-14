"""
Side-by-side table from benchmark CSVs sharing main.py's column layout.

Pass any number of result files; each becomes a column set keyed by its test_type.
When exactly two columns exist, a ratio row-by-row column (first ÷ second) is added on
real_time_factor, the rate-independent metric.

    uv run python benchmark/compare.py benchmark/data/*.csv
"""

import argparse
import csv
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int], dict]:
    """{(test_type, n_worlds): row} from one benchmark CSV (last row wins on dupes)."""
    rows: dict[tuple[str, int], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["test_type"], int(row["n_worlds"]))] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("files", nargs="+", type=Path, help="benchmark result CSVs")
    args = ap.parse_args()

    merged: dict[tuple[str, int], dict] = {}
    for path in args.files:
        merged.update(load(path))

    columns = sorted({t for t, _ in merged})  # test_types, e.g. skyflow_env / gym_env
    worlds = sorted({n for _, n in merged})
    ratio = len(columns) == 2

    header = ["n_worlds"]
    for t in columns:
        header += [f"{t} fps", f"{t} rtf"]
    if ratio:
        header.append(f"rtf {columns[0]}/{columns[1]}")
    widths = [max(10, len(h) + 2) for h in header]
    print("".join(h.rjust(w) for h, w in zip(header, widths, strict=True)))

    for n in worlds:
        cells = [str(n)]
        rtfs = []
        for t in columns:
            row = merged.get((t, n))
            rtfs.append(float(row["real_time_factor"]) if row else None)
            cells += (
                [f"{float(row['fps']):.3e}", f"{float(row['real_time_factor']):.3e}"]
                if row
                else ["-", "-"]
            )
        if ratio:
            a, b = rtfs
            cells.append(f"{a / b:.2f}x" if a is not None and b is not None else "-")
        print("".join(c.rjust(w) for c, w in zip(cells, widths, strict=True)))


if __name__ == "__main__":
    main()
