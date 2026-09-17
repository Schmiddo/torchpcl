"""Diff two saved benchmark reports: `python benchmarks/compare.py old.json new.json`.

Pure stdlib — no torch, no Open3D — so a comparison runs in any environment
that can read the files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Heuristic floor for flagging movement, not a statistical significance test.
NOISE = 0.02


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--backend", help="restrict to one backend (e.g. torchpcl)")
    parser.add_argument("--before-backend", help="backend selected from the baseline report")
    parser.add_argument("--after-backend", help="backend selected from the candidate report")
    args = parser.parse_args(argv)
    if bool(args.before_backend) != bool(args.after_backend):
        parser.error("--before-backend and --after-backend must be supplied together")
    if args.backend and args.before_backend:
        parser.error("use --backend or the before/after selectors, not both")
    return args


def load(path: Path) -> dict:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from error
    if "config" not in report or not ("results" in report or "pairs" in report):
        raise SystemExit(f"{path}: not a benchmark report "
                         "(expected 'config' and 'results' or 'pairs')")
    return report


def rows(report: dict) -> dict[tuple[str, ...], dict]:
    """Flatten a report to case key -> result.

    The registration benchmark nests its rows under one entry per input pair;
    every other script reports a flat list. Pair rows are keyed by the
    directory name, matching the label that script prints.
    """
    if "results" in report:
        return unique((key(result), result) for result in report["results"])
    return unique(
        ((Path(pair["directory"]).name, *key(result)), result)
        for pair in report["pairs"] for result in pair["results"]
    )


def unique(entries) -> dict[tuple[str, ...], dict]:
    """Index rows by key, refusing to let one silently replace another.

    No current driver can emit a repeated key: one invocation runs one
    configuration, and its operations, backend and devices are each distinct.
    A sweep would break that -- several rows sharing a name and differing only
    in a parameter -- and the comparison would then quietly report whichever
    row happened to land last. That is a wrong number rather than an error, so
    it is rejected here. A sweep needs the key to carry the parameter that
    varies, not a relaxation of this check.
    """
    indexed = {}
    for entry, result in entries:
        if entry in indexed:
            raise SystemExit(f"duplicate row {'/'.join(entry)}: a report must "
                             "identify each row uniquely")
        indexed[entry] = result
    return indexed


def key(result: dict) -> tuple[str, str, str]:
    return result["name"], result["backend"], result["device"]


def config_differences(before: dict, after: dict) -> list[str]:
    """Parameters that changed, which would invalidate the comparison."""
    return sorted(
        f"{name}: {before['config'].get(name)!r} -> {after['config'].get(name)!r}"
        for name in set(before["config"]) | set(after["config"])
        if before["config"].get(name) != after["config"].get(name)
    )


def separated(before: dict, after: dict) -> bool:
    """True when the runs' measured ranges do not overlap.

    A median moving by more than the spread of either run is the weakest claim
    that survives scrutiny; anything inside the spread is indistinguishable
    from clock drift.
    """
    return (before["time_ms_max"] < after["time_ms_min"]
            or after["time_ms_max"] < before["time_ms_min"])


def metric_changes(before: dict, after: dict) -> list[str]:
    """Accuracy gates that moved, so a speedup that broke a result is visible."""
    changes = []
    for name, old in before["median"].items():
        if name == "time_ms":
            continue
        new = after["median"].get(name)
        if new is not None and old != new:
            changes.append(f"{name} {old:.6g} -> {new:.6g}")
    return changes


def describe(report: dict) -> str:
    environment = report.get("environment", {})
    return (f"{environment.get('timestamp', 'unknown time')}  "
            f"{environment.get('revision') or 'no revision'}")


def print_environment(label: str, report: dict) -> None:
    environment = report.get("environment", {})
    fields = ("machine", "platform", "gpu", "python", "torch", "torchpcl",
              "open3d", "small_gicp")
    details = "; ".join(f"{field}={environment[field]}" for field in fields
                        if environment.get(field) is not None)
    print(f"{label} environment: {details or 'unknown'}")


def selected_rows(report: dict, backend: str) -> dict:
    """Drop backend from matching keys only after selecting one explicitly."""
    selected = {k[:-2] + (k[-1],): row for k, row in rows(report).items()
                if row["backend"] == backend}
    if not selected:
        raise SystemExit(f"no results for backend {backend!r}")
    return selected


def comparison_notes(old: dict, new: dict) -> list[str]:
    """Existing workload caveats, without inferring automatic compatibility."""
    notes = []
    name, device = old["name"], old["device"]
    different_backends = old["backend"] != new["backend"]
    if different_backends and name == "voxelize":
        notes.append("bare voxelization excludes centroid reduction; compare voxelize-reduce")
    if different_backends and name in ("index-build", "knn") and device == "cuda":
        notes.append("CUDA index-build work is split differently; compare knn-build-query")
    if different_backends and name == "radius":
        notes.append("capped torchpcl vs unbounded Open3D radius search; sampled cap check only")
    if "open3d-legacy" in (old["backend"], new["backend"]):
        notes.append("Open3D legacy computes in float64")
    if ("small-gicp" in (old["backend"], new["backend"]) and different_backends
            and "point-to-plane" in (old.get("objective"), new.get("objective"))):
        notes.append("small-gicp estimates its own normals; point-to-plane "
                     "compares two normal implementations, not one shared array")
    for field in ("input_dtype", "compute_dtype", "transform_dtype"):
        if old.get(field) != new.get(field):
            notes.append(f"{field}: {old.get(field, 'unknown')} -> {new.get(field, 'unknown')}")
    return notes


def main(argv=None):
    args = parse_args(argv)
    before, after = load(args.before), load(args.after)

    print(f"before  {args.before}  {describe(before)}")
    print(f"after   {args.after}  {describe(after)}")
    print_environment("before", before)
    print_environment("after", after)
    for difference in config_differences(before, after):
        print(f"! config {difference}")

    cross_backend = args.before_backend is not None
    if cross_backend:
        old_rows = selected_rows(before, args.before_backend)
        new_rows = selected_rows(after, args.after_backend)
        print(f"Backends: {args.before_backend} -> {args.after_backend}")
        print("Ratios require matching workloads and hardware; config differences are not validated.")
    else:
        old_rows, new_rows = rows(before), rows(after)
    if args.backend:
        old_rows = {k: v for k, v in old_rows.items() if v["backend"] == args.backend}
        new_rows = {k: v for k, v in new_rows.items() if v["backend"] == args.backend}

    column = "speedup" if cross_backend else "change"
    print(f"\n{'case':40} {'before ms':>10} {'after ms':>10} {column:>9}")
    for name in sorted(old_rows.keys() & new_rows.keys()):
        old, new = old_rows[name], new_rows[name]
        old_ms, new_ms = old["median"]["time_ms"], new["median"]["time_ms"]
        if cross_backend:
            speedup = f"{old_ms / new_ms:.2f}x" if new_ms else "n/a"
            print(f"{'/'.join(name):40} {old_ms:10.3f} {new_ms:10.3f} {speedup:>9}")
            print(f"{'':40}   range ms: {old['time_ms_min']:.3f}–{old['time_ms_max']:.3f}"
                  f" -> {new['time_ms_min']:.3f}–{new['time_ms_max']:.3f}")
            for note in comparison_notes(old, new):
                print(f"{'':40}   note: {note}")
        else:
            ratio = new_ms / old_ms - 1 if old_ms else float("nan")
            mark = "*" if abs(ratio) > NOISE and separated(old, new) else " "
            print(f"{'/'.join(name):40} {old_ms:10.3f} {new_ms:10.3f} "
                  f"{100 * ratio:+8.1f}%{mark}")
            for change in metric_changes(old, new):
                print(f"{'':40}   gate: {change}")

    for name in sorted(old_rows.keys() - new_rows.keys()):
        print(f"{'/'.join(name):40} only in {args.before}")
    for name in sorted(new_rows.keys() - old_rows.keys()):
        print(f"{'/'.join(name):40} only in {args.after}")
    if cross_backend:
        print("\nSpeedup = before / after; greater than 1 means the candidate is faster.")
    else:
        print("\n* change exceeds 2% and both runs' measured spread (heuristic).")


if __name__ == "__main__":
    main()
