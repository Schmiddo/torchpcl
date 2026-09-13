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

# Relative movement below this is reported but never marked significant; it is
# the floor of what repeated runs of an untouched kernel reproduce.
NOISE = 0.02


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--backend", help="restrict to one backend (e.g. torchpcl)")
    args = parser.parse_args(argv)
    return args


def load(path: Path) -> dict:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from error
    if "results" not in report or "config" not in report:
        raise SystemExit(f"{path}: not a benchmark report "
                         "(expected 'config' and 'results')")
    return report


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


def main(argv=None):
    args = parse_args(argv)
    before, after = load(args.before), load(args.after)

    print(f"before  {args.before}  {describe(before)}")
    print(f"after   {args.after}  {describe(after)}")
    for difference in config_differences(before, after):
        print(f"! config {difference}")

    old_rows = {key(result): result for result in before["results"]}
    new_rows = {key(result): result for result in after["results"]}
    if args.backend:
        old_rows = {k: v for k, v in old_rows.items() if k[1] == args.backend}
        new_rows = {k: v for k, v in new_rows.items() if k[1] == args.backend}

    print(f"\n{'case':40} {'before ms':>10} {'after ms':>10} {'change':>9}")
    for name in sorted(old_rows.keys() & new_rows.keys()):
        old, new = old_rows[name], new_rows[name]
        old_ms, new_ms = old["median"]["time_ms"], new["median"]["time_ms"]
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
    print("\n* change exceeds both runs' measured spread; unmarked rows are "
          "within noise.")


if __name__ == "__main__":
    main()
