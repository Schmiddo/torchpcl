"""Measure every routine profile, and optionally keep the results.

    python -m benchmarks.bench [--keep] [--profile NAME ...] [run options]
    python -m benchmarks.bench --baselines [--keep]   # in the baseline env

One command per revision: each profile runs in a fresh process -- so one
suite's allocations never inflate the next one's peak memory -- and writes its
report into one run directory under `runs/`. `--keep` promotes those reports
into the history. Keeping needs a committed tree, which is checked before
anything is measured rather than after the sweep has spent its time.

`--baselines` measures every third-party backend of each profile instead of
torchpcl. A baseline is a reference value per library version, so it needs
refreshing only when that library, the machine or a profile changes, not
on every commit. Options after the known ones (`--device cuda`,
`--repeats 9`) reach every run; anything a profile pins is rejected there.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import history
from benchmarks.cli import source_revision
from benchmarks.profiles import PROFILES
from benchmarks.suites import SUITES

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", action="append", choices=sorted(PROFILES),
                        dest="profiles",
                        help="run only this profile (repeatable); default: every "
                             "routine profile")
    parser.add_argument("--baselines", action="store_true",
                        help="measure the third-party backends instead of torchpcl")
    parser.add_argument("--keep", action="store_true",
                        help="promote the reports into the history")
    parser.add_argument("--history", type=Path, default=history.DEFAULT_HISTORY)
    parser.add_argument("--out", type=Path,
                        help="report directory (default: runs/<UTC timestamp>)")
    return parser.parse_known_args(argv)


def plan(names: list[str] | None, baselines: bool) -> list[tuple]:
    """(profile, backend) pairs to run, in order."""
    profiles = ([PROFILES[name] for name in names] if names
                else [profile for profile in PROFILES.values() if profile.routine])
    return [(profile, backend) for profile in profiles
            for backend in SUITES[profile.suite].backends
            if (backend != "torchpcl") == baselines]


def main(argv=None) -> int:
    args, extra = parse_args(argv)
    if args.keep:
        revision = source_revision()
        if revision is None or revision.endswith("-dirty"):
            raise SystemExit(
                f"--keep needs a committed tree (revision: {revision}); commit "
                "first, or run without --keep to measure without keeping")
    out = (args.out or ROOT / "runs" /
           datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    reports, failed, skipped = [], [], []
    for profile, backend in plan(args.profiles, args.baselines):
        name = f"{profile.name} · {backend}"
        missing = profile.missing(ROOT)
        if missing:
            print(f"\n=== {name}: skipped, missing {missing[0]} "
                  "(`just download-data`) ===", flush=True)
            skipped.append(name)
            continue
        print(f"\n=== {name} ===", flush=True)
        path = out / f"{profile.name}-{backend}.json"
        command = [sys.executable, "-m", "benchmarks.run", "--profile", profile.name,
                   "--backend", backend, "--json", str(path), *extra]
        # One failing leg must not lose the legs that already measured.
        if subprocess.run(command, cwd=ROOT).returncode == 0:
            reports.append(path)
        else:
            failed.append(name)

    print(f"\n{len(reports)} reports in {out}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    if args.keep and reports:
        history.add(reports, args.history)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
