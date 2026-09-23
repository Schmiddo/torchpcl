"""Keep profile runs as a durable history: `benchmarks/history/<machine>/`.

`runs/` is scratch: every invocation writes a fresh report there and the
directory is ignored. This keeps what `bench.py --keep` (or `add`) promotes,
split by what the numbers are:

- `torchpcl.jsonl` -- one row per case per kept revision. These are the series
  the dashboard plots, keyed on profile name and version.
- `baselines.jsonl` -- one row per case per third-party library version. A
  baseline does not change with torchpcl's commits, so it is a reference
  value, not a series; re-measuring the same library version replaces it.
- `profiles.json` -- the pinned parameters each profile version was first
  kept with, so a profile edited without a version bump is refused rather
  than silently bending its series.

The machine is a label describing the hardware, since timings only compare
within one computer. Only profile runs of a committed tree are kept: an ad hoc
run has no series to join, and a dirty tree has no commit to hang a point on.

Pure stdlib, so this runs in any environment that can read the reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare import rows as report_rows

DEFAULT_HISTORY = Path(__file__).resolve().parent / "history"
SERIES = "torchpcl.jsonl"
BASELINES = "baselines.jsonl"
PROFILES = "profiles.json"
#: Which environment field versions each baseline backend.
BASELINE_LIBRARY = {"open3d": "open3d", "open3d-tensor": "open3d",
                    "open3d-legacy": "open3d", "small-gicp": "small_gicp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY,
                        help=f"history directory (default: {DEFAULT_HISTORY})")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="promote profile reports into the history")
    add.add_argument("reports", type=Path, nargs="+")
    add.add_argument("--replace", action="store_true",
                     help="overwrite torchpcl points already kept for the same revision")

    listing = commands.add_parser("list", help="summarize what is stored")
    listing.add_argument("--machine")

    drop = commands.add_parser("drop", help="remove every torchpcl point of one revision")
    drop.add_argument("revision")
    drop.add_argument("--machine")

    relabel = commands.add_parser("relabel", help="rename a stored machine")
    relabel.add_argument("old")
    relabel.add_argument("new")
    return parser.parse_args(argv)


def load_report(path: Path) -> dict:
    """A keepable report, or an exit saying why this one is not."""
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from error
    if "checks" in report:
        raise SystemExit(f"{path}: an accuracy report measures agreement, not "
                         "speed, and has no place on a performance history")
    if "results" not in report or "environment" not in report:
        raise SystemExit(f"{path}: not a benchmark report")
    if not report.get("profile"):
        raise SystemExit(f"{path}: an ad hoc run; only --profile runs are kept, "
                         "because only they measure the same work on every revision")
    revision = report["environment"].get("revision")
    if not revision:
        raise SystemExit(f"{path}: no revision recorded; a point with no commit "
                         "cannot be placed on a history")
    if revision.endswith("-dirty"):
        raise SystemExit(f"{path}: revision {revision} was measured with uncommitted "
                         "changes, so it cannot be attributed to a commit. Commit "
                         "and re-run.")
    return report


def check_profile(profiles: dict, report: dict, path: Path) -> None:
    """Refuse a profile version whose pinned values moved since it was first kept.

    Parameters the stored copy lacks are accepted and recorded: a newly added
    argument pinned at the value it always implicitly had is the same work.
    """
    profile = report["profile"]
    label = f"{profile['name']}@{profile['version']}"
    stored = profiles.setdefault(label, {})
    changed = sorted(name for name, value in stored.items()
                     if profile["params"].get(name, object()) != value)
    if changed:
        raise SystemExit(
            f"{path}: profile {label} now pins {', '.join(changed)} differently from "
            "the points already kept. Bump its version in benchmarks/profiles.py.")
    stored.update(profile["params"])


def measured(result: dict) -> dict:
    """The timing, memory and quality fields a history row keeps."""
    memory = result.get("memory") or {}
    return {
        "median_ms": result["median"]["time_ms"],
        "min_ms": result.get("time_ms_min"),
        "max_ms": result.get("time_ms_max"),
        "rss_peak_mib": memory.get("process_rss_peak_mib"),
        "device_peak_mib": memory.get("device_peak_mib"),
        "metrics": {name: value for name, value in result["median"].items()
                    if name != "time_ms"},
    }


def redwood_rows(report: dict):
    """Redwood's per-pair rows are far too many to plot; use its own summary:
    mean per-pair time and the recall the whole benchmark exists to report."""
    for summary in report.get("summary", []):
        yield {"key": ["fgr-pair", summary["backend"], summary["device"]],
               "name": "fgr-pair", "group": None,
               "backend": summary["backend"], "device": summary["device"],
               "median_ms": summary["mean_pair_time_ms"],
               "min_ms": None, "max_ms": None,
               "rss_peak_mib": None, "device_peak_mib": None,
               "metrics": {"registration_recall": summary["mean_registration_recall"],
                           "pairs": summary["pairs"]}}


def flatten(report: dict, path: Path):
    """One history row per result, carrying everything a chart needs to be honest."""
    environment, profile = report["environment"], report["profile"]
    shared = {
        "machine": environment["machine"],
        "profile": profile["name"], "profile_version": profile["version"],
        "suite": report["suite"],
        "revision": environment["revision"], "timestamp": environment.get("timestamp"),
        "gpu": environment.get("gpu"), "platform": environment.get("platform"),
        "versions": {name: environment.get(name) for name in
                     ("python", "torch", "torchpcl", "open3d", "small_gicp")},
        "source": str(path),
    }
    if report["suite"] == "redwood":
        entries = list(redwood_rows(report))
    else:
        entries = [{"key": list(key), "name": result["name"],
                    "group": result.get("group"), "backend": result["backend"],
                    "device": result["device"], **measured(result)}
                   for key, result in report_rows(report).items()]
    for entry in entries:
        row = {**shared, **entry}
        library = BASELINE_LIBRARY.get(row["backend"])
        if library is not None:
            row["baseline_version"] = environment.get(library)
        yield row


def machine_dir(directory: Path, machine: str) -> Path:
    return directory / "".join(c if c.isalnum() or c in "-._" else "-" for c in machine)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: {error}") from error
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row.get("timestamp") or "", row["profile"],
                                         row["key"]))
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_profiles(directory: Path) -> dict:
    path = directory / PROFILES
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: {error}") from error


def identity(row: dict) -> tuple:
    """What makes two stored rows the same measurement.

    A torchpcl point belongs to a revision; a baseline to its library version,
    whatever torchpcl revision happened to be checked out when it was taken.
    """
    base = (row["profile"], row["profile_version"], tuple(row["key"]))
    if "baseline_version" in row:
        return base + (row["baseline_version"],)
    return base + (row["revision"],)


def add(paths: list[Path], directory: Path = DEFAULT_HISTORY, *,
        replace: bool = False) -> int:
    """Promote reports; validates all of them before writing any."""
    profiles = read_profiles(directory)
    incoming: dict[Path, list[dict]] = {}
    for path in paths:
        report = load_report(path)
        check_profile(profiles, report, path)
        for row in flatten(report, path):
            target = machine_dir(directory, row["machine"])
            target /= BASELINES if "baseline_version" in row else SERIES
            incoming.setdefault(target, []).append(row)

    counts = {"added": 0, "replaced": 0, "skipped": 0}
    for path, rows in incoming.items():
        stored = read_rows(path)
        index = {identity(row): position for position, row in enumerate(stored)}
        for row in rows:
            position = index.get(identity(row))
            if position is None:
                index[identity(row)] = len(stored)
                stored.append(row)
                counts["added"] += 1
            elif replace or path.name == BASELINES:
                # A baseline re-measured at the same library version is a
                # refresh: the newest measurement is the reference.
                stored[position] = row
                counts["replaced"] += 1
            else:
                counts["skipped"] += 1
        write_rows(path, stored)
        print(f"{path}: {len(stored)} rows")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROFILES).write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n")
    print("added {added}, replaced {replaced}, skipped {skipped} already kept".format(**counts))
    if counts["skipped"]:
        print("Pass --replace to overwrite the kept torchpcl points instead.")
    return 0


def machines(directory: Path, machine: str | None = None) -> list[Path]:
    if machine:
        path = machine_dir(directory, machine)
        return [path] if path.is_dir() else []
    return sorted(path for path in directory.glob("*/") if path.is_dir()) \
        if directory.is_dir() else []


def listing(args) -> int:
    paths = machines(args.history, args.machine)
    if not paths:
        print(f"No history in {args.history}")
        return 0
    for path in paths:
        series = read_rows(path / SERIES)
        revisions: dict[str, tuple[str, set]] = {}
        for row in series:
            when, profiles = revisions.get(row["revision"], (row["timestamp"] or "", set()))
            profiles.add(row["profile"])
            revisions[row["revision"]] = (min(when, row["timestamp"] or ""), profiles)
        print(f"\n{path.name}: {len(series)} torchpcl points, {len(revisions)} revisions")
        for revision, (when, profiles) in sorted(revisions.items(), key=lambda item: item[1][0]):
            print(f"  {when[:16]:17} {revision:12} {', '.join(sorted(profiles))}")
        baselines = sorted({(row["backend"], row["baseline_version"], row["profile"])
                            for row in read_rows(path / BASELINES)})
        for backend, version, profile in baselines:
            print(f"  baseline {backend} {version} · {profile}")
    return 0


def drop(args) -> int:
    removed = 0
    for path in machines(args.history, args.machine):
        stored = read_rows(path / SERIES)
        kept = [row for row in stored if row["revision"] != args.revision]
        removed += len(stored) - len(kept)
        if len(kept) != len(stored):
            write_rows(path / SERIES, kept)
    print(f"removed {removed} points for {args.revision}")
    return 0


def relabel(args) -> int:
    """Move one machine's rows under a different label.

    Merging into an existing label is refused: two computers silently joined
    onto one series is the failure the label exists to prevent.
    """
    source, target = machine_dir(args.history, args.old), machine_dir(args.history, args.new)
    if not source.is_dir():
        raise SystemExit(f"{source}: no history stored for {args.old!r}")
    if target.exists():
        raise SystemExit(f"{target}: already holds points; merging two machines "
                         "onto one series would compare different hardware")
    target.mkdir(parents=True)
    for name in (SERIES, BASELINES):
        rows = read_rows(source / name)
        for row in rows:
            row["machine"] = args.new
        if rows:
            write_rows(target / name, rows)
        (source / name).unlink(missing_ok=True)
    source.rmdir()
    print(f"{target}: relabelled {args.old!r} -> {args.new!r}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "add":
        return add(args.reports, args.history, replace=args.replace)
    return {"list": listing, "drop": drop, "relabel": relabel}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
