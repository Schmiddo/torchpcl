"""Promote saved benchmark reports into a durable, comparable history.

`runs/` is scratch: every invocation writes a fresh report there and the
directory is ignored, so nothing accumulates. This script copies the rows you
decide to keep into `benchmarks/history/<machine>.jsonl`, one flat row per
result, which is what `dashboard.py` plots.

Promotion is deliberately manual. A benchmark number is only worth keeping when
you know the machine was quiet and the configuration was the standard one; that
is a judgement the runner cannot make. Points are refused when they cannot be
attributed -- a dirty working tree has no commit to hang a measurement on.

Pure stdlib, so this runs in any environment that can read the reports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare import rows as report_rows

DEFAULT_HISTORY = Path(__file__).resolve().parent / "history"
#: Digest -> stable config, so a chart can say what a digest means.
CONFIGS = "configs.json"

#: Config fields that describe *which* subset ran or *where* it ran, rather
#: than what work was done. Two runs differing only in these measure the same
#: thing and belong on the same series. `repeats` is here because taking more
#: samples of one workload does not change the workload; `warmup` is not,
#: because a run without warmup measures a cold kernel and is a different
#: number entirely.
VOLATILE_CONFIG = frozenset({
    "backend", "devices", "requested_device", "open3d_devices",
    "open3d_tensor_devices", "open3d_icp_devices",
    "ops", "methods", "scenes", "repeats", "data",
})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY,
                        help=f"history directory (default: {DEFAULT_HISTORY})")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="promote reports into the history")
    add.add_argument("reports", type=Path, nargs="+")
    add.add_argument("--machine", help="override the report's recorded machine name")
    add.add_argument("--note", help="free-text label stored with every added row")
    add.add_argument("--allow-dirty", action="store_true",
                     help="accept a report measured on a dirty working tree")
    add.add_argument("--replace", action="store_true",
                     help="overwrite points already stored for the same revision")

    listing = commands.add_parser("list", help="summarize what is stored")
    listing.add_argument("--machine")

    drop = commands.add_parser("drop", help="remove every point of one revision")
    drop.add_argument("revision")
    drop.add_argument("--machine")

    return parser.parse_args(argv)


def load_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from error
    if "config" not in report or "environment" not in report:
        raise SystemExit(f"{path}: not a benchmark report "
                         "(expected 'config' and 'environment')")
    return report


def suite_of(report: dict) -> tuple[str, str | None]:
    """Which driver wrote this report, and which workload variant it ran.

    Detected from the report's shape rather than from the recorded command,
    which may be a path, a `-m` invocation or a wrapper.
    """
    config = report["config"]
    if "scenes" in report:
        return "redwood", None
    if "pairs" in report:
        return "registration", None
    if "results" not in report:
        # `accuracy.py` shares the operation config but reports checks, not
        # timings. It answers a different question and has no place on a
        # performance history.
        kind = "an accuracy report" if "checks" in report else "not a timing report"
        raise SystemExit(f"{kind}: it has no measured results to plot")
    if "ops" in config:
        return "operations", config.get("input")
    if "num_samples" in config:
        return "sampling", None
    raise SystemExit("unrecognized report shape: expected scenes, pairs, "
                     "config.ops or config.num_samples")


def stable_config(config: dict) -> dict:
    """The part of a config that defines the workload rather than the run."""
    return {name: value for name, value in sorted(config.items())
            if name not in VOLATILE_CONFIG}


def config_digest(config: dict) -> str:
    """Short hash of the workload-defining configuration.

    Two rows may only share a series when this matches. It is what keeps a
    voxel-size change from being read as a regression: the points land on
    separate charts instead of on one misleading line.
    """
    encoded = json.dumps(stable_config(config), sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def read_configs(directory: Path) -> dict:
    """Digest -> the stable config it stands for, shared by every machine.

    Kept beside the histories rather than repeated on every row: the digest is
    content-addressed, so recording one twice is a no-op, and the dashboard
    needs the values to label a chart with what actually changed.
    """
    path = directory / CONFIGS
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: {error}") from error


def write_configs(directory: Path, configs: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIGS).write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")


def measured(result: dict) -> dict:
    """The timing, memory and quality fields a history row keeps."""
    memory = result.get("memory") or {}
    metrics = {name: value for name, value in (result.get("median") or {}).items()
               if name != "time_ms"}
    return {
        "median_ms": result["median"]["time_ms"],
        "min_ms": result.get("time_ms_min"),
        "max_ms": result.get("time_ms_max"),
        "rss_peak_mib": memory.get("process_rss_peak_mib"),
        "device_peak_mib": memory.get("device_peak_mib"),
        "metrics": metrics,
    }


def redwood_rows(report: dict):
    """Redwood's per-pair rows are far too many to plot; use its own summary.

    One row per backend/device carrying mean per-pair time and the recall that
    the whole benchmark exists to report.
    """
    for summary in report.get("summary", []):
        backend, device = summary["backend"], summary["device"]
        yield ("fgr-pair", backend, device), {
            "backend": backend, "device": device,
            "median_ms": summary["mean_pair_time_ms"],
            "min_ms": None, "max_ms": None,
            "rss_peak_mib": None, "device_peak_mib": None,
            "metrics": {"registration_recall": summary["mean_registration_recall"],
                        "pairs": summary["pairs"]},
        }


def flatten(report: dict, path: Path, *, machine: str | None, note: str | None):
    """One history row per result, carrying everything a chart needs to be honest."""
    environment = report["environment"]
    try:
        suite, variant = suite_of(report)
    except SystemExit as error:
        raise SystemExit(f"{path}: {error}") from error
    name = machine or environment.get("machine")
    if not name:
        raise SystemExit(
            f"{path}: no machine recorded (an older report); pass --machine NAME "
            "so its points are not mixed with another computer's")
    digest = config_digest(report["config"])
    shared = {
        "machine": name,
        "revision": environment.get("revision"),
        "timestamp": environment.get("timestamp"),
        "suite": suite,
        "variant": variant,
        "config_digest": digest,
        "threads": report["config"].get("threads"),
        "dtype": report["config"].get("dtype"),
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "torch": environment.get("torch"),
        "torchpcl": environment.get("torchpcl"),
        "open3d": environment.get("open3d"),
        "small_gicp": environment.get("small_gicp"),
        "gpu": environment.get("gpu"),
        "note": note,
        "source": str(path),
    }
    if suite == "redwood":
        for key, row in redwood_rows(report):
            yield {**shared, "key": list(key), "name": key[0], **row}
        return
    for key, result in report_rows(report).items():
        yield {**shared, "key": list(key), "name": result["name"],
               "backend": result["backend"], "device": result["device"],
               **measured(result)}


def history_path(directory: Path, machine: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in machine)
    return directory / f"{safe}.jsonl"


def read_history(path: Path) -> list[dict]:
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


def write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def identity(row: dict) -> tuple:
    """What makes two stored points the same measurement."""
    return (row["suite"], row.get("variant"), row["config_digest"],
            tuple(row["key"]), row["revision"])


def add(args) -> int:
    added = skipped = replaced = 0
    per_machine: dict[str, list[dict]] = defaultdict(list)
    configs = read_configs(args.history)
    for path in args.reports:
        report = load_report(path)
        configs[config_digest(report["config"])] = stable_config(report["config"])
        revision = report["environment"].get("revision")
        if not revision:
            raise SystemExit(f"{path}: no revision recorded; a point with no "
                             "commit cannot be placed on a history")
        if revision.endswith("-dirty") and not args.allow_dirty:
            raise SystemExit(
                f"{path}: revision {revision} was measured with uncommitted "
                "changes, so it cannot be attributed to a commit. Commit the "
                "tree and re-run, or pass --allow-dirty to keep it anyway.")
        for row in flatten(report, path, machine=args.machine, note=args.note):
            per_machine[row["machine"]].append(row)

    for machine, incoming in per_machine.items():
        path = history_path(args.history, machine)
        stored = read_history(path)
        index = {identity(row): position for position, row in enumerate(stored)}
        for row in incoming:
            position = index.get(identity(row))
            if position is None:
                index[identity(row)] = len(stored)
                stored.append(row)
                added += 1
            elif args.replace:
                stored[position] = row
                replaced += 1
            else:
                skipped += 1
        stored.sort(key=lambda row: (row.get("timestamp") or "", row["suite"],
                                     tuple(row["key"])))
        write_history(path, stored)
        print(f"{path}: {len(stored)} points")
    write_configs(args.history, configs)
    print(f"added {added}, replaced {replaced}, skipped {skipped} already present")
    if skipped and not args.replace:
        print("Re-run with --replace to overwrite the stored points instead.")
    return 0


def histories(directory: Path, machine: str | None) -> list[Path]:
    if machine:
        path = history_path(directory, machine)
        return [path] if path.is_file() else []
    return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []


def listing(args) -> int:
    paths = histories(args.history, args.machine)
    if not paths:
        print(f"No history in {args.history}")
        return 0
    for path in paths:
        rows = read_history(path)
        revisions = {}
        for row in rows:
            revisions.setdefault(row["revision"], set()).add(row["suite"])
        print(f"\n{path.stem}: {len(rows)} points, {len(revisions)} revisions")
        for revision, suites in sorted(
                revisions.items(),
                key=lambda item: min(row["timestamp"] or "" for row in rows
                                     if row["revision"] == item[0])):
            when = min(row["timestamp"] or "" for row in rows
                       if row["revision"] == revision)
            print(f"  {when[:16]:17} {revision:18} {', '.join(sorted(suites))}")
    return 0


def drop(args) -> int:
    removed = 0
    for path in histories(args.history, args.machine):
        stored = read_history(path)
        kept = [row for row in stored if row["revision"] != args.revision]
        removed += len(stored) - len(kept)
        if len(kept) != len(stored):
            write_history(path, kept)
            print(f"{path}: {len(kept)} points")
    print(f"removed {removed} points for {args.revision}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"add": add, "list": listing, "drop": drop}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
