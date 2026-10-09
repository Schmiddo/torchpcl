"""Run one benchmark suite and print, and optionally save, its report.

    python -m benchmarks.run <suite> [options]      # ad hoc, any parameters
    python -m benchmarks.run --profile NAME [...]   # pinned, keepable workload

Suites: operations, sampling, registration, redwood. `<suite> --help` lists a
suite's own options. A profile (see `profiles.py`) fixes every workload
argument; it still accepts --backend, --device, --repeats, --json and the
suite's selection options, and rejects anything that would change the work.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Preserve direct-script usage as well as python -m benchmarks.run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.case import select_devices
from benchmarks.cli import (
    SMALL_GICP_CPU_ONLY,
    configure_threads,
    default_threads,
    environment_metadata,
    format_optional,
    positive_int,
    require_open3d,
    require_small_gicp,
    write_json,
)
from benchmarks.profiles import PROFILES, normalize
from benchmarks.runner import measure
from benchmarks.suites import SUITES

#: Runner arguments that select what runs or where output goes. Every other
#: argument, the runner's `--threads`/`--warmup` included, defines the work.
SELECTION = ("profile", "backend", "device", "repeats", "json")


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_parser(suite, profile=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"run.py {suite.name}", description=suite.__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        help="pin the workload to a named profile a history can keep")
    parser.add_argument("--backend", choices=suite.backends, default="torchpcl")
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all",
                        help="all measures CPU and, when available, CUDA; an "
                             "explicit cuda fails without it")
    parser.add_argument("--repeats", type=positive_int, default=suite.repeats)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads(),
                        help="CPU threads (default: physical core count)")
    parser.add_argument("--json", type=Path, help="write the report to PATH")
    suite.add_arguments(parser)
    if profile is not None:
        parser.set_defaults(**profile.params)
    return parser


def pinned(parser, suite) -> set[str]:
    dests = {action.dest for action in parser._actions if action.dest != "help"}
    return dests - set(SELECTION) - set(suite.selection)


def check_profile(parser, suite, profile, args) -> None:
    """A profile names every workload argument, and the run changed none of them."""
    names = pinned(parser, suite)
    if names != profile.params.keys():
        raise SystemExit(
            f"profile {profile.name} is out of date with the {suite.name} suite: "
            f"unpinned {sorted(names - profile.params.keys())}, "
            f"unknown {sorted(profile.params.keys() - names)}")
    for name in sorted(names):
        if normalize(getattr(args, name)) != normalize(profile.params[name]):
            parser.error(f"--{name.replace('_', '-')} is pinned to "
                         f"{profile.params[name]!r} by profile {profile.label}; "
                         "run without --profile to change it")


def parse_args(argv=None):
    """The suite, its profile if any, and the parsed arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES))
    known, rest = pre.parse_known_args(argv)
    profile = PROFILES[known.profile] if known.profile else None
    if rest and rest[0] in SUITES:
        name, rest = rest[0], rest[1:]
        if profile is not None and profile.suite != name:
            raise SystemExit(f"profile {profile.name} belongs to the "
                             f"{profile.suite} suite, not {name}")
    elif profile is not None:
        name = profile.suite
    else:
        print(__doc__.strip(), file=sys.stderr)
        print(f"\nprofiles: {', '.join(sorted(PROFILES))}", file=sys.stderr)
        raise SystemExit(0 if {"-h", "--help"} & set(rest) else 2)
    suite = SUITES[name]
    parser = build_parser(suite, profile)
    args = parser.parse_args(rest + (["--profile", profile.name] if profile else []))
    suite.resolve(args, parser)
    if profile is not None:
        check_profile(parser, suite, profile, args)
    return suite, profile, args


def backend_support(backend: str) -> tuple[bool, str | None]:
    """Import the backend, or exit with setup guidance; report its CUDA support."""
    if backend == "torchpcl":
        import torch
        from torchpcl._backend import has_cuda

        return torch.cuda.is_available() and has_cuda(), None
    if backend == "small-gicp":
        require_small_gicp()
        return False, SMALL_GICP_CPU_ONLY
    return require_open3d().core.cuda.is_available(), None


class Table:
    """One console row per measured case, the header printed on the first."""

    def __init__(self, columns):
        self.columns = columns
        self.started = False

    def row(self, result: dict) -> None:
        extra = "".join(f" {header:>10}" for header, _, _ in self.columns)
        if not self.started:
            print(f"\n{'case':34} {'backend/device':22} {'median ms':>10} "
                  f"{'min ms':>10} {'RAM MiB':>9} {'VRAM MiB':>9}{extra}")
            self.started = True
        label = "/".join(filter(None, (result.get("group"), result["name"])))
        memory, median = result["memory"], result["median"]
        values = "".join(
            f" {format_optional(median.get(metric), fmt):>10}"
            for _, metric, fmt in self.columns)
        print(f"{label:34} {result['backend'] + '/' + result['device']:22} "
              f"{median['time_ms']:10.3f} {result['time_ms_min']:10.3f} "
              f"{memory['process_rss_peak_mib']:9.1f} "
              f"{format_optional(memory['device_peak_mib']):>9}{values}", flush=True)


def main(argv=None):
    suite, profile, args = parse_args(argv)
    configure_threads(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    cuda_available, reason = backend_support(args.backend)
    if args.backend in suite.cpu_only:
        cuda_available, reason = False, suite.cpu_only[args.backend]
    try:
        devices, skipped = select_devices(args.device, cuda_available,
                                          backend=args.backend, reason=reason)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    report = {
        "environment": environment_metadata(devices),
        "suite": suite.name,
        "profile": None if profile is None else {
            "name": profile.name, "version": profile.version,
            "params": normalize(profile.params)},
        "config": {**{name: normalize(value) for name, value in sorted(vars(args).items())
                      if name not in ("json", "profile")}, "devices": devices},
        "input": {},
        "skipped": skipped,
        "results": [],
    }
    print(f"{suite.name}"
          + (f" · profile {profile.label}" if profile else " · ad hoc")
          + f" · {args.backend} on {', '.join(devices)} · threads {args.threads}")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")

    table = Table(suite.columns(args))
    for item in suite.cases(args, devices, report):
        if isinstance(item, dict):
            skipped.append(item)
            what = item.get("op") or item.get("method") or ""
            print(f"Skipped {item['backend']}/{item['device']} {what}: {item['reason']}")
            continue
        result = measure(item, repeats=args.repeats, warmup=args.warmup,
                         evaluate=suite.evaluate)
        del item  # free device memory before the next case is built
        report["results"].append(result)
        table.row(result)
    if not report["results"]:
        raise SystemExit("nothing was measured; see the skips above")
    suite.finish(report)

    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
