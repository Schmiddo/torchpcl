"""Command-line benchmark: torchpcl's per-operation kernels vs. Open3D.

One script rather than one per operation: the operations share every argument
and differ only in which cases they build, so a single `--op` selector avoids
repeating the reporting loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Preserve direct-script usage as well as python -m benchmarks.operations.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import workload
from benchmarks.case import select_devices
from benchmarks.cli import (
    configure_threads,
    default_threads,
    environment_metadata,
    format_optional,
    positive_int,
    require_open3d,
    write_json,
)
from benchmarks import features_cases as features
from benchmarks import neighbors_cases as neighbors
from benchmarks import preprocess_cases as preprocess
from benchmarks.runner import measure

ALL_OPS = workload.ALL_OPS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torchpcl", "open3d"], default="torchpcl")
    parser.add_argument("--op", choices=[*ALL_OPS, "all"], default="all")
    workload.add_arguments(parser)
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads(),
                        help="CPU threads (default: physical core count)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    workload.resolve_defaults(args, parser)
    return args


def _print_table(op, rows):
    """One performance table per operation."""
    print(f"\n{op}")
    print(f"{'backend/device':24} {'median ms':>10} {'min ms':>10} "
          f"{'RAM MiB':>10} {'VRAM MiB':>10}")
    for result in rows:
        memory = result["memory"]
        print(f"{result['backend'] + '/' + result['device']:24} "
              f"{result['median']['time_ms']:10.3f} "
              f"{result['time_ms_min']:10.3f} "
              f"{memory['process_rss_peak_mib']:10.1f} "
              f"{format_optional(memory['device_peak_mib']):>10}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    o3d = require_open3d() if args.backend == "open3d" else None
    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "torchpcl":
        from torchpcl import _C

        cuda_available = torch.cuda.is_available() and _C.has_cuda()
    else:
        cuda_available = o3d.core.cuda.is_available()
    try:
        devices, skipped = select_devices(args.device, cuda_available, backend=args.backend)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    ops = list(ALL_OPS) if args.op == "all" else [args.op]
    try:
        prepared = workload.prepare(args, ops)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    report = {
        "environment": environment_metadata(
            devices, open3d_version=o3d.__version__ if o3d is not None else None),
        "config": {"backend": args.backend, **prepared.config(),
                   "threads": args.threads, "repeats": args.repeats,
                   "warmup": args.warmup, "devices": devices},
        "input": prepared.input_metadata(),
        "skipped": skipped,
        "results": [],
    }

    print(f"Backend: {args.backend}; input: {args.input}; "
          f"dtype: {args.dtype}; threads: {args.threads}")
    for line in prepared.describe():
        print(line)
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")

    cases = [(args.backend, device) for device in devices]
    for op in ops:
        rows = []
        for backend, device in cases:
            if op in neighbors.OPS:
                case = neighbors.make_case(prepared.neighbors, op, backend, device)
            elif op in preprocess.OPS:
                case = preprocess.make_case(prepared.cloud, op, backend, device)
            else:
                case = features.make_case(prepared.features, op, backend, device)
            result = measure(case, repeats=args.repeats, warmup=args.warmup, evaluate=False)
            result["op"] = op
            report["results"].append(result)
            rows.append(result)
            del case
        _print_table(op, rows)

    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
