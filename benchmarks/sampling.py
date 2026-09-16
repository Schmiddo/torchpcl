"""Command-line benchmark: torchpcl's farthest_point_sample vs. Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Preserve direct-script usage as well as python -m benchmarks.sampling.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from benchmarks.runner import measure
from benchmarks.sampling_cases import BACKENDS, SamplingConfig, make_case, prepare_cloud


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="torchpcl")
    parser.add_argument("--num-points", type=positive_int, default=200_000)
    parser.add_argument("--num-samples", type=positive_int, default=5_000)
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all",
                        help="device(s) for torchpcl and Open3D's tensor API; "
                             "Open3D's legacy API is CPU-only")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="point precision (default: float32); "
                             "Open3D's legacy API always computes in float64")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads(),
                        help="CPU threads (default: physical core count)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.backend == "open3d-legacy" and args.device == "cuda":
        parser.error("Open3D legacy sampling supports CPU only")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if args.num_samples >= args.num_points:
        parser.error("--num-samples must be less than --num-points")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    o3d = require_open3d() if args.backend != "torchpcl" else None
    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "torchpcl":
        from torchpcl import _C

        cuda_available = torch.cuda.is_available() and _C.has_cuda()
    else:
        cuda_available = args.backend == "open3d-tensor" and o3d.core.cuda.is_available()
    try:
        devices, skipped = select_devices(args.device, cuda_available, backend=args.backend)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.backend == "open3d-legacy":
        for skip in skipped:
            skip["reason"] = "legacy sampling supports CPU only"

    config = SamplingConfig(num_points=args.num_points, num_samples=args.num_samples,
                             dtype=args.dtype, seed=args.seed)
    prepared = prepare_cloud(config)

    report = {
        "environment": environment_metadata(
            devices, open3d_version=o3d.__version__ if o3d is not None else None),
        "config": {
            "backend": args.backend,
            "num_points": config.num_points, "num_samples": config.num_samples,
            "dtype": config.dtype, "seed": config.seed, "threads": args.threads,
            "repeats": args.repeats, "warmup": args.warmup, "devices": devices,
        },
        "skipped": skipped,
        "results": [],
    }
    print(f"Points: {args.num_points}; samples: {args.num_samples}; dtype: {args.dtype}; "
          f"{args.backend}: {', '.join(devices)}; threads: {args.threads}")
    if args.backend == "open3d-legacy":
        print("Open3D legacy sampling computes in float64.")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")
    print(f"{'backend/device':24} {'median ms':>10} {'min ms':>10} "
          f"{'RAM MiB':>10} {'VRAM MiB':>10}")

    cases = [(args.backend, device) for device in devices]
    for backend, device in cases:
        case = make_case(prepared, backend, device)
        result = measure(case, repeats=args.repeats, warmup=args.warmup, evaluate=False)
        report["results"].append(result)
        memory = result["memory"]
        device_text = format_optional(memory["device_peak_mib"])
        print(f"{backend + '/' + device:24} {result['median']['time_ms']:10.3f} "
              f"{result['time_ms_min']:10.3f} "
              f"{memory['process_rss_peak_mib']:10.1f} {device_text:>10}", flush=True)
        del case

    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
