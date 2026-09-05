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
from benchmarks.sampling_cases import SamplingConfig, make_case, prepare_cloud


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-points", type=positive_int, default=200_000)
    parser.add_argument("--num-samples", type=positive_int, default=5_000)
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all",
                        help="torchpcl device(s); Open3D FPS is legacy CPU-only")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="point precision (default: float32); Open3D FPS is legacy float64")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads(),
                        help="CPU threads (default: physical core count)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if args.num_samples >= args.num_points:
        parser.error("--num-samples must be less than --num-points")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    o3d = require_open3d()
    import torch
    from torchpcl import _C

    torch.set_num_threads(args.threads)
    cuda_available = torch.cuda.is_available() and _C.has_cuda()
    try:
        devices, skipped = select_devices(args.device, cuda_available)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    config = SamplingConfig(num_points=args.num_points, num_samples=args.num_samples,
                             dtype=args.dtype, seed=args.seed)
    prepared = prepare_cloud(config)

    report = {
        "environment": environment_metadata(devices, open3d_version=o3d.__version__),
        "config": {
            "num_points": config.num_points, "num_samples": config.num_samples,
            "dtype": config.dtype, "seed": config.seed, "threads": args.threads,
            "repeats": args.repeats, "warmup": args.warmup, "devices": devices,
        },
        "skipped": skipped,
        "results": [],
    }
    print(f"Points: {args.num_points}; samples: {args.num_samples}; dtype: {args.dtype}; "
          f"torchpcl: {', '.join(devices)}; threads: {args.threads}")
    print("Open3D farthest_point_down_sample: legacy CPU/float64 reference only.")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")
    print(f"{'backend/device':24} {'median ms':>10} {'RAM MiB':>10} {'VRAM MiB':>10}")

    for backend, device in ([("torchpcl", d) for d in devices] + [("open3d", "cpu")]):
        case = make_case(prepared, backend, device)
        result = measure(case, repeats=args.repeats, warmup=args.warmup)
        report["results"].append(result)
        memory = result["memory"]
        device_text = format_optional(memory["device_peak_mib"])
        print(f"{backend + '/' + device:24} {result['median']['time_ms']:10.3f} "
              f"{memory['process_rss_peak_mib']:10.1f} {device_text:>10}", flush=True)
        del case

    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
