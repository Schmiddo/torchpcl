"""Command-line runner for the reusable registration benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

# Preserve direct-script usage as well as python -m benchmarks.registration.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.case import select_devices
from benchmarks.cli import (
    configure_threads,
    default_threads,
    environment_metadata,
    format_optional,
    positive_float,
    positive_int,
    require_open3d,
    write_json,
)
from benchmarks.registration_cases import RegistrationConfig, prepare_pair, make_case
from benchmarks.runner import measure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data")],
                        help="pair directories containing source.ply, target.ply, T_target_source.txt")
    parser.add_argument("--method", choices=["all", "icp", "fgr"], default="all")
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all",
                        help="default: CPU and CUDA; skip unavailable CUDA in all mode")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="point input precision (default: float32); Open3D FGR is legacy float64")
    parser.add_argument("--voxel-size", type=positive_float, default=0.05)
    parser.add_argument("--icp-distance", type=positive_float, default=None,
                        help="default: 5 * voxel size")
    parser.add_argument("--fgr-distance", type=positive_float, default=None,
                        help="default: 0.5 * voxel size")
    parser.add_argument("--icp-iterations", type=positive_int, default=50)
    parser.add_argument("--fgr-iterations", type=positive_int, default=64)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads(),
                        help="CPU threads (default: physical core count)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")

    return args


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    o3d = require_open3d()
    import torch
    from torchpcl import _C

    torch.set_num_threads(args.threads)
    cuda_available = torch.cuda.is_available() and _C.has_cuda()
    methods = ("icp", "fgr") if args.method == "all" else (args.method,)
    try:
        devices, skipped = select_devices(args.device, cuda_available)
        open3d_devices, open3d_skipped = (select_devices(
            args.device, o3d.core.cuda.is_available(), backend="open3d"
        ) if "icp" in methods else ([], []))
        skipped.extend(open3d_skipped)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = RegistrationConfig(
        voxel_size=args.voxel_size, dtype=args.dtype, seed=args.seed,
        icp_distance=args.icp_distance, fgr_distance=args.fgr_distance,
        icp_iterations=args.icp_iterations, fgr_iterations=args.fgr_iterations,
    )
    report = {
        "environment": environment_metadata(devices, open3d_version=o3d.__version__),
        "config": {**asdict(config), "devices": devices, "open3d_icp_devices": open3d_devices,
                   "requested_device": args.device,
                   "methods": methods, "threads": args.threads,
                   "repeats": args.repeats, "warmup": args.warmup},
        "skipped": skipped,
        "pairs": [],
    }
    print("Registration only; shared voxelization/FPFH excluded from timing.")
    print(f"Input dtype: {args.dtype}; torchpcl: {', '.join(devices)}; "
          f"threads: {args.threads}")
    if "icp" in methods:
        print(f"Open3D tensor ICP: {', '.join(open3d_devices)}/{args.dtype} points; float64 pose solver.")
    if "fgr" in methods:
        print("Open3D FGR: legacy CPU/float64 reference only; no matching tensor/CUDA API.")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")
    print(f"{'pair/method/backend/device':48} {'median ms':>10} {'min ms':>10} {'RAM MiB':>10} "
          f"{'VRAM MiB':>10} {'rot deg':>10} {'trans':>10} {'GT RMSE':>10} {'fitness':>9}")
    for directory in args.data:
        pair = prepare_pair(directory, config, with_features="fgr" in methods)
        pair_report = {**pair.metadata(), "results": []}
        report["pairs"].append(pair_report)
        for method in methods:
            baseline_devices = open3d_devices if method == "icp" else ["cpu"]
            for backend, device in ([("torchpcl", d) for d in devices]
                                    + [("open3d", d) for d in baseline_devices]):
                case = make_case(pair, method, backend, device)
                result = measure(case, repeats=args.repeats, warmup=args.warmup)
                pair_report["results"].append(result)
                median = result["median"]
                memory = result["memory"]
                device_text = format_optional(memory["device_peak_mib"])
                label = f"{directory.name}/{method}/{backend}/{device}"
                print(f"{label:48} {median['time_ms']:10.3f} "
                      f"{result['time_ms_min']:10.3f} "
                      f"{memory['process_rss_peak_mib']:10.1f} {device_text:>10} "
                      f"{median['rotation_error_deg']:10.4f} "
                      f"{median['translation_error']:10.5f} {median['source_displacement_rmse']:10.5f} "
                      f"{median['fitness']:9.4f}", flush=True)
                del case
    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
