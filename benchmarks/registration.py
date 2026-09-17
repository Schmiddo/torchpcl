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
from benchmarks.cli import SMALL_GICP_CPU_ONLY, require_small_gicp
from benchmarks.registration_cases import RegistrationConfig, prepare_pair, make_case
from benchmarks.runner import measure


def _package_version(name):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data")],
                        help="pair directories containing source.ply, target.ply, T_target_source.txt")
    parser.add_argument("--backend", choices=["torchpcl", "open3d", "small-gicp"],
                        default="torchpcl")
    parser.add_argument("--objective", choices=["point-to-point", "point-to-plane"],
                        default="point-to-point", help="ICP residual; FGR ignores it")
    parser.add_argument("--open3d-metrics", action="store_true",
                        help="also report Open3D fitness and inlier RMSE; needs Open3D")
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
    needs_open3d = args.backend == "open3d" or args.open3d_metrics
    o3d = require_open3d() if needs_open3d else None
    gicp = require_small_gicp() if args.backend == "small-gicp" else None
    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "torchpcl":
        from torchpcl import _C

        cuda_available = torch.cuda.is_available() and _C.has_cuda()
    elif args.backend == "small-gicp":
        cuda_available = False
    else:
        cuda_available = o3d.core.cuda.is_available()
    methods = ("icp", "fgr") if args.method == "all" else (args.method,)
    try:
        devices, skipped = select_devices(
            args.device, cuda_available, backend=args.backend,
            reason=SMALL_GICP_CPU_ONLY if args.backend == "small-gicp" else None)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    schedule = []
    for method in methods:
        for device in devices:
            if args.backend == "open3d" and method == "fgr" and device != "cpu":
                skipped.append({"backend": "open3d", "device": device,
                                "reason": "Open3D FGR has no tensor or CUDA API"})
                continue
            if args.backend == "small-gicp" and method == "fgr":
                skipped.append({"backend": "small-gicp", "device": device,
                                "reason": "small-gicp has no global registration"})
                continue
            schedule.append((method, device))
    if not schedule:
        raise SystemExit("no runnable method/device combination; see the skip reasons above")

    config = RegistrationConfig(
        voxel_size=args.voxel_size, dtype=args.dtype, seed=args.seed,
        icp_distance=args.icp_distance, fgr_distance=args.fgr_distance,
        icp_iterations=args.icp_iterations, fgr_iterations=args.fgr_iterations,
    )
    report = {
        "environment": environment_metadata(
            devices, open3d_version=o3d.__version__ if o3d is not None else None,
            small_gicp_version=_package_version("small-gicp") if gicp else None),
        "config": {**asdict(config), "backend": args.backend, "devices": devices,
                   "requested_device": args.device, "methods": methods,
                   "objective": args.objective,
                   "open3d_metrics": args.open3d_metrics, "threads": args.threads,
                   "repeats": args.repeats, "warmup": args.warmup},
        "skipped": skipped,
        "pairs": [],
    }
    print(f"Backend: {args.backend}; registration only. Preprocessing is torchpcl's "
          "and excluded from timing.")
    if args.objective == "point-to-plane":
        print("Point-to-plane uses the shared normals, except for small-gicp, "
              "which has no normal setter and estimates its own.")
    print(f"Input dtype: {args.dtype}; devices: {', '.join(devices)}; "
          f"threads: {args.threads}")
    if args.backend == "open3d" and "icp" in methods:
        print("Open3D tensor ICP stores points at the requested dtype/device; "
              "its pose solver is CPU float64.")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")
    quality = f"{'fitness':>9}" if args.open3d_metrics else ""
    print(f"{'pair/method/device':40} {'median ms':>10} {'min ms':>10} {'RAM MiB':>10} "
          f"{'VRAM MiB':>10} {'rot deg':>10} {'trans':>10} {'GT RMSE':>10}{quality}")
    for directory in args.data:
        pair = prepare_pair(directory, config,
                            with_normals=args.objective == "point-to-plane",
                            with_features="fgr" in methods)
        pair_report = {**pair.metadata(), "results": []}
        report["pairs"].append(pair_report)
        for method, device in schedule:
            case = make_case(pair, method, args.backend, device,
                             open3d_metrics=args.open3d_metrics,
                             objective=args.objective, threads=args.threads)
            result = measure(case, repeats=args.repeats, warmup=args.warmup)
            pair_report["results"].append(result)
            median = result["median"]
            memory = result["memory"]
            device_text = format_optional(memory["device_peak_mib"])
            label = f"{directory.name}/{method}/{device}"
            print(f"{label:40} {median['time_ms']:10.3f} "
                  f"{result['time_ms_min']:10.3f} "
                  f"{memory['process_rss_peak_mib']:10.1f} {device_text:>10} "
                  f"{median['rotation_error_deg']:10.4f} "
                  f"{median['translation_error']:10.5f} "
                  f"{median['source_displacement_rmse']:10.5f}"
                  + (f"{median['fitness']:9.4f}" if args.open3d_metrics else ""),
                  flush=True)
            del case
    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
