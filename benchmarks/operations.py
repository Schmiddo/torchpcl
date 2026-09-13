"""Command-line benchmark: torchpcl's per-operation kernels vs. Open3D.

One script rather than one per operation: the operations share every argument
and differ only in which cases they build and which accuracy metric they
report, so a single `--op` selector avoids repeating the reporting loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Preserve direct-script usage as well as python -m benchmarks.operations.
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
from benchmarks.inputs import fragment_points, random_points
from benchmarks import features_cases as features
from benchmarks import neighbors_cases as neighbors
from benchmarks import preprocess_cases as preprocess
from benchmarks.runner import measure

DEFAULT_FRAGMENT = Path("data/livingroom1-fragments-ply/cloud_bin_0.ply")
# Redwood fragments are metric and far denser than uniform points in a unit
# cube, so radii tuned for one overrun the neighbor cap on the other.
DEFAULT_RADIUS = {"random": 0.02, "redwood": 0.01}
DEFAULT_OUTLIER_RADIUS = {"random": 0.05, "redwood": 0.02}
# Kept under the 64-neighbor cap at the default cloud sizes, so neither
# library summarizes a truncated neighborhood.
DEFAULT_FEATURE_RADIUS = {"random": 0.03, "redwood": 0.015}

ALL_OPS = (*neighbors.OPS, *preprocess.OPS, *features.OPS)

# The accuracy metric each operation's cases report, or None where a run has
# nothing to score (an index build returns no neighbors; bare voxelize returns
# no centroids, so its companion case carries the check).
GATE_METRICS = {
    "index-build": None,
    "knn": "max_distance2_error",
    "knn-build-query": "max_distance2_error",
    "radius": "count_mismatch_fraction",
    "voxelize": None,
    "voxelize-reduce": "centroid_mismatch_fraction",
    "normals": "min_normal_alignment",
    "statistical-outliers": "mask_mismatch_fraction",
    "radius-outliers": "mask_mismatch_fraction",
    "fpfh": "descriptor_mismatch_fraction",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=[*ALL_OPS, "all"], default="all")
    parser.add_argument("--input", choices=["random", "redwood"], default="random",
                        help="uniform points in a unit cube, or a scan fragment")
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT,
                        help="PLY fragment used by --input redwood")
    parser.add_argument("--num-reference", type=positive_int, default=200_000,
                        help="random-input cloud size; a fragment's own size wins")
    parser.add_argument("--num-queries", type=positive_int, default=50_000)
    parser.add_argument("--k", type=positive_int, default=30,
                        help=f"neighbors per query (at most {neighbors.MAX_NEIGHBORS})")
    parser.add_argument("--radius", type=positive_float,
                        help=f"radius-search radius (default: {DEFAULT_RADIUS})")
    parser.add_argument("--voxel-size", type=positive_float, default=0.02)
    parser.add_argument("--normal-k", type=positive_int, default=30)
    parser.add_argument("--outlier-k", type=positive_int, default=20,
                        help="maps directly to Open3D's nb_neighbors / nb_points")
    parser.add_argument("--std-ratio", type=positive_float, default=2.0)
    parser.add_argument("--outlier-radius", type=positive_float,
                        help=f"default: {DEFAULT_OUTLIER_RADIUS}")
    parser.add_argument("--feature-radius", type=positive_float,
                        help=f"FPFH radius (default: {DEFAULT_FEATURE_RADIUS})")
    parser.add_argument("--feature-neighbors", type=positive_int,
                        default=features.MAX_NEIGHBORS,
                        help=f"FPFH neighbors per point (at most "
                             f"{features.MAX_NEIGHBORS})")
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads(),
                        help="CPU threads (default: physical core count)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if args.k > neighbors.MAX_NEIGHBORS:
        parser.error(f"--k must be at most {neighbors.MAX_NEIGHBORS}")
    if args.radius is None:
        args.radius = DEFAULT_RADIUS[args.input]
    if args.outlier_radius is None:
        args.outlier_radius = DEFAULT_OUTLIER_RADIUS[args.input]
    if args.feature_radius is None:
        args.feature_radius = DEFAULT_FEATURE_RADIUS[args.input]
    if args.feature_neighbors > features.MAX_NEIGHBORS:
        parser.error(f"--feature-neighbors must be at most {features.MAX_NEIGHBORS}")
    return args


def _print_table(op, rows, metric):
    """One table per operation, since the accuracy column differs."""
    print(f"\n{op}")
    print(f"{'backend/device':24} {'median ms':>10} {'min ms':>10} "
          f"{'RAM MiB':>10} {'VRAM MiB':>10} {(metric or ''):>23}")
    for result in rows:
        memory = result["memory"]
        gate = "" if metric is None else f"{result['median'][metric]:23.3e}"
        print(f"{result['backend'] + '/' + result['device']:24} "
              f"{result['median']['time_ms']:10.3f} "
              f"{result['time_ms_min']:10.3f} "
              f"{memory['process_rss_peak_mib']:10.1f} "
              f"{format_optional(memory['device_peak_mib']):>10} "
              f"{gate}", flush=True)


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
        o3d_devices, o3d_skipped = select_devices(
            args.device, o3d.core.cuda.is_available(), backend="open3d")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    skipped = skipped + o3d_skipped

    ops = list(ALL_OPS) if args.op == "all" else [args.op]
    neighbor_ops = [op for op in ops if op in neighbors.OPS]
    preprocess_ops = [op for op in ops if op in preprocess.OPS]
    feature_ops = [op for op in ops if op in features.OPS]

    try:
        fragment = (None if args.input == "random"
                    else fragment_points(args.fragment, args.dtype))
        prepared_neighbors = prepared_cloud = prepared_features = None
        if neighbor_ops:
            prepared_neighbors = neighbors.prepare_neighbors(
                neighbors.NeighborsConfig(
                    num_reference=args.num_reference, num_queries=args.num_queries,
                    k=args.k, radius=args.radius, dtype=args.dtype, seed=args.seed),
                fragment)
        if preprocess_ops:
            points = fragment if fragment is not None else random_points(
                args.num_reference, args.dtype, args.seed)
            prepared_cloud = preprocess.prepare_cloud(
                preprocess.PreprocessConfig(
                    voxel_size=args.voxel_size, normal_k=args.normal_k,
                    outlier_k=args.outlier_k, std_ratio=args.std_ratio,
                    outlier_radius=args.outlier_radius, dtype=args.dtype),
                points, "fragment" if fragment is not None else "random")
        if feature_ops:
            points = fragment if fragment is not None else random_points(
                args.num_reference, args.dtype, args.seed)
            prepared_features = features.prepare_features(
                features.FeaturesConfig(
                    radius=args.feature_radius,
                    max_neighbors=args.feature_neighbors,
                    normal_k=args.normal_k, dtype=args.dtype),
                points, "fragment" if fragment is not None else "random")
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    report = {
        "environment": environment_metadata(devices, open3d_version=o3d.__version__),
        "config": {
            "ops": ops, "input": args.input,
            "fragment": str(args.fragment) if args.input == "redwood" else None,
            "num_reference": args.num_reference, "num_queries": args.num_queries,
            "k": args.k, "radius": args.radius, "voxel_size": args.voxel_size,
            "normal_k": args.normal_k, "outlier_k": args.outlier_k,
            "std_ratio": args.std_ratio, "outlier_radius": args.outlier_radius,
            "feature_radius": args.feature_radius,
            "feature_neighbors": args.feature_neighbors,
            "dtype": args.dtype, "seed": args.seed, "threads": args.threads,
            "repeats": args.repeats, "warmup": args.warmup, "devices": devices,
            "open3d_devices": o3d_devices,
        },
        "input": {},
        "skipped": skipped,
        "results": [],
    }

    print(f"Input: {args.input}; dtype: {args.dtype}; threads: {args.threads}")
    if prepared_neighbors is not None:
        report["input"]["neighbors"] = prepared_neighbors.metadata()
        info = prepared_neighbors.metadata()
        print(f"  search: reference {info['num_reference']}, queries "
              f"{info['num_queries']}, k {args.k}, radius {args.radius}; "
              f"densest scored neighborhood {info['max_radius_count']} "
              f"(torchpcl returns at most {neighbors.MAX_NEIGHBORS})")
    if prepared_cloud is not None:
        report["input"]["preprocess"] = prepared_cloud.metadata()
        print(f"  preprocess: {prepared_cloud.metadata()['num_points']} points, "
              f"voxel {args.voxel_size}, normal k {args.normal_k}, "
              f"outlier k {args.outlier_k}")
    if prepared_features is not None:
        report["input"]["features"] = prepared_features.metadata()
        print(f"  features: {prepared_features.metadata()['num_points']} points, "
              f"radius {args.feature_radius}, "
              f"neighbors {args.feature_neighbors}")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")

    cases = ([("torchpcl", device) for device in devices]
             + [("open3d", device) for device in o3d_devices])
    for op in ops:
        # Open3D's CPU result is the target every case is scored against; a
        # case reporting zero error reproduces the established implementation.
        if op in neighbors.OPS:
            reference = None
        elif op in preprocess.OPS:
            reference = preprocess.reference_output(prepared_cloud, op)
        else:
            reference = features.reference_output(prepared_features)
        rows = []
        for backend, device in cases:
            if op in neighbors.OPS:
                case = neighbors.make_case(prepared_neighbors, op, backend, device)
            elif op in preprocess.OPS:
                case = preprocess.make_case(
                    prepared_cloud, op, backend, device, reference)
            else:
                case = features.make_case(
                    prepared_features, op, backend, device, reference)
            result = measure(case, repeats=args.repeats, warmup=args.warmup)
            result["op"] = op
            report["results"].append(result)
            rows.append(result)
            del case
        _print_table(op, rows, GATE_METRICS[op])

    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
