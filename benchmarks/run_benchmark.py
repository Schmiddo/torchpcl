"""Registration benchmark on the sample scans in data/.

Registers data/source.ply to data/target.ply and reports the rotation /
translation error against data/T_target_source.txt plus the wall time,
for torchpcl, small_gicp, and (if importable) open3d.

Protocol: both clouds are voxel-downsampled once and normals/covariances
are estimated once, outside the timed region, shared by all methods.
Each timed run includes the library's own search-structure build
(BVH / KdTree) and the full registration from an identity init.

Usage: uv run python benchmarks/run_benchmark.py [--voxel 0.25] [--repeats 5]
"""

import argparse
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import small_gicp
import torch

import torchpcl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def pose_errors(t_est: np.ndarray, t_gt: np.ndarray) -> tuple[float, float]:
    """Return (rotation error in degrees, translation error) of t_est vs t_gt."""
    err = np.linalg.inv(t_gt) @ t_est
    cos_angle = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle)), float(np.linalg.norm(err[:3, 3]))


def timed(fn, repeats: int, sync=None):
    """Run fn repeats+1 times (first is warmup); return (last result, median seconds)."""
    fn()  # warmup: extension JIT compilation, allocator pools, caches
    times = []
    for _ in range(repeats):
        if sync:
            sync()
        start = time.perf_counter()
        result = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def preprocess(args):
    source = small_gicp.read_ply(str(DATA_DIR / "source.ply"))
    target = small_gicp.read_ply(str(DATA_DIR / "target.ply"))
    t_gt = np.loadtxt(DATA_DIR / "T_target_source.txt")

    source = small_gicp.voxelgrid_sampling(source, args.voxel)
    target = small_gicp.voxelgrid_sampling(target, args.voxel)
    small_gicp.estimate_normals_covariances(source, num_threads=args.threads)
    small_gicp.estimate_normals_covariances(target, num_threads=args.threads)
    return source, target, t_gt


def bench_torchpcl(rows, source, target, t_gt, args, device):
    # float32 inputs: torchpcl works in the input precision (only the
    # transformation and the small solves stay float64).
    src = torch.from_numpy(source.points()[:, :3]).to(device, torch.float32)
    tgt = torch.from_numpy(target.points()[:, :3]).to(device, torch.float32)
    normals = torch.from_numpy(target.normals()[:, :3]).to(device, torch.float32)
    criteria = torchpcl.ICPConvergenceCriteria(max_iteration=args.max_iters)
    sync = torch.cuda.synchronize if device.type == "cuda" else None

    methods = {
        "point-to-point": dict(estimation=torchpcl.PointToPoint()),
        "point-to-plane": dict(
            estimation=torchpcl.PointToPlane(), target_normals=normals
        ),
    }
    for name, kwargs in methods.items():
        result, seconds = timed(
            lambda kwargs=kwargs: torchpcl.icp(
                src, tgt, args.max_corr_dist, criteria=criteria, **kwargs
            ),
            args.repeats,
            sync=sync,
        )
        rot_err, trans_err = pose_errors(result.transformation.cpu().numpy(), t_gt)
        rows.append((
            f"torchpcl {name} [{device.type}]",
            seconds, rot_err, trans_err, result.num_iterations, result.converged,
        ))


def bench_small_gicp(rows, source, target, t_gt, args):
    for reg_type in ("ICP", "PLANE_ICP", "GICP"):
        def run(reg_type=reg_type):
            tree = small_gicp.KdTree(target, num_threads=args.threads)
            return small_gicp.align(
                target, source, tree,
                registration_type=reg_type,
                max_correspondence_distance=args.max_corr_dist,
                max_iterations=args.max_iters,
                num_threads=args.threads,
            )

        result, seconds = timed(run, args.repeats)
        rot_err, trans_err = pose_errors(result.T_target_source, t_gt)
        rows.append((
            f"small_gicp {reg_type} [{args.threads}t]",
            seconds, rot_err, trans_err, result.iterations, result.converged,
        ))


def bench_open3d(rows, source, target, t_gt, args):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not importable in this environment -- skipping "
              "(no Python 3.14 wheels; see README)")
        return

    def to_o3d(pcd, with_normals):
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pcd.points()[:, :3])
        if with_normals:
            out.normals = o3d.utility.Vector3dVector(pcd.normals()[:, :3])
        return out

    src = to_o3d(source, with_normals=False)
    tgt = to_o3d(target, with_normals=True)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=args.max_iters
    )
    methods = {
        "point-to-point": o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        "point-to-plane": o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    }
    for name, estimation in methods.items():
        result, seconds = timed(
            lambda estimation=estimation: o3d.pipelines.registration.registration_icp(
                src, tgt, args.max_corr_dist, np.eye(4), estimation, criteria
            ),
            args.repeats,
        )
        rot_err, trans_err = pose_errors(np.asarray(result.transformation), t_gt)
        rows.append((f"open3d {name}", seconds, rot_err, trans_err, None, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel", type=float, default=0.25,
                        help="voxel downsampling resolution")
    parser.add_argument("--max-corr-dist", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="threads for small_gicp")
    args = parser.parse_args()

    source, target, t_gt = preprocess(args)
    print(f"source: {source.size()} pts, target: {target.size()} pts "
          f"(voxel {args.voxel}), max_corr_dist {args.max_corr_dist}, "
          f"max_iters {args.max_iters}, median of {args.repeats} runs\n")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []
    for device in devices:
        bench_torchpcl(rows, source, target, t_gt, args, device)
    bench_small_gicp(rows, source, target, t_gt, args)
    bench_open3d(rows, source, target, t_gt, args)

    header = f"{'method':<34} {'time':>9} {'rot err':>10} {'trans err':>10} {'iters':>6}  conv"
    print(header)
    print("-" * len(header))
    for name, seconds, rot_err, trans_err, iters, converged in rows:
        iters_s = "-" if iters is None else str(iters)
        conv_s = "-" if converged is None else ("yes" if converged else "no")
        print(f"{name:<34} {seconds * 1e3:>7.2f}ms {rot_err:>9.4f}° "
              f"{trans_err:>9.4f}m {iters_s:>6}  {conv_s}")


if __name__ == "__main__":
    main()
