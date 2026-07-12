"""Benchmarks on the sample scans in data/.

Runs torchpcl registration and preprocessing benchmarks without requiring
comparison libraries. Install the benchmark dependency group to add
small_gicp and open3d comparison rows.

Usage:
    uv run python benchmarks/run_benchmark.py [--task all] [--voxel 0.25] [--repeats 5]
    uv run python benchmarks/run_benchmark.py --task knn --knn-sizes 512 2048 8192
    uv run --group benchmark python benchmarks/run_benchmark.py
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

import torchpcl

from _timing import timed, torch_devices

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_ply_points(path: Path) -> np.ndarray:
    """Read vertex xyz columns from an ASCII or binary little-endian PLY."""
    with path.open("rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no PLY end_header")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        if header[0] != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = next(line.split()[1] for line in header if line.startswith("format "))
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts and parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                properties.append((parts[1], parts[2]))
        if vertex_count <= 0:
            raise ValueError(f"{path} has no vertices")

        dtype_map = {
            "char": "i1",
            "uchar": "u1",
            "int8": "i1",
            "uint8": "u1",
            "short": "i2",
            "ushort": "u2",
            "int16": "i2",
            "uint16": "u2",
            "int": "i4",
            "uint": "u4",
            "int32": "i4",
            "uint32": "u4",
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "float64": "f8",
        }
        names = [name for _, name in properties]
        if names[:3] != ["x", "y", "z"]:
            raise ValueError(f"{path} must store x/y/z as the first vertex properties")

        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, "<" + dtype_map[kind]) for kind, name in properties])
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=vertex_count, dtype=np.float32)
            return data[:, :3]
        raise ValueError(f"unsupported PLY format: {fmt}")


def pose_errors(t_est: np.ndarray, t_gt: np.ndarray) -> tuple[float, float]:
    """Return (rotation error in degrees, translation error) of t_est vs t_gt."""
    err = np.linalg.inv(t_gt) @ t_est
    cos_angle = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle)), float(np.linalg.norm(err[:3, 3]))


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = read_ply_points(DATA_DIR / "source.ply")
    target = read_ply_points(DATA_DIR / "target.ply")
    t_gt = np.loadtxt(DATA_DIR / "T_target_source.txt")
    return source, target, t_gt


def bench_torchpcl_registration(rows, source_np, target_np, t_gt, args, device):
    source = torch.from_numpy(source_np).to(device)
    target = torch.from_numpy(target_np).to(device)

    source_down = torchpcl.voxel_downsample(source, args.voxel)
    target_down = torchpcl.voxel_downsample(target, args.voxel)
    normals = torchpcl.estimate_normals(target_down, k=args.normal_k).normals
    methods = {
        "point-to-point": dict(method="point_to_point"),
        "point-to-plane": dict(method="point_to_plane", target_normals=normals),
    }
    for name, kwargs in methods.items():
        result, seconds = timed(
            lambda kwargs=kwargs: torchpcl.icp(
                source_down,
                target_down,
                args.max_corr_dist,
                max_iterations=args.max_iters,
                **kwargs,
            ),
            args.repeats,
            device=device,
        )
        rot_err, trans_err = pose_errors(result.transforms[0].cpu().numpy(), t_gt)
        rows.append((
            "registration",
            f"torchpcl {name} [{device.type}]",
            seconds,
            f"rot {rot_err:.4f} deg, trans {trans_err:.4f} m, "
            f"iters {result.iterations[0].item()}, "
            f"conv {'yes' if result.converged[0].item() else 'no'}",
        ))


def bench_small_gicp_registration(rows, t_gt, args):
    try:
        import small_gicp
    except ImportError:
        print("small_gicp not importable -- skipping registration comparison")
        return

    source = small_gicp.voxelgrid_sampling(
        small_gicp.read_ply(str(DATA_DIR / "source.ply")), args.voxel
    )
    target = small_gicp.voxelgrid_sampling(
        small_gicp.read_ply(str(DATA_DIR / "target.ply")), args.voxel
    )
    small_gicp.estimate_normals_covariances(source, num_threads=args.threads)
    small_gicp.estimate_normals_covariances(target, num_threads=args.threads)

    for reg_type in ("ICP", "PLANE_ICP", "GICP"):

        def run(reg_type=reg_type):
            tree = small_gicp.KdTree(target, num_threads=args.threads)
            return small_gicp.align(
                target,
                source,
                tree,
                registration_type=reg_type,
                max_correspondence_distance=args.max_corr_dist,
                max_iterations=args.max_iters,
                num_threads=args.threads,
            )

        result, seconds = timed(run, args.repeats)
        rot_err, trans_err = pose_errors(result.T_target_source, t_gt)
        rows.append((
            "registration",
            f"small_gicp {reg_type} [{args.threads}t]",
            seconds,
            f"rot {rot_err:.4f} deg, trans {trans_err:.4f} m, "
            f"iters {result.iterations}, conv {'yes' if result.converged else 'no'}",
        ))


def bench_open3d_registration(rows, source_np, target_np, t_gt, args):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not importable -- skipping registration comparison")
        return

    def to_o3d(points: np.ndarray):
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        return out

    src = to_o3d(source_np).voxel_down_sample(args.voxel)
    tgt = to_o3d(target_np).voxel_down_sample(args.voxel)
    tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_k))
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
        rows.append((
            "registration",
            f"open3d {name}",
            seconds,
            f"rot {rot_err:.4f} deg, trans {trans_err:.4f} m",
        ))


def bench_torchpcl_preprocess(rows, target_np, args, device):
    target = torch.from_numpy(target_np).to(device)

    down, seconds = timed(
        lambda: torchpcl.voxel_downsample(target, args.voxel),
        args.repeats,
        device=device,
    )
    rows.append((
        "preprocess",
        f"torchpcl voxel_downsample [{device.type}]",
        seconds,
        f"{len(target)} -> {len(down)} points",
    ))

    normal_result, seconds = timed(
        lambda: torchpcl.estimate_normals(down, k=args.normal_k),
        args.repeats,
        device=device,
    )
    rows.append((
        "preprocess",
        f"torchpcl estimate_normals [{device.type}]",
        seconds,
        f"{len(normal_result.normals)} normals, k={args.normal_k}",
    ))


def bench_small_gicp_preprocess(rows, args):
    try:
        import small_gicp
    except ImportError:
        print("small_gicp not importable -- skipping preprocessing comparison")
        return

    target = small_gicp.read_ply(str(DATA_DIR / "target.ply"))
    down, seconds = timed(
        lambda: small_gicp.voxelgrid_sampling(target, args.voxel), args.repeats
    )
    rows.append((
        "preprocess",
        f"small_gicp voxelgrid_sampling [{args.threads}t]",
        seconds,
        f"{target.size()} -> {down.size()} points",
    ))

    _, seconds = timed(
        lambda: small_gicp.estimate_normals_covariances(
            down, num_threads=args.threads
        ),
        args.repeats,
    )
    rows.append((
        "preprocess",
        f"small_gicp estimate_normals_covariances [{args.threads}t]",
        seconds,
        f"{down.size()} normals/covariances",
    ))


def bench_open3d_preprocess(rows, target_np, args):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not importable -- skipping preprocessing comparison")
        return

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_np.astype(np.float64))
    down, seconds = timed(lambda: target.voxel_down_sample(args.voxel), args.repeats)
    rows.append((
        "preprocess",
        "open3d voxel_down_sample",
        seconds,
        f"{len(target.points)} -> {len(down.points)} points",
    ))

    _, seconds = timed(
        lambda: down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_k)
        ),
        args.repeats,
    )
    rows.append((
        "preprocess",
        "open3d estimate_normals",
        seconds,
        f"{len(down.points)} normals, k={args.normal_k}",
    ))


def bench_chamfer(rows, source_np, target_np, args, device):
    source = torchpcl.voxel_downsample(torch.from_numpy(source_np).to(device), args.voxel)
    target = torchpcl.voxel_downsample(torch.from_numpy(target_np).to(device), args.voxel)
    # The brute-force baseline materializes an N x M distance matrix that is
    # kept alive for the backward pass, so cap the cloud sizes.
    max_points = 10_000
    source = source[:max_points]
    target = target[:max_points]
    detail = f"{len(source)} vs {len(target)} points, fwd+bwd"

    def run_torchpcl():
        prediction = source.detach().requires_grad_()
        loss = torchpcl.chamfer_loss(prediction, target)
        loss.backward()
        return loss

    _, seconds = timed(run_torchpcl, args.repeats, device=device)
    rows.append(("chamfer", f"torchpcl chamfer_loss [{device.type}]", seconds, detail))

    def run_bruteforce():
        prediction = source.detach().requires_grad_()
        d2 = torch.cdist(prediction, target).square()
        loss = (d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()) / 2
        loss.backward()
        return loss

    _, seconds = timed(run_bruteforce, args.repeats, device=device)
    rows.append(("chamfer", f"torch.cdist brute force [{device.type}]", seconds, detail))


def bench_knn(rows, args, device):
    """Compare index construction and steady-state queries for both backends."""
    from torchpcl import search
    from torchpcl.search import NearestNeighborSearch

    if device.type == "cuda" and search._bruteforce_cuda is None:
        print("brute-force CUDA extension not importable -- skipping CUDA k-NN")
        return
    generator = torch.Generator().manual_seed(1234)

    for num_points in args.knn_sizes:
        num_queries = min(args.knn_queries, num_points)
        points = torch.rand(num_points, 3, generator=generator).to(device)
        queries = torch.rand(num_queries, 3, generator=generator).to(device)
        detail = f"N={num_points}, M={num_queries}, k={args.knn_k}"

        for backend in ("bvh", "bruteforce"):
            def make_search(backend=backend):
                return NearestNeighborSearch(points, math.inf, backend=backend)

            search_index, build_seconds = timed(
                make_search, args.repeats, device=device
            )
            rows.append((
                "knn",
                f"{backend} build [{device.type}]",
                build_seconds,
                detail,
            ))

            _, query_seconds = timed(
                lambda: search_index.knn_query(queries, args.knn_k),
                args.repeats,
                device=device,
            )
            query_rate = num_queries / query_seconds
            rows.append((
                "knn",
                f"{backend} query [{device.type}]",
                query_seconds,
                f"{detail}, {query_rate:,.0f} queries/s",
            ))


def print_rows(rows):
    header = f"{'task':<13} {'method':<52} {'time':>10}  detail"
    print(header)
    print("-" * len(header))
    for task, name, seconds, detail in rows:
        print(f"{task:<13} {name:<52} {seconds * 1e3:>8.2f}ms  {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "registration", "preprocess", "chamfer", "knn"),
                        default="all")
    parser.add_argument("--voxel", type=float, default=0.25,
                        help="voxel downsampling resolution")
    parser.add_argument("--normal-k", type=int, default=30,
                        help="neighbors for normal estimation")
    parser.add_argument("--max-corr-dist", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="threads for small_gicp")
    parser.add_argument("--knn-sizes", type=int, nargs="+", default=[512, 2048, 8192],
                        help="reference point counts for the k-NN benchmark")
    parser.add_argument("--knn-queries", type=int, default=1024,
                        help="maximum query count for each k-NN case")
    parser.add_argument("--knn-k", type=int, default=30,
                        help="neighbor count for the k-NN benchmark")
    args = parser.parse_args()

    if not 1 <= args.knn_k <= 64:
        parser.error("--knn-k must be in [1, 64]")
    if args.knn_queries < 1 or any(size < 1 for size in args.knn_sizes):
        parser.error("--knn-queries and every --knn-sizes value must be positive")

    source = target = t_gt = None
    if args.task != "knn":
        source, target, t_gt = load_inputs()
        print(f"source: {len(source)} pts, target: {len(target)} pts, voxel {args.voxel}, "
              f"normal_k {args.normal_k}, median of {args.repeats} runs\n")
    else:
        print(f"synthetic k-NN, median of {args.repeats} runs\n")

    rows = []
    if args.task in {"all", "preprocess"}:
        for device in torch_devices():
            bench_torchpcl_preprocess(rows, target, args, device)
        bench_small_gicp_preprocess(rows, args)
        bench_open3d_preprocess(rows, target, args)

    if args.task in {"all", "registration"}:
        for device in torch_devices():
            bench_torchpcl_registration(rows, source, target, t_gt, args, device)
        bench_small_gicp_registration(rows, t_gt, args)
        bench_open3d_registration(rows, source, target, t_gt, args)

    if args.task in {"all", "chamfer"}:
        for device in torch_devices():
            bench_chamfer(rows, source, target, args, device)

    if args.task in {"all", "knn"}:
        for device in torch_devices():
            bench_knn(rows, args, device)

    print_rows(rows)


if __name__ == "__main__":
    main()
