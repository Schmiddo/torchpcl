"""Release-oriented benchmarks for torchpcl 0.2.

Open3D and small_gicp rows are enabled explicitly with ``--comparisons``.

Examples:
    uv run --group benchmark python benchmarks/run_benchmark.py --task all --device cpu
    uv run --group benchmark python benchmarks/run_benchmark.py --task preprocess --device cuda
    uv run --group benchmark python benchmarks/run_benchmark.py --task search --search-sizes 512 2048
    uv run --group benchmark python benchmarks/run_benchmark.py --comparisons
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

import torchpcl
from torchpcl import _C

from _timing import TimingResult, synchronize, timed

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkRow:
    task: str
    case: str
    implementation: str
    scope: str
    device: str
    dtype: str
    parameters: dict[str, Any]
    samples_seconds: tuple[float, ...]
    median_seconds: float
    iqr_seconds: float
    detail: dict[str, Any]


def add_row(
    rows: list[BenchmarkRow],
    *,
    task: str,
    case: str,
    implementation: str,
    scope: str,
    device: torch.device | str,
    dtype: torch.dtype | str,
    parameters: dict[str, Any],
    timing: TimingResult,
    detail: dict[str, Any],
) -> None:
    rows.append(
        BenchmarkRow(
            task=task,
            case=case,
            implementation=implementation,
            scope=scope,
            device=str(device),
            dtype=str(dtype).removeprefix("torch."),
            parameters=parameters,
            samples_seconds=timing.samples_seconds,
            median_seconds=timing.median_seconds,
            iqr_seconds=timing.iqr_seconds,
            detail=detail,
        )
    )


def downsample_points(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    return torchpcl.voxelize(points, voxel_size).cloud.points


def read_ply_points(path: Path) -> np.ndarray:
    """Read vertex xyz columns from a PLY file."""
    vertices = PlyData.read(path)["vertex"].data
    return np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float32, copy=False)


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = read_ply_points(DATA_DIR / "source.ply")
    target = read_ply_points(DATA_DIR / "target.ply")
    transform = np.loadtxt(DATA_DIR / "T_target_source.txt")
    return source, target, transform


def pose_errors(
    estimated: np.ndarray, expected: np.ndarray
) -> tuple[float, float]:
    """Return rotation error in degrees and translation error."""
    error = np.linalg.inv(expected) @ estimated
    cosine = np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return (
        math.degrees(math.acos(cosine)),
        float(np.linalg.norm(error[:3, 3])),
    )


def batched_pose_errors(
    estimated: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    errors = [
        pose_errors(actual, wanted)
        for actual, wanted in zip(
            estimated.detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
        )
    ]
    return max(value[0] for value in errors), max(value[1] for value in errors)


def ragged_lengths(
    batch_count: int,
    typical_length: int,
    *,
    minimum: int = 1,
) -> list[int]:
    factors = (0.50, 1.50, 0.75, 1.25, 1.00, 0.625, 1.375)
    return [
        max(minimum, round(typical_length * factors[index % len(factors)]))
        for index in range(batch_count)
    ]


def length_summary(lengths: list[int]) -> dict[str, int | float]:
    return {
        "batch_count": len(lengths),
        "total": sum(lengths),
        "minimum": min(lengths),
        "median": statistics.median(lengths),
        "maximum": max(lengths),
    }


def offsets_from_lengths(
    lengths: list[int], device: torch.device
) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def random_packed_cloud(
    lengths: list[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torchpcl.PointCloud:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    points = torch.rand(
        sum(lengths), 3, generator=generator, dtype=dtype
    ).to(device)
    return torchpcl.PointCloud(points, offsets_from_lengths(lengths, device))


def cuda_memory_detail(
    function,
    device: torch.device,
) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    synchronize(device)
    starting_allocated = torch.cuda.memory_allocated(device)
    starting_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = function()
    synchronize(device)
    detail = {
        "starting_allocated_bytes": starting_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "incremental_peak_allocated_bytes": max(
            0, torch.cuda.max_memory_allocated(device) - starting_allocated
        ),
        "starting_reserved_bytes": starting_reserved,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "incremental_peak_reserved_bytes": max(
            0, torch.cuda.max_memory_reserved(device) - starting_reserved
        ),
    }
    del result
    return detail


def benchmark_search(
    rows: list[BenchmarkRow],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for point_count in args.search_sizes:
        query_count = min(args.search_queries, point_count)
        generator = torch.Generator(device="cpu").manual_seed(1234 + point_count)
        points = torch.rand(
            point_count, 3, generator=generator, dtype=dtype
        ).to(device)
        queries = torch.rand(
            query_count, 3, generator=generator, dtype=dtype
        ).to(device)

        bvh, timing = timed(
            lambda: torchpcl.NeighborIndex(points, algorithm="bvh"),
            args.repeats,
            device=device,
        )
        add_row(
            rows,
            task="search",
            case="single_cloud_bvh_build",
            implementation="torchpcl",
            scope="index-build",
            device=device,
            dtype=dtype,
            parameters={"reference_points": point_count},
            timing=timing,
            detail={},
        )

        brute_force = torchpcl.NeighborIndex(points, algorithm="bruteforce")
        for k in args.search_k:
            for name, index in (("bvh", bvh), ("bruteforce", brute_force)):
                result, timing = timed(
                    lambda index=index, k=k: index.knn(queries, k),
                    args.repeats,
                    device=device,
                )
                add_row(
                    rows,
                    task="search",
                    case=f"single_cloud_{name}_query",
                    implementation="torchpcl",
                    scope="query",
                    device=device,
                    dtype=dtype,
                    parameters={
                        "reference_points": point_count,
                        "query_points": query_count,
                        "k": k,
                    },
                    timing=timing,
                    detail={
                        "valid_results": int(result.valid.sum().item()),
                        "queries_per_second": (
                            query_count / timing.median_seconds
                        ),
                    },
                )

    reference_lengths = ragged_lengths(
        args.packed_batches,
        args.packed_points,
        minimum=max(args.search_k),
    )
    query_lengths = ragged_lengths(
        args.packed_batches,
        args.packed_queries,
    )
    reference = random_packed_cloud(
        reference_lengths, seed=2401, device=device, dtype=dtype
    )
    queries = random_packed_cloud(
        query_lengths, seed=2402, device=device, dtype=dtype
    )
    index = torchpcl.NeighborIndex(reference, algorithm="bruteforce")
    for k in args.search_k:
        result, timing = timed(
            lambda k=k: index.knn(queries, k),
            args.repeats,
            device=device,
        )
        add_row(
            rows,
            task="search",
            case="ragged_packed_bruteforce_query",
            implementation="torchpcl",
            scope="query",
            device=device,
            dtype=dtype,
            parameters={
                "k": k,
                "reference_lengths": length_summary(reference_lengths),
                "query_lengths": length_summary(query_lengths),
            },
            timing=timing,
            detail={
                "valid_results": int(result.valid.sum().item()),
                "queries_per_second": (
                    sum(query_lengths) / timing.median_seconds
                ),
            },
        )


def benchmark_preprocess(
    rows: list[BenchmarkRow],
    target_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    target = torch.from_numpy(target_np).to(device=device, dtype=dtype)
    resolutions = [("ordinary", args.voxel), ("fine", args.fine_voxel)]
    for label, voxel_size in resolutions:
        partition, voxel_timing = timed(
            lambda voxel_size=voxel_size: torchpcl.voxelize(
                target, voxel_size
            ),
            args.repeats,
            device=device,
        )
        downsampled = partition.cloud.points
        add_row(
            rows,
            task="preprocess",
            case=f"voxelize_{label}",
            implementation="torchpcl",
            scope="operation",
            device=device,
            dtype=dtype,
            parameters={"voxel_size": voxel_size},
            timing=voxel_timing,
            detail={
                "input_points": len(target),
                "output_points": len(downsampled),
            },
        )

        result, normal_timing = timed(
            lambda downsampled=downsampled: torchpcl.estimate_normals(
                downsampled, k=args.normal_k
            ),
            args.repeats,
            device=device,
        )
        detail: dict[str, Any] = {
            "input_points": len(downsampled),
            "valid_normals": int(result.valid.sum().item()),
        }
        if label == "fine":
            detail.update(
                cuda_memory_detail(
                    lambda: torchpcl.estimate_normals(
                        downsampled, k=args.normal_k
                    ),
                    device,
                )
            )
        add_row(
            rows,
            task="preprocess",
            case=f"estimate_normals_{label}",
            implementation="torchpcl",
            scope="operation",
            device=device,
            dtype=dtype,
            parameters={
                "voxel_size": voxel_size,
                "k": args.normal_k,
            },
            timing=normal_timing,
            detail=detail,
        )

    generator = torch.Generator(device="cpu").manual_seed(3100)
    factors = torch.randn(
        args.eigh_matrices, 3, 3, generator=generator, dtype=dtype
    )
    covariances = (factors @ factors.mT).to(device).contiguous()
    eigen_result, eigen_timing = timed(
        lambda: _C.symmetric_eigh_3x3(covariances),
        args.repeats,
        device=device,
    )
    eigenvalues, eigenvectors = eigen_result
    residual = (
        (covariances @ eigenvectors.unsqueeze(-1)).squeeze(-1)
        - eigenvalues[:, :1] * eigenvectors
    )
    eigen_detail: dict[str, Any] = {
        "finite_matrices": int(
            (
                torch.isfinite(eigenvalues).all(dim=1)
                & torch.isfinite(eigenvectors).all(dim=1)
            )
            .sum()
            .item()
        ),
        "max_smallest_eigenpair_residual": float(
            residual.norm(dim=1).max().item()
        ),
    }
    eigen_detail.update(
        cuda_memory_detail(
            lambda: _C.symmetric_eigh_3x3(covariances),
            device,
        )
    )
    add_row(
        rows,
        task="preprocess",
        case="native_symmetric_eigh_3x3",
        implementation="torchpcl native",
        scope="kernel",
        device=device,
        dtype=dtype,
        parameters={"matrix_count": args.eigh_matrices},
        timing=eigen_timing,
        detail=eigen_detail,
    )

    lengths = ragged_lengths(
        args.packed_batches,
        args.packed_points,
        minimum=args.normal_k,
    )
    cloud = random_packed_cloud(
        lengths, seed=3200, device=device, dtype=dtype
    )
    result, timing = timed(
        lambda: torchpcl.estimate_normals(cloud, k=args.normal_k),
        args.repeats,
        device=device,
    )
    add_row(
        rows,
        task="preprocess",
        case="ragged_packed_estimate_normals",
        implementation="torchpcl",
        scope="operation",
        device=device,
        dtype=dtype,
        parameters={
            "lengths": length_summary(lengths),
            "k": args.normal_k,
        },
        timing=timing,
        detail={
            "input_points": len(cloud.points),
            "valid_normals": int(result.valid.sum().item()),
        },
    )


def benchmark_metrics(
    rows: list[BenchmarkRow],
    source_np: np.ndarray,
    target_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    source = downsample_points(
        torch.from_numpy(source_np).to(device=device, dtype=dtype),
        args.voxel,
    )[: args.metric_max_points]
    target = downsample_points(
        torch.from_numpy(target_np).to(device=device, dtype=dtype),
        args.voxel,
    )[: args.metric_max_points]
    common_parameters = {
        "source_points": len(source),
        "target_points": len(target),
        "squared": True,
    }

    cases = (
        ("directed_forward", False),
        ("bidirectional_forward", True),
    )
    for case, bidirectional in cases:
        loss, timing = timed(
            lambda bidirectional=bidirectional: torchpcl.chamfer_distance(
                source,
                target,
                bidirectional=bidirectional,
            ),
            args.repeats,
            device=device,
        )
        add_row(
            rows,
            task="metrics",
            case=case,
            implementation="torchpcl chamfer_distance",
            scope="forward",
            device=device,
            dtype=dtype,
            parameters={
                **common_parameters,
                "bidirectional": bidirectional,
            },
            timing=timing,
            detail={"loss": float(loss.item())},
        )

    def torchpcl_forward_backward() -> torch.Tensor:
        prediction = source.detach().requires_grad_()
        loss = torchpcl.chamfer_distance(prediction, target)
        loss.backward()
        return loss.detach()

    loss, timing = timed(
        torchpcl_forward_backward,
        args.repeats,
        device=device,
    )
    add_row(
        rows,
        task="metrics",
        case="bidirectional_forward_backward",
        implementation="torchpcl chamfer_distance",
        scope="forward+backward",
        device=device,
        dtype=dtype,
        parameters={**common_parameters, "bidirectional": True},
        timing=timing,
        detail={"loss": float(loss.item())},
    )

    source_lengths = ragged_lengths(
        args.packed_batches, args.packed_queries
    )
    target_lengths = ragged_lengths(
        args.packed_batches,
        args.packed_points,
    )
    packed_source = random_packed_cloud(
        source_lengths, seed=4100, device=device, dtype=dtype
    )
    packed_target = random_packed_cloud(
        target_lengths, seed=4101, device=device, dtype=dtype
    )
    packed_loss, timing = timed(
        lambda: torchpcl.chamfer_distance(
            packed_source,
            packed_target,
            bidirectional=False,
        ),
        args.repeats,
        device=device,
    )
    add_row(
        rows,
        task="metrics",
        case="ragged_packed_directed_forward",
        implementation="torchpcl chamfer_distance",
        scope="forward",
        device=device,
        dtype=dtype,
        parameters={
            "source_lengths": length_summary(source_lengths),
            "target_lengths": length_summary(target_lengths),
            "squared": True,
            "bidirectional": False,
        },
        timing=timing,
        detail={"loss": float(packed_loss.item())},
    )

    def cdist_forward_backward() -> torch.Tensor:
        prediction = source.detach().requires_grad_()
        distances2 = torch.cdist(prediction, target).square()
        loss = (
            distances2.min(dim=1).values.mean()
            + distances2.min(dim=0).values.mean()
        ) / 2
        loss.backward()
        return loss.detach()

    cdist_loss, timing = timed(
        cdist_forward_backward,
        args.repeats,
        device=device,
    )
    add_row(
        rows,
        task="metrics",
        case="bidirectional_forward_backward",
        implementation="torch.cdist",
        scope="forward+backward",
        device=device,
        dtype=dtype,
        parameters={**common_parameters, "bidirectional": True},
        timing=timing,
        detail={"loss": float(cdist_loss.item())},
    )


def configured_levels(args: argparse.Namespace) -> list[torchpcl.ICPLevel]:
    return [
        torchpcl.ICPLevel(
            voxel_size=voxel_size,
            max_correspondence_distance=distance,
            max_iterations=iterations,
        )
        for voxel_size, distance, iterations in zip(
            args.level_voxels,
            args.level_distances,
            args.level_iterations,
        )
    ]


def level_parameters(
    levels: list[torchpcl.ICPLevel],
) -> list[dict[str, float | int | None]]:
    return [
        {
            "voxel_size": level.voxel_size,
            "max_correspondence_distance": (
                level.max_correspondence_distance
            ),
            "max_iterations": level.max_iterations,
        }
        for level in levels
    ]


def icp_detail(
    result: torchpcl.ICPResult,
    expected: np.ndarray,
) -> dict[str, Any]:
    rotation_error, translation_error = pose_errors(
        result.transforms[0].detach().cpu().numpy(),
        expected,
    )
    return {
        "rotation_error_degrees": rotation_error,
        "translation_error": translation_error,
        "iterations": int(result.iterations[0].item()),
        "level_iterations": [
            int(level.iterations[0].item())
            for level in result.level_results
        ],
        "converged": bool(result.converged[0].item()),
        "fitness": float(result.fitness[0].item()),
        "inlier_rmse": float(result.inlier_rmse[0].item()),
    }


def small_batched_transforms(
    batch_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    angles = torch.linspace(-0.025, 0.025, batch_count, dtype=dtype, device=device)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    transforms = torch.eye(4, dtype=dtype, device=device).repeat(
        batch_count, 1, 1
    )
    transforms[:, 0, 0] = cosine
    transforms[:, 0, 1] = -sine
    transforms[:, 1, 0] = sine
    transforms[:, 1, 1] = cosine
    transforms[:, 0, 3] = torch.linspace(
        -0.015, 0.015, batch_count, dtype=dtype, device=device
    )
    transforms[:, 1, 3] = torch.linspace(
        0.01, -0.01, batch_count, dtype=dtype, device=device
    )
    return transforms


def benchmark_registration(
    rows: list[BenchmarkRow],
    source_np: np.ndarray,
    target_np: np.ndarray,
    expected: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    source = torch.from_numpy(source_np).to(device=device, dtype=dtype)
    target = torch.from_numpy(target_np).to(device=device, dtype=dtype)
    source_down = downsample_points(source, args.voxel)
    target_down = downsample_points(target, args.voxel)
    target_normals = torchpcl.estimate_normals(
        target_down, k=args.normal_k
    ).normals
    target_with_normals = torchpcl.PointCloud.from_points(
        target_down, normals=target_normals
    )
    single_level = [
        torchpcl.ICPLevel(
            max_correspondence_distance=args.max_corr_dist,
            max_iterations=args.max_iters,
            voxel_size=None,
        )
    ]
    single_cases = (
        (
            "point_to_point",
            target_down,
            torchpcl.ICPOptions(objective=torchpcl.PointToPoint()),
        ),
        (
            "point_to_plane",
            target_with_normals,
            torchpcl.ICPOptions(objective=torchpcl.PointToPlane()),
        ),
    )
    for objective_name, case_target, options in single_cases:
        result, timing = timed(
            lambda case_target=case_target, options=options: torchpcl.icp(
                source_down,
                case_target,
                single_level,
                options=options,
            ),
            args.repeats,
            device=device,
        )
        add_row(
            rows,
            task="registration",
            case=f"single_level_{objective_name}",
            implementation="torchpcl icp",
            scope="alignment-only",
            device=device,
            dtype=dtype,
            parameters={
                "source_points": len(source_down),
                "target_points": len(target_down),
                "objective": objective_name,
                "levels": level_parameters(single_level),
            },
            timing=timing,
            detail=icp_detail(result, expected),
        )

    levels = configured_levels(args)
    for objective_name, objective in (
        ("point_to_point", torchpcl.PointToPoint()),
        ("point_to_plane", torchpcl.PointToPlane()),
    ):

        def run_end_to_end(
            objective=objective,
        ) -> torchpcl.ICPResult:
            case_target: torch.Tensor | torchpcl.PointCloud = target
            if isinstance(objective, torchpcl.PointToPlane):
                normals = torchpcl.estimate_normals(
                    target, k=args.normal_k
                ).normals
                case_target = torchpcl.PointCloud.from_points(
                    target, normals=normals
                )
            return torchpcl.icp(
                source,
                case_target,
                levels,
                options=torchpcl.ICPOptions(objective=objective),
            )

        result, timing = timed(
            run_end_to_end,
            args.repeats,
            device=device,
        )
        add_row(
            rows,
            task="registration",
            case=f"multi_level_{objective_name}",
            implementation="torchpcl icp",
            scope="end-to-end",
            device=device,
            dtype=dtype,
            parameters={
                "source_points": len(source),
                "target_points": len(target),
                "objective": objective_name,
                "levels": level_parameters(levels),
                "normal_k": (
                    args.normal_k if objective_name == "point_to_plane" else None
                ),
            },
            timing=timing,
            detail=icp_detail(result, expected),
        )

    lengths = ragged_lengths(
        args.packed_batches,
        args.packed_points,
        minimum=32,
    )
    packed_target = random_packed_cloud(
        lengths, seed=5100, device=device, dtype=dtype
    )
    expected_transforms = small_batched_transforms(
        len(lengths), device=device, dtype=dtype
    )
    packed_source = torchpcl.transform(
        packed_target, torch.linalg.inv(expected_transforms)
    )
    assert isinstance(packed_source, torchpcl.PointCloud)
    packed_level = [
        torchpcl.ICPLevel(
            max_correspondence_distance=args.max_corr_dist,
            max_iterations=args.max_iters,
            voxel_size=None,
        )
    ]
    result, timing = timed(
        lambda: torchpcl.icp(
            packed_source,
            packed_target,
            packed_level,
        ),
        args.repeats,
        device=device,
    )
    rotation_error, translation_error = batched_pose_errors(
        result.transforms, expected_transforms
    )
    iteration_values = result.iterations.detach().cpu().tolist()
    add_row(
        rows,
        task="registration",
        case="ragged_packed_single_level_point_to_point",
        implementation="torchpcl icp",
        scope="alignment-only",
        device=device,
        dtype=dtype,
        parameters={
            "lengths": length_summary(lengths),
            "objective": "point_to_point",
            "levels": level_parameters(packed_level),
        },
        timing=timing,
        detail={
            "maximum_rotation_error_degrees": rotation_error,
            "maximum_translation_error": translation_error,
            "iterations_minimum": min(iteration_values),
            "iterations_median": statistics.median(iteration_values),
            "iterations_maximum": max(iteration_values),
            "converged_batches": int(result.converged.sum().item()),
            "fitness_minimum": float(result.fitness.min().item()),
            "inlier_rmse_maximum": float(result.inlier_rmse.max().item()),
        },
    )


def benchmark_small_gicp_preprocess(
    rows: list[BenchmarkRow], args: argparse.Namespace
) -> None:
    try:
        import small_gicp
    except ImportError:
        print("small_gicp is unavailable; skipping its comparison rows")
        return
    target = small_gicp.read_ply(str(DATA_DIR / "target.ply"))
    downsampled, timing = timed(
        lambda: small_gicp.voxelgrid_sampling(target, args.voxel),
        args.repeats,
    )
    add_row(
        rows,
        task="preprocess",
        case="voxelize_ordinary",
        implementation="small_gicp voxelgrid_sampling",
        scope="operation",
        device="cpu",
        dtype="implementation-default",
        parameters={"voxel_size": args.voxel, "threads": args.threads},
        timing=timing,
        detail={
            "input_points": int(target.size()),
            "output_points": int(downsampled.size()),
        },
    )
    _, timing = timed(
        lambda: small_gicp.estimate_normals_covariances(
            downsampled, num_threads=args.threads
        ),
        args.repeats,
    )
    add_row(
        rows,
        task="preprocess",
        case="estimate_normals_ordinary",
        implementation="small_gicp estimate_normals_covariances",
        scope="operation",
        device="cpu",
        dtype="implementation-default",
        parameters={"k": "implementation-default", "threads": args.threads},
        timing=timing,
        detail={"input_points": int(downsampled.size())},
    )


def benchmark_open3d_preprocess(
    rows: list[BenchmarkRow],
    target_np: np.ndarray,
    args: argparse.Namespace,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is unavailable; skipping its comparison rows")
        return
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_np.astype(np.float64))
    downsampled, timing = timed(
        lambda: target.voxel_down_sample(args.voxel),
        args.repeats,
    )
    add_row(
        rows,
        task="preprocess",
        case="voxelize_ordinary",
        implementation="Open3D voxel_down_sample",
        scope="operation",
        device="cpu",
        dtype="float64",
        parameters={"voxel_size": args.voxel, "threads": "uncontrolled"},
        timing=timing,
        detail={
            "input_points": len(target.points),
            "output_points": len(downsampled.points),
        },
    )
    _, timing = timed(
        lambda: downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(
                knn=args.normal_k
            )
        ),
        args.repeats,
    )
    add_row(
        rows,
        task="preprocess",
        case="estimate_normals_ordinary",
        implementation="Open3D estimate_normals",
        scope="operation",
        device="cpu",
        dtype="float64",
        parameters={"k": args.normal_k, "threads": "uncontrolled"},
        timing=timing,
        detail={"input_points": len(downsampled.points)},
    )


def benchmark_small_gicp_registration(
    rows: list[BenchmarkRow],
    expected: np.ndarray,
    args: argparse.Namespace,
) -> None:
    try:
        import small_gicp
    except ImportError:
        print("small_gicp is unavailable; skipping its comparison rows")
        return
    source = small_gicp.voxelgrid_sampling(
        small_gicp.read_ply(str(DATA_DIR / "source.ply")), args.voxel
    )
    target = small_gicp.voxelgrid_sampling(
        small_gicp.read_ply(str(DATA_DIR / "target.ply")), args.voxel
    )
    small_gicp.estimate_normals_covariances(
        source, num_threads=args.threads
    )
    small_gicp.estimate_normals_covariances(
        target, num_threads=args.threads
    )
    for method in ("ICP", "PLANE_ICP", "GICP"):

        def run(method=method):
            tree = small_gicp.KdTree(
                target, num_threads=args.threads
            )
            return small_gicp.align(
                target,
                source,
                tree,
                registration_type=method,
                max_correspondence_distance=args.max_corr_dist,
                max_iterations=args.max_iters,
                num_threads=args.threads,
            )

        result, timing = timed(run, args.repeats)
        rotation_error, translation_error = pose_errors(
            result.T_target_source, expected
        )
        add_row(
            rows,
            task="registration",
            case=f"single_level_{method.lower()}",
            implementation="small_gicp",
            scope="alignment+index-build",
            device="cpu",
            dtype="implementation-default",
            parameters={
                "registration_type": method,
                "threads": args.threads,
                "max_correspondence_distance": args.max_corr_dist,
                "max_iterations": args.max_iters,
            },
            timing=timing,
            detail={
                "rotation_error_degrees": rotation_error,
                "translation_error": translation_error,
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
            },
        )


def benchmark_open3d_registration(
    rows: list[BenchmarkRow],
    source_np: np.ndarray,
    target_np: np.ndarray,
    expected: np.ndarray,
    args: argparse.Namespace,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is unavailable; skipping its comparison rows")
        return

    def to_open3d(points: np.ndarray):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            points.astype(np.float64)
        )
        return cloud

    source = to_open3d(source_np).voxel_down_sample(args.voxel)
    target = to_open3d(target_np).voxel_down_sample(args.voxel)
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(
            knn=args.normal_k
        )
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=args.max_iters
    )
    methods = {
        "point_to_point": (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        ),
        "point_to_plane": (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        ),
    }
    for name, estimation in methods.items():
        result, timing = timed(
            lambda estimation=estimation: (
                o3d.pipelines.registration.registration_icp(
                    source,
                    target,
                    args.max_corr_dist,
                    np.eye(4),
                    estimation,
                    criteria,
                )
            ),
            args.repeats,
        )
        rotation_error, translation_error = pose_errors(
            np.asarray(result.transformation), expected
        )
        add_row(
            rows,
            task="registration",
            case=f"single_level_{name}",
            implementation="Open3D registration_icp",
            scope="alignment-only",
            device="cpu",
            dtype="float64",
            parameters={
                "objective": name,
                "threads": "uncontrolled",
                "max_correspondence_distance": args.max_corr_dist,
                "max_iterations": args.max_iters,
            },
            timing=timing,
            detail={
                "rotation_error_degrees": rotation_error,
                "translation_error": translation_error,
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
            },
        )


def command_output(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    revision = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    driver = command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_bytes": properties.total_memory,
                }
            )
    argument_values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": argument_values,
        "git_revision": revision,
        "git_dirty": status is not None,
        "torchpcl_version": torchpcl.__version__,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_build_version": torch.version.cuda,
        "cuda_driver_version": driver,
        "cuda_available": torch.cuda.is_available(),
        "extension_has_cuda": bool(_C.has_cuda()),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "pytorch_thread_count": torch.get_num_threads(),
        "gpus": gpus,
    }


def resolve_devices(
    requested: str,
) -> list[torch.device]:
    cuda_usable = torch.cuda.is_available() and _C.has_cuda()
    if requested == "cpu":
        return [torch.device("cpu")]
    if requested == "cuda":
        if not cuda_usable:
            raise ValueError(
                "CUDA was requested, but PyTorch CUDA and a CUDA-enabled "
                "torchpcl extension are not both available"
            )
        return [torch.device("cuda")]
    devices = [torch.device("cpu")]
    if cuda_usable:
        devices.append(torch.device("cuda"))
    return devices


def detail_text(detail: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={json.dumps(value, separators=(',', ':'))}"
        for key, value in detail.items()
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    header = (
        f"{'task':<13} {'case':<48} {'impl':<28} "
        f"{'median':>10} {'iqr':>10}  detail"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.task:<13} {row.case:<48} "
            f"{row.implementation:<28} "
            f"{row.median_seconds * 1e3:>8.2f}ms "
            f"{row.iqr_seconds * 1e3:>8.2f}ms  "
            f"{detail_text(row.detail)}"
        )


def write_json(
    path: Path,
    metadata: dict[str, Any],
    rows: list[BenchmarkRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "results": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "search", "preprocess", "metrics", "registration"),
        default="all",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--comparisons", action="store_true")

    parser.add_argument("--voxel", type=float, default=0.25)
    parser.add_argument("--fine-voxel", type=float, default=0.05)
    parser.add_argument("--normal-k", type=int, default=30)
    parser.add_argument("--eigh-matrices", type=int, default=30_000)

    parser.add_argument(
        "--search-sizes",
        type=int,
        nargs="+",
        default=[512, 2048, 8192],
    )
    parser.add_argument("--search-queries", type=int, default=1024)
    parser.add_argument(
        "--search-k",
        type=int,
        nargs="+",
        default=[1, 30],
    )

    parser.add_argument("--packed-batches", type=int, default=32)
    parser.add_argument("--packed-points", type=int, default=256)
    parser.add_argument("--packed-queries", type=int, default=64)
    parser.add_argument("--metric-max-points", type=int, default=4096)

    parser.add_argument("--max-corr-dist", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument(
        "--level-voxels",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25],
    )
    parser.add_argument(
        "--level-distances",
        type=float,
        nargs="+",
        default=[2.0, 1.0, 0.5],
    )
    parser.add_argument(
        "--level-iterations",
        type=int,
        nargs="+",
        default=[30, 20, 15],
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.voxel <= 0 or args.fine_voxel <= 0:
        parser.error("--voxel and --fine-voxel must be positive")
    if not 3 <= args.normal_k <= 64:
        parser.error("--normal-k must be in [3, 64]")
    if args.eigh_matrices < 1:
        parser.error("--eigh-matrices must be positive")
    if (
        args.search_queries < 1
        or any(size < 1 for size in args.search_sizes)
    ):
        parser.error(
            "--search-queries and every --search-sizes value must be positive"
        )
    if any(not 1 <= k <= 64 for k in args.search_k):
        parser.error("every --search-k value must be in [1, 64]")
    if min(
        args.packed_batches,
        args.packed_points,
        args.packed_queries,
        args.metric_max_points,
    ) < 1:
        parser.error("packed sizes and --metric-max-points must be positive")
    if args.max_corr_dist <= 0 or args.max_iters < 0:
        parser.error(
            "--max-corr-dist must be positive and --max-iters nonnegative"
        )
    level_lengths = {
        len(args.level_voxels),
        len(args.level_distances),
        len(args.level_iterations),
    }
    if len(level_lengths) != 1 or not args.level_voxels:
        parser.error("ICP level option lists must have the same nonzero length")
    if any(value <= 0 for value in args.level_voxels):
        parser.error("every --level-voxels value must be positive")
    if any(value <= 0 for value in args.level_distances):
        parser.error("every --level-distances value must be positive")
    if any(value < 0 for value in args.level_iterations):
        parser.error("every --level-iterations value must be nonnegative")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    try:
        devices = resolve_devices(args.device)
    except ValueError as error:
        parser.error(str(error))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_num_threads(args.threads)

    needs_scan = args.task != "search"
    source = target = expected = None
    if needs_scan:
        source, target, expected = load_inputs()
        print(
            f"source={len(source)} target={len(target)} "
            f"dtype={args.dtype} repeats={args.repeats} "
            f"threads={torch.get_num_threads()} "
            f"devices={','.join(str(device) for device in devices)}\n"
        )
    else:
        print(
            f"synthetic search dtype={args.dtype} repeats={args.repeats} "
            f"threads={torch.get_num_threads()} "
            f"devices={','.join(str(device) for device in devices)}\n"
        )

    rows: list[BenchmarkRow] = []
    if args.task in {"all", "search"}:
        for device in devices:
            benchmark_search(rows, args, device, dtype)

    if args.task in {"all", "preprocess"}:
        assert target is not None
        for device in devices:
            benchmark_preprocess(rows, target, args, device, dtype)
        if args.comparisons and torch.device("cpu") in devices:
            benchmark_small_gicp_preprocess(rows, args)
            benchmark_open3d_preprocess(rows, target, args)

    if args.task in {"all", "metrics"}:
        assert source is not None and target is not None
        for device in devices:
            benchmark_metrics(rows, source, target, args, device, dtype)

    if args.task in {"all", "registration"}:
        assert source is not None and target is not None and expected is not None
        for device in devices:
            benchmark_registration(
                rows,
                source,
                target,
                expected,
                args,
                device,
                dtype,
            )
        if args.comparisons and torch.device("cpu") in devices:
            benchmark_small_gicp_registration(rows, expected, args)
            benchmark_open3d_registration(
                rows, source, target, expected, args
            )

    metadata = environment_metadata(args)
    print_rows(rows)
    if args.json is not None:
        write_json(args.json, metadata, rows)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
