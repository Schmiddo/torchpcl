#!/usr/bin/env python3
"""Register a source PLY point cloud to a target PLY point cloud."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import torch

import torchpcl as tp


_RGB_NAMES = ("red", "green", "blue")


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _neighbor_count(value: str) -> int:
    result = int(value)
    if not 3 <= result <= 64:
        raise argparse.ArgumentTypeError("must be in [3, 64]")
    return result


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        from torchpcl import _C

        requested = "cuda" if torch.cuda.is_available() and _C.has_cuda() else "cpu"

    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("torchpcl search supports CPU and CUDA devices")
    if device.type == "cuda":
        from torchpcl import _C

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but PyTorch cannot access it")
        if not _C.has_cuda():
            raise ValueError("CUDA was requested, but torchpcl was built without it")
    return device


def _load_ply_with_colors(
    path: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tp.PointCloud:
    """Load standard PLY RGB properties when all three are present."""
    try:
        cloud = tp.load_ply(path, dtype=dtype, feature_names=_RGB_NAMES)
    except ValueError as error:
        if "missing feature properties" not in str(error):
            raise
        cloud = tp.load_ply(path, dtype=dtype)
    return cloud.to(device)


def _default_colors(
    point_count: int,
    reference: torch.Tensor | None,
    channel: int,
    device: torch.device,
) -> torch.Tensor:
    if reference is None:
        dtype = torch.uint8
        intensity = 255
    elif reference.dtype.is_floating_point:
        dtype = reference.dtype
        intensity = 1.0
    else:
        dtype = reference.dtype
        intensity = torch.iinfo(reference.dtype).max
    colors = torch.zeros(
        (point_count, 3),
        dtype=dtype,
        device=device,
    )
    colors[:, channel] = intensity
    return colors


def _combine_with_colors(
    source: tp.PointCloud,
    target: tp.PointCloud,
) -> tp.PointCloud:
    source_colors = source.features
    target_colors = target.features
    reference = source_colors if source_colors is not None else target_colors
    if source_colors is None:
        source_colors = _default_colors(
            source.points.shape[0], reference, 0, source.device
        )
    if target_colors is None:
        target_colors = _default_colors(
            target.points.shape[0], reference, 2, target.device
        )
    colors = torch.cat((source_colors, target_colors), dim=0)
    return tp.PointCloud.from_points(
        torch.cat((source.points, target.points), dim=0),
        features=colors,
    )


def _describe(
    cloud: tp.PointCloud,
    *,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
    max_neighbors: int,
) -> tp.PointCloud:
    cloud = tp.voxelize(cloud, voxel_size).cloud
    normal_result = tp.estimate_normals(
        cloud,
        radius=normal_radius,
        k=min(30, max_neighbors),
    )
    cloud = cloud.with_normals(normal_result.normals)

    # The centroid transforms with the cloud, so this orientation convention is
    # consistent between source and target without requiring camera metadata.
    viewpoint = cloud.points.mean(dim=0)
    cloud = tp.orient_normals_toward_viewpoint(cloud, viewpoint)
    feature_result = tp.fpfh(
        cloud,
        feature_radius,
        max_neighbors=max_neighbors,
    )
    if feature_result.valid.count_nonzero().item() < 10:
        raise ValueError(
            "global registration needs at least 10 valid FPFH rows; "
            "increase the radii or use --no-global-registration"
        )
    return cloud.select_points(feature_result.valid).with_features(
        feature_result.descriptors[feature_result.valid]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source PLY to transform")
    parser.add_argument("target", type=Path, help="target PLY to align to")
    parser.add_argument(
        "--global-registration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run FPFH/FGR before ICP (default: enabled)",
    )
    parser.add_argument(
        "--voxel-size",
        type=_positive_float,
        default=0.05,
        help="global-registration voxel size (default: %(default)s)",
    )
    parser.add_argument(
        "--normal-radius",
        type=_positive_float,
        help="normal radius (default: 2 * voxel size)",
    )
    parser.add_argument(
        "--feature-radius",
        type=_positive_float,
        help="FPFH radius (default: 5 * voxel size)",
    )
    parser.add_argument(
        "--fgr-distance",
        type=_positive_float,
        help="FGR correspondence distance (default: 1.5 * voxel size)",
    )
    parser.add_argument(
        "--fgr-iterations",
        type=_nonnegative_int,
        default=64,
        help="FGR iteration budget (default: %(default)s)",
    )
    parser.add_argument(
        "--max-neighbors",
        type=_neighbor_count,
        default=64,
        help="normal/FPFH neighbor cap in [3, 64] (default: %(default)s)",
    )
    parser.add_argument(
        "--icp-distance",
        type=_positive_float,
        help="ICP correspondence distance (default: 1.5 * voxel size)",
    )
    parser.add_argument(
        "--icp-iterations",
        type=_nonnegative_int,
        default=50,
        help="ICP iteration budget (default: %(default)s)",
    )
    parser.add_argument(
        "--icp-objective",
        choices=("point-to-point", "point-to-plane"),
        default="point-to-point",
        help="ICP objective (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, or auto (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="geometry dtype (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="optional PLY for transformed source combined with target",
    )
    return parser


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    source = _load_ply_with_colors(
        args.source,
        dtype=dtype,
        device=device,
    )
    target = _load_ply_with_colors(
        args.target,
        dtype=dtype,
        device=device,
    )

    normal_radius = args.normal_radius or 2.0 * args.voxel_size
    feature_radius = args.feature_radius or 5.0 * args.voxel_size
    fgr_distance = args.fgr_distance or 1.5 * args.voxel_size
    icp_distance = args.icp_distance or 1.5 * args.voxel_size

    initial_transform = None
    if args.global_registration:
        source_features = _describe(
            source,
            voxel_size=args.voxel_size,
            normal_radius=normal_radius,
            feature_radius=feature_radius,
            max_neighbors=args.max_neighbors,
        )
        target_features = _describe(
            target,
            voxel_size=args.voxel_size,
            normal_radius=normal_radius,
            feature_radius=feature_radius,
            max_neighbors=args.max_neighbors,
        )
        coarse = tp.fast_global_registration(
            source_features,
            target_features,
            options=tp.FGROptions(
                max_correspondence_distance=fgr_distance,
                max_iterations=args.fgr_iterations,
            ),
        )
        initial_transform = coarse.transforms
        print(
            "Global registration: "
            f"success={coarse.success.item()} "
            f"correspondences={coarse.correspondence_counts.item()} "
            f"fitness={coarse.fitness.item():.6g} "
            f"inlier_rmse={coarse.inlier_rmse.item():.6g}"
        )

    if args.icp_objective == "point-to-plane":
        if target.normals is None:
            normals = tp.estimate_normals(
                target,
                radius=normal_radius,
                k=min(30, args.max_neighbors),
            )
            target = target.with_normals(normals.normals)
        objective: tp.PointToPoint | tp.PointToPlane = tp.PointToPlane()
    else:
        objective = tp.PointToPoint()

    result = tp.icp(
        source,
        target,
        [
            tp.ICPLevel(
                max_correspondence_distance=icp_distance,
                max_iterations=args.icp_iterations,
            )
        ],
        init=initial_transform,
        options=tp.ICPOptions(objective=objective),
    )

    print("Final transformation (source -> target):")
    print(result.transforms[0].detach().cpu())
    print(f"Fitness: {result.fitness.item():.9g}")
    print(f"Inlier RMSE (registration error): {result.inlier_rmse.item():.9g}")

    if args.output is not None:
        transformed = tp.transform(source, result.transforms)
        assert isinstance(transformed, tp.PointCloud)
        combined = _combine_with_colors(transformed, target)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tp.save_ply(args.output, combined, feature_names=_RGB_NAMES)
        print(f"Combined point cloud: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
