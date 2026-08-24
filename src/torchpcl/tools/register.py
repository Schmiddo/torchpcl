#!/usr/bin/env python3
"""Register two or more PLY point clouds into one consistent point cloud."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import torchpcl as tp


_RGB_NAMES = ("red", "green", "blue")


@dataclass(frozen=True)
class _PairRegistration:
    source: int
    target: int
    transform: torch.Tensor
    fitness: float
    inlier_rmse: float
    success: bool


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


def _colors_as_uint8(colors: torch.Tensor) -> torch.Tensor:
    if colors.dtype == torch.uint8:
        return colors
    values = colors
    if colors.dtype.is_floating_point and colors.numel() > 0:
        if bool((colors >= 0).all()) and bool((colors <= 1).all()):
            values = colors * 255
        values = values.round()
    return values.clamp(0, 255).to(torch.uint8)


def _random_solid_colors(
    point_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    color = torch.randint(
        32,
        256,
        (3,),
        dtype=torch.int64,
        generator=generator,
    ).to(device=device, dtype=torch.uint8)
    return color.expand(point_count, -1)


def _combine_clouds_with_colors(
    clouds: Sequence[tp.PointCloud],
    *,
    color_seed: int = 0,
) -> tp.PointCloud:
    generator = torch.Generator(device="cpu").manual_seed(color_seed)
    colors = [
        _random_solid_colors(cloud.points.shape[0], generator, cloud.device)
        if cloud.features is None
        else _colors_as_uint8(cloud.features)
        for cloud in clouds
    ]
    return tp.PointCloud.from_points(
        torch.cat([cloud.points for cloud in clouds]),
        features=torch.cat(colors),
    )


def _combine_with_colors(
    source: tp.PointCloud,
    target: tp.PointCloud,
) -> tp.PointCloud:
    return _combine_clouds_with_colors((source, target))


def _maximum_spanning_tree(
    edges: Sequence[_PairRegistration],
    cloud_count: int,
) -> list[_PairRegistration]:
    parents = list(range(cloud_count))

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    result: list[_PairRegistration] = []
    candidates = (
        edge
        for edge in edges
        if edge.success and math.isfinite(edge.fitness)
    )
    for edge in sorted(
        candidates,
        key=lambda item: (-item.fitness, item.source, item.target),
    ):
        source_root = find(edge.source)
        target_root = find(edge.target)
        if source_root == target_root:
            continue
        parents[source_root] = target_root
        result.append(edge)
        if len(result) == cloud_count - 1:
            return result
    raise ValueError("global registration could not connect all input clouds")


def _propagate_poses(
    edges: Sequence[_PairRegistration],
    cloud_count: int,
    reference: int,
) -> torch.Tensor:
    adjacency: list[list[tuple[int, _PairRegistration]]] = [
        [] for _ in range(cloud_count)
    ]
    for edge in edges:
        adjacency[edge.source].append((edge.target, edge))
        adjacency[edge.target].append((edge.source, edge))

    sample = edges[0].transform
    poses: list[torch.Tensor | None] = [None] * cloud_count
    poses[reference] = torch.eye(4, dtype=sample.dtype, device=sample.device)
    pending = [reference]
    while pending:
        node = pending.pop()
        assert poses[node] is not None
        for neighbor, edge in adjacency[node]:
            if poses[neighbor] is not None:
                continue
            if node == edge.target:
                poses[neighbor] = poses[node] @ edge.transform
            else:
                poses[neighbor] = poses[node] @ torch.linalg.inv(edge.transform)
            pending.append(neighbor)
    if any(pose is None for pose in poses):
        raise ValueError("registration tree does not connect all input clouds")
    return torch.stack([pose for pose in poses if pose is not None])


def _unique_stems(paths: Sequence[Path]) -> list[str]:
    next_suffix: dict[str, int] = {}
    used: set[str] = set()
    result = []
    for path in paths:
        stem = path.stem
        key = stem
        suffix = next_suffix.get(stem, 1)
        while key in used:
            key = f"{stem}-{suffix}"
            suffix += 1
        next_suffix[stem] = suffix
        used.add(key)
        result.append(key)
    return result


def _ownership_masks(
    paths: Sequence[Path],
    clouds: Sequence[tp.PointCloud],
) -> dict[str, np.ndarray]:
    lengths = [cloud.points.shape[0] for cloud in clouds]
    total = sum(lengths)
    result = {}
    offset = 0
    for key, length in zip(_unique_stems(paths), lengths, strict=True):
        mask = np.zeros(total, dtype=np.bool_)
        mask[offset : offset + length] = True
        result[key] = mask
        offset += length
    return result


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


def _global_registration_edges(
    paths: Sequence[Path],
    described: Sequence[tp.PointCloud],
    *,
    max_correspondence_distance: float,
    max_iterations: int,
) -> list[_PairRegistration]:
    edges = []
    for source in range(len(described)):
        for target in range(source + 1, len(described)):
            result = tp.fast_global_registration(
                described[source],
                described[target],
                options=tp.FGROptions(
                    max_correspondence_distance=max_correspondence_distance,
                    max_iterations=max_iterations,
                ),
            )
            edge = _PairRegistration(
                source=source,
                target=target,
                transform=result.transforms[0],
                fitness=float(result.fitness.item()),
                inlier_rmse=float(result.inlier_rmse.item()),
                success=bool(result.success.item()),
            )
            edges.append(edge)
            print(
                "Global registration: "
                f"success={edge.success} "
                f"source={paths[source].name} "
                f"target={paths[target].name} "
                f"correspondences={result.correspondence_counts.item()} "
                f"fitness={edge.fitness:.6g} "
                f"inlier_rmse={edge.inlier_rmse:.6g}"
            )
    return edges


def _identity_chain(
    clouds: Sequence[tp.PointCloud],
) -> list[_PairRegistration]:
    return [
        _PairRegistration(
            source=index,
            target=index + 1,
            transform=torch.eye(
                4, dtype=clouds[index].dtype, device=clouds[index].device
            ),
            fitness=0.0,
            inlier_rmse=math.inf,
            success=True,
        )
        for index in range(len(clouds) - 1)
    ]


def _refine_tree_edges(
    paths: Sequence[Path],
    clouds: Sequence[tp.PointCloud],
    edges: Sequence[_PairRegistration],
    *,
    max_correspondence_distance: float,
    max_iterations: int,
    normal_radius: float,
    max_neighbors: int,
    point_to_plane: bool,
) -> list[_PairRegistration]:
    if point_to_plane:
        icp_clouds = []
        for cloud in clouds:
            if cloud.normals is None:
                normals = tp.estimate_normals(
                    cloud,
                    radius=normal_radius,
                    k=min(30, max_neighbors),
                )
                cloud = cloud.with_normals(normals.normals)
            icp_clouds.append(cloud)
        objective: tp.PointToPoint | tp.PointToPlane = tp.PointToPlane()
    else:
        icp_clouds = list(clouds)
        objective = tp.PointToPoint()

    refined = []
    for edge in edges:
        result = tp.icp(
            icp_clouds[edge.source],
            icp_clouds[edge.target],
            [
                tp.ICPLevel(
                    max_correspondence_distance=max_correspondence_distance,
                    max_iterations=max_iterations,
                )
            ],
            init=edge.transform,
            options=tp.ICPOptions(objective=objective),
        )
        updated = _PairRegistration(
            source=edge.source,
            target=edge.target,
            transform=result.transforms[0],
            fitness=float(result.fitness.item()),
            inlier_rmse=float(result.inlier_rmse.item()),
            success=True,
        )
        refined.append(updated)
        print(
            "ICP tree edge: "
            f"source={paths[edge.source].name} "
            f"target={paths[edge.target].name} "
            f"fitness={updated.fitness:.9g} "
            f"inlier_rmse={updated.inlier_rmse:.9g}"
        )
    return refined


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="PLY inputs; the final input defines the output reference frame",
    )
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
        help="optional combined PLY; ownership masks use the same path with .npz",
    )
    return parser


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if len(args.inputs) < 2:
        parser.error("at least two input PLY files are required")
    device = _resolve_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    paths = args.inputs
    clouds = [
        _load_ply_with_colors(path, dtype=dtype, device=device) for path in paths
    ]

    normal_radius = args.normal_radius or 2.0 * args.voxel_size
    feature_radius = args.feature_radius or 5.0 * args.voxel_size
    fgr_distance = args.fgr_distance or 1.5 * args.voxel_size
    icp_distance = args.icp_distance or 1.5 * args.voxel_size

    if args.global_registration:
        described = [
            _describe(
                cloud,
                voxel_size=args.voxel_size,
                normal_radius=normal_radius,
                feature_radius=feature_radius,
                max_neighbors=args.max_neighbors,
            )
            for cloud in clouds
        ]
        candidates = _global_registration_edges(
            paths,
            described,
            max_correspondence_distance=fgr_distance,
            max_iterations=args.fgr_iterations,
        )
        tree = _maximum_spanning_tree(candidates, len(clouds))
    else:
        tree = _identity_chain(clouds)

    refined = _refine_tree_edges(
        paths,
        clouds,
        tree,
        max_correspondence_distance=icp_distance,
        max_iterations=args.icp_iterations,
        normal_radius=normal_radius,
        max_neighbors=args.max_neighbors,
        point_to_plane=args.icp_objective == "point-to-plane",
    )
    poses = _propagate_poses(refined, len(clouds), reference=len(clouds) - 1)

    if len(clouds) == 2:
        print("Final transformation (source -> target):")
        print(poses[0].detach().cpu())
        print(f"Fitness: {refined[0].fitness:.9g}")
        print(
            "Inlier RMSE (registration error): "
            f"{refined[0].inlier_rmse:.9g}"
        )
    else:
        print(f"Final transformations (input -> reference {paths[-1].name}):")
        for key, pose in zip(_unique_stems(paths), poses, strict=True):
            print(f"{key}:")
            print(pose.detach().cpu())

    if args.output is not None:
        transformed = []
        for cloud, pose in zip(clouds, poses, strict=True):
            aligned = tp.transform(cloud, pose)
            assert isinstance(aligned, tp.PointCloud)
            transformed.append(aligned)
        combined = _combine_clouds_with_colors(transformed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tp.save_ply(args.output, combined, feature_names=_RGB_NAMES)
        print(f"Combined point cloud: {args.output}")
        mask_path = args.output.with_suffix(".npz")
        np.savez(mask_path, **_ownership_masks(paths, transformed))
        print(f"Point ownership masks: {mask_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
