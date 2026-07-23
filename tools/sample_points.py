#!/usr/bin/env python3
"""Convert static geometry to a point cloud using Trimesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def load_geometry(
    path: Path,
) -> trimesh.Trimesh | trimesh.Scene | trimesh.points.PointCloud:
    """Load a triangle mesh or point cloud with Trimesh."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in {path}")
        if not all(
            isinstance(geometry, trimesh.Trimesh)
            for geometry in loaded.geometry.values()
        ):
            types = sorted(
                {type(geometry).__name__ for geometry in loaded.geometry.values()}
            )
            raise TypeError(f"Unsupported scene geometry types: {', '.join(types)}")
        return loaded
    if not isinstance(loaded, (trimesh.Trimesh, trimesh.points.PointCloud)):
        raise TypeError(f"Unsupported geometry type: {type(loaded).__name__}")
    return loaded


def _cloud_colors(cloud: trimesh.points.PointCloud) -> np.ndarray | None:
    colors = np.asarray(cloud.colors)
    return colors if len(colors) == len(cloud.vertices) else None


def _sample_scene_surface(
    scene: trimesh.Scene, num_points: int, seed: int | None
) -> trimesh.points.PointCloud:
    """Uniformly sample a scene without discarding per-geometry materials."""
    meshes = [
        geometry
        for geometry in scene.dump()
        if isinstance(geometry, trimesh.Trimesh) and geometry.faces.size > 0
    ]
    if not meshes:
        raise ValueError("Scene contains no triangle meshes")

    areas = np.asarray([mesh.area for mesh in meshes], dtype=float)
    total_area = areas.sum()
    if not np.isfinite(areas).all() or total_area <= 0:
        raise ValueError("Scene meshes must have finite, positive surface area")

    random = np.random.default_rng(seed)
    counts = random.multinomial(num_points, areas / total_area)
    clouds = [
        sample_surface(
            mesh,
            int(count),
            seed=int(random.integers(0, np.iinfo(np.int64).max)),
        )
        for mesh, count in zip(meshes, counts)
        if count > 0
    ]

    colors = None
    if all(_cloud_colors(cloud) is not None for cloud in clouds):
        colors = np.concatenate([cloud.colors for cloud in clouds])
    return trimesh.points.PointCloud(
        vertices=np.concatenate([cloud.vertices for cloud in clouds]),
        colors=colors,
    )


def sample_surface(
    mesh: trimesh.Trimesh | trimesh.Scene,
    num_points: int,
    seed: int | None,
) -> trimesh.points.PointCloud:
    """Uniformly sample positions and colors on a mesh."""
    if isinstance(mesh, trimesh.Scene):
        return _sample_scene_surface(mesh, num_points, seed)

    points, face_indices = trimesh.sample.sample_surface(
        mesh, num_points, seed=seed
    )
    colors = sample_colors(mesh, points, face_indices)
    return trimesh.points.PointCloud(vertices=points, colors=colors)


def downsample_cloud(
    cloud: trimesh.points.PointCloud,
    num_points: int,
    seed: int | None,
) -> trimesh.points.PointCloud:
    """Randomly retain at most num_points and matching colors."""
    if len(cloud.vertices) <= num_points:
        return cloud
    indices = np.random.default_rng(seed).choice(
        len(cloud.vertices), size=num_points, replace=False, shuffle=False
    )
    colors = _cloud_colors(cloud)
    if colors is not None:
        colors = colors[indices]
    return trimesh.points.PointCloud(vertices=cloud.vertices[indices], colors=colors)


def sample_colors(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray | None:
    """Sample RGBA colors for points already sampled on mesh faces."""
    visual = mesh.visual
    if visual is None or not getattr(visual, "defined", visual.kind is not None):
        return None

    if visual.kind == "face":
        return np.asarray(visual.face_colors[face_indices], dtype=np.uint8)

    triangles = mesh.triangles[face_indices]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
    vertex_indices = mesh.faces[face_indices]

    if visual.kind == "vertex":
        corner_colors = np.asarray(visual.vertex_colors[vertex_indices], dtype=float)
        colors = np.einsum("ni,nic->nc", barycentric, corner_colors)
        return np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    if visual.kind == "texture":
        uv = np.asarray(visual.uv)
        if uv.shape != (len(mesh.vertices), 2):
            raise ValueError("Textured mesh must have one UV coordinate per vertex.")
        sampled_uv = np.einsum("ni,nic->nc", barycentric, uv[vertex_indices])
        return sample_material_colors(visual, sampled_uv, face_indices)

    return None


def sample_material_colors(
    visual: trimesh.visual.TextureVisuals,
    uv: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    """Sample a texture visual, including visuals with multiple materials."""
    material = visual.material
    face_materials = visual.face_materials

    if face_materials is None or not hasattr(material, "get"):
        return material_colors(material, uv)

    material_indices = np.asarray(face_materials)[face_indices]
    colors = np.empty((len(uv), 4), dtype=np.uint8)
    for material_index in np.unique(material_indices):
        mask = material_indices == material_index
        colors[mask] = material_colors(material.get(int(material_index)), uv[mask])
    return colors


def material_colors(material: object, uv: np.ndarray) -> np.ndarray:
    """Return one uint8 RGBA material color per UV coordinate."""
    colors = material.to_color(uv) if hasattr(material, "to_color") else None
    if colors is None:
        colors = getattr(material, "main_color", None)
    if colors is None:
        raise ValueError("Texture material does not provide sampleable colors.")

    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = np.broadcast_to(colors, (len(uv), colors.shape[0]))
    if colors.shape == (len(uv), 3):
        colors = np.column_stack((colors, np.full(len(uv), 255)))
    if colors.shape != (len(uv), 4):
        raise ValueError("Texture material colors must have shape (N, 3) or (N, 4).")
    if np.issubdtype(colors.dtype, np.floating) and colors.size and colors.max() <= 1:
        colors = colors * 255
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_geometry", type=Path, help="Input mesh or point cloud")
    parser.add_argument("output_cloud", type=Path, help="Output point-cloud file")
    parser.add_argument(
        "-n",
        "--num-points",
        type=positive_int,
        default=100_000,
        help="mesh sample count or maximum cloud size (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducible sampling"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry = load_geometry(args.input_geometry)
    if isinstance(geometry, trimesh.points.PointCloud):
        original_count = len(geometry.vertices)
        cloud = downsample_cloud(geometry, args.num_points, args.seed)
        action = "Downsampled" if len(cloud.vertices) < original_count else "Copied"
    else:
        cloud = sample_surface(geometry, args.num_points, args.seed)
        action = "Sampled"

    cloud.export(args.output_cloud)
    print(
        f"{action} {len(cloud.vertices):,} points from {args.input_geometry} "
        f"to {args.output_cloud}"
    )


if __name__ == "__main__":
    main()
