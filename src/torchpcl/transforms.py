"""Rigid transformation helpers (pure torch, device- and dtype-agnostic)."""

from typing import overload

import torch

from .cloud import PointCloud, PointCloudLike, _normalize_cloud, batch_ids


def _check_points(points: torch.Tensor, name: str) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(points.shape)}")
    if not points.dtype.is_floating_point:
        raise ValueError(f"{name} must have a floating point dtype, got {points.dtype}")


def transform_points(points: torch.Tensor, transformation: torch.Tensor) -> torch.Tensor:
    """Apply a (4, 4) rigid transformation to (N, 3) points."""
    rotation = transformation[:3, :3].to(points.dtype)
    translation = transformation[:3, 3].to(points.dtype)
    return points @ rotation.T + translation


@overload
def transform(
    cloud: torch.Tensor,
    transformations: torch.Tensor,
) -> torch.Tensor: ...


@overload
def transform(
    cloud: PointCloud,
    transformations: torch.Tensor,
) -> PointCloud: ...


def transform(
    cloud: PointCloudLike,
    transformations: torch.Tensor,
) -> torch.Tensor | PointCloud:
    """Apply rigid transforms to points or a packed point-cloud batch.

    Tensor inputs may have shape ``(N, 3)`` or ``(B, N, 3)`` and retain that
    shape. A single ``(4, 4)`` transform broadcasts over the batch; otherwise
    transformations must have shape ``(B, 4, 4)``. Attached normals are
    rotated without translation and attached features are shared. Geometry
    uses the promoted dtype of the points and transforms.
    """
    if not isinstance(transformations, torch.Tensor):
        raise TypeError("transformations must be a torch.Tensor")
    if transformations.dtype not in (torch.float32, torch.float64):
        raise ValueError("transformations must have dtype float32 or float64")

    normalized = _normalize_cloud(cloud, "cloud")
    packed = normalized.cloud
    if transformations.device != packed.device:
        raise ValueError("cloud and transformations must be on the same device")
    if transformations.shape == (4, 4):
        transformations = transformations.unsqueeze(0).expand(
            packed.batch_size, -1, -1
        )
    elif transformations.shape != (packed.batch_size, 4, 4):
        raise ValueError(
            f"transformations must have shape (4, 4) or "
            f"({packed.batch_size}, 4, 4), got {tuple(transformations.shape)}"
        )

    dtype = torch.promote_types(packed.dtype, transformations.dtype)
    points = packed.points.to(dtype)
    matrices = transformations.to(dtype)
    if packed.batch_size == 1:
        rotation = matrices[0, :3, :3]
        transformed_points = points @ rotation.T + matrices[0, :3, 3]
        transformed_normals = None
        if packed.normals is not None:
            transformed_normals = packed.normals.to(dtype) @ rotation.T
        result = PointCloud._from_validated(
            transformed_points,
            packed.offsets,
            transformed_normals,
            packed.features,
        )
    else:
        ids = batch_ids(packed.offsets, packed.points.shape[0])
        rotations = matrices[ids, :3, :3]
        translations = matrices[ids, :3, 3]
        transformed_points = torch.bmm(
            rotations, points.unsqueeze(-1)
        ).squeeze(-1)
        transformed_points = transformed_points + translations

        transformed_normals = None
        if packed.normals is not None:
            normals = packed.normals.to(dtype)
            transformed_normals = torch.bmm(
                rotations, normals.unsqueeze(-1)
            ).squeeze(-1)

        result = PointCloud._from_validated(
            transformed_points,
            packed.offsets,
            transformed_normals,
            packed.features,
        )

    if normalized.tensor_input:
        return result.points.reshape(*normalized.leading_shape, 3)
    return result


def pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Convert a 6-vector pose [rx, ry, rz, tx, ty, tz] to a (4, 4) matrix.

    The rotation is composed as R = Rz(rz) @ Ry(ry) @ Rx(rx), matching
    Open3D's PoseToTransformationImpl (ZYX euler angles).
    """
    pose = pose.to(torch.float64)
    rx, ry, rz = pose[0], pose[1], pose[2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    transformation = torch.eye(4, dtype=torch.float64, device=pose.device)
    transformation[0, 0] = cz * cy
    transformation[0, 1] = -sz * cx + cz * sy * sx
    transformation[0, 2] = sz * sx + cz * sy * cx
    transformation[1, 0] = sz * cy
    transformation[1, 1] = cz * cx + sz * sy * sx
    transformation[1, 2] = -cz * sx + sz * sy * cx
    transformation[2, 0] = -sy
    transformation[2, 1] = cy * sx
    transformation[2, 2] = cy * cx
    transformation[:3, 3] = pose[3:]
    return transformation
