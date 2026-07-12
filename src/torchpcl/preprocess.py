"""Compatibility imports for point-cloud preprocessing."""

from .normals import NormalResult, estimate_normals
from .voxel import Voxelization, voxel_downsample, voxelize

__all__ = [
    "NormalResult",
    "Voxelization",
    "estimate_normals",
    "voxel_downsample",
    "voxelize",
]
