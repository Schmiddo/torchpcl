"""Point cloud registration and processing built on torch and cuBQL."""

from .alignment import ProcrustesResult, procrustes
from .cloud import PointCloud, PointCloudLike, as_point_cloud
from .metrics import (
    FScoreResult,
    PointCloudMetrics,
    chamfer_distance,
    fscore,
    point_cloud_metrics,
)
from .multiscale import ICPScale, PointCloudPyramid, build_pyramid, multiscale_icp
from .neighbors import (
    NeighborIndex,
    Neighbors,
    hybrid_neighbors,
    knn,
    radius_neighbors,
)
from .normals import NormalResult, estimate_normals
from .registration import evaluate_registration, icp
from .types import ICPResult, RegistrationMetrics
from .transforms import transform
from .voxel import Voxelization, voxelize

__version__ = "0.2.0.dev0"

__all__ = [
    "FScoreResult",
    "ICPResult",
    "ICPScale",
    "NeighborIndex",
    "Neighbors",
    "NormalResult",
    "PointCloud",
    "PointCloudLike",
    "PointCloudMetrics",
    "PointCloudPyramid",
    "ProcrustesResult",
    "RegistrationMetrics",
    "Voxelization",
    "as_point_cloud",
    "chamfer_distance",
    "build_pyramid",
    "estimate_normals",
    "evaluate_registration",
    "fscore",
    "hybrid_neighbors",
    "icp",
    "knn",
    "multiscale_icp",
    "point_cloud_metrics",
    "procrustes",
    "radius_neighbors",
    "transform",
    "voxelize",
    "__version__",
]
