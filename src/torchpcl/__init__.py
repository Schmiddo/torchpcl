"""Point cloud registration and processing built on torch and cuBQL."""

from .alignment import ProcrustesResult, procrustes
from .cloud import PointCloud, PointCloudLike, as_point_cloud, cat, cat_batch
from .filters import (
    radius_outliers,
    statistical_outliers,
)
from .feature_matching import CorrespondenceSet, match_features
from .features import FPFHResult, fpfh
from .global_registration import (
    FGROptions,
    FGRResult,
    fast_global_registration,
    fast_global_registration_from_correspondences,
)
from .io import load_ply, save_ply
from .local_geometry import (
    CovarianceResult,
    covariance_from_neighbors,
    estimate_covariances,
)
from .metrics import (
    FScoreResult,
    PointCloudMetrics,
    chamfer_distance,
    fscore,
    point_cloud_metrics,
)
from .neighbors import (
    NeighborIndex,
    Neighbors,
    hybrid_neighbors,
    knn,
    radius_neighbors,
)
from .normals import (
    NormalResult,
    estimate_normals,
    orient_normals_to_direction,
    orient_normals_toward_viewpoint,
)
from .registration import (
    ConvergenceCriteria,
    HuberLoss,
    ICPLevel,
    ICPLevelResult,
    ICPOptions,
    ICPResult,
    PointToPlane,
    PointToPoint,
    RegistrationMetrics,
    evaluate_registration,
    icp,
)
from .transforms import transform
from .voxel import Voxelization, voxelize

__version__ = "0.2.0"

__all__ = [
    "ConvergenceCriteria",
    "CorrespondenceSet",
    "CovarianceResult",
    "FGROptions",
    "FGRResult",
    "FPFHResult",
    "FScoreResult",
    "HuberLoss",
    "ICPLevel",
    "ICPLevelResult",
    "ICPOptions",
    "ICPResult",
    "NeighborIndex",
    "Neighbors",
    "NormalResult",
    "PointCloud",
    "PointCloudLike",
    "PointCloudMetrics",
    "PointToPlane",
    "PointToPoint",
    "ProcrustesResult",
    "RegistrationMetrics",
    "Voxelization",
    "as_point_cloud",
    "cat",
    "cat_batch",
    "chamfer_distance",
    "covariance_from_neighbors",
    "estimate_covariances",
    "estimate_normals",
    "evaluate_registration",
    "fast_global_registration",
    "fast_global_registration_from_correspondences",
    "fpfh",
    "fscore",
    "hybrid_neighbors",
    "icp",
    "knn",
    "load_ply",
    "match_features",
    "point_cloud_metrics",
    "procrustes",
    "orient_normals_to_direction",
    "orient_normals_toward_viewpoint",
    "radius_neighbors",
    "radius_outliers",
    "save_ply",
    "statistical_outliers",
    "transform",
    "voxelize",
    "__version__",
]
