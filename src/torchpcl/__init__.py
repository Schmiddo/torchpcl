"""Point cloud registration and processing built on torch and cuBQL."""

from .estimation import PointToPlane, PointToPoint, TransformationEstimation
from .metrics import PointCloudMetrics, point_cloud_metrics
from .preprocess import estimate_normals, voxel_downsample
from .registration import evaluate_registration, icp
from .types import ICPConvergenceCriteria, RegistrationResult

__version__ = "0.1.0"

__all__ = [
    "ICPConvergenceCriteria",
    "PointCloudMetrics",
    "PointToPlane",
    "PointToPoint",
    "RegistrationResult",
    "TransformationEstimation",
    "estimate_normals",
    "evaluate_registration",
    "icp",
    "point_cloud_metrics",
    "voxel_downsample",
    "__version__",
]
