"""Minimal ICP registration pipeline built on torch and warp."""

from .estimation import PointToPlane, PointToPoint, TransformationEstimation
from .registration import evaluate_registration, icp
from .types import ICPConvergenceCriteria, RegistrationResult

__version__ = "0.1.0"

__all__ = [
    "ICPConvergenceCriteria",
    "PointToPlane",
    "PointToPoint",
    "RegistrationResult",
    "TransformationEstimation",
    "evaluate_registration",
    "icp",
    "__version__",
]
