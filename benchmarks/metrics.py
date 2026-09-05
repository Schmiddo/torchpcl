"""Ground-truth loading and pose accuracy metrics."""

import numpy as np


def pose_errors(estimate, expected, source):
    """Pose errors and displacement of the same source points under both poses."""
    # Project onto SO(3): text ground truth may have rounded rotation entries.
    def rotation(matrix):
        u, _, vt = np.linalg.svd(matrix[:3, :3])
        return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt

    relative = rotation(estimate) @ rotation(expected).T
    angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    displacement = source @ (estimate[:3, :3] - expected[:3, :3]).T
    displacement += estimate[:3, 3] - expected[:3, 3]
    return {
        "rotation_error_deg": float(angle),
        "translation_error": float(np.linalg.norm(estimate[:3, 3] - expected[:3, 3])),
        "source_displacement_rmse": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
    }


def load_transform(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: expected a finite 4x4 source-to-target matrix")
    r = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(r.T @ r, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(r), 1, atol=1e-4)
    ):
        raise ValueError(f"{path}: expected a rigid transform")
    return matrix


