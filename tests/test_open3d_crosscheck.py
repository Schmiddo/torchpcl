"""Cross-check against Open3D's legacy registration pipeline.

Skipped unless open3d is importable (it has no Python 3.14 wheels at the
time of writing; install best-effort with `uv pip install open3d`).

Correspondence index arrays are never compared directly — nearest-neighbor
tie-breaking differs between the cuBQL BVH and Open3D's KDTree.
"""

import numpy as np
import pytest
import torch

o3d = pytest.importorskip("open3d")

from torchpcl import ICPConvergenceCriteria, PointToPlane, PointToPoint, icp
from torchpcl.transforms import transform_points

from conftest import random_cloud, random_rigid_transform


def _to_o3d(points: torch.Tensor, normals: torch.Tensor | None = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
    return pcd


@pytest.mark.parametrize("estimation_name", ["point_to_point", "point_to_plane"])
def test_matches_open3d(estimation_name, search_device):
    device = search_device
    target = random_cloud(1000, device, seed=0)
    if estimation_name == "point_to_plane":
        gen = torch.Generator().manual_seed(2)
        normals = torch.randn((1000, 3), generator=gen, dtype=torch.float64)
        normals = normals / normals.norm(dim=1, keepdim=True)
    else:
        normals = None
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))
    max_dist = 0.1
    criteria = ICPConvergenceCriteria(max_iteration=30)

    if estimation_name == "point_to_point":
        ours = icp(source, target, max_dist, estimation=PointToPoint(), criteria=criteria)
        o3d_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        ours = icp(
            source, target, max_dist,
            estimation=PointToPlane(), target_normals=normals, criteria=criteria,
        )
        o3d_est = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    theirs = o3d.pipelines.registration.registration_icp(
        _to_o3d(source),
        _to_o3d(target, normals),
        max_dist,
        np.eye(4),
        o3d_est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )

    assert ours.fitness == pytest.approx(theirs.fitness, abs=1e-5)
    assert ours.inlier_rmse == pytest.approx(theirs.inlier_rmse, abs=1e-5)
    np.testing.assert_allclose(
        ours.transformation.cpu().numpy(), theirs.transformation, atol=1e-5
    )
