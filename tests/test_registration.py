import pytest
import torch

from torchpcl import (
    ICPConvergenceCriteria,
    PointToPlane,
    PointToPoint,
    evaluate_registration,
    icp,
)
from torchpcl.transforms import transform_points

from conftest import random_cloud, random_rigid_transform


def test_icp_point_to_point_recovers_transform(device):
    target = random_cloud(1000, device, seed=0)
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    result = icp(source, target, max_correspondence_distance=0.1)
    assert result.converged
    assert result.fitness == pytest.approx(1.0)
    assert result.inlier_rmse == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(result.transformation, gt, atol=1e-5)


def _random_normals(n, device, seed):
    # Independent random normals: surface-consistent normals on simple
    # synthetic shapes (e.g. a sphere) leave rotation unobservable for
    # point-to-plane (cross(s, n) ≈ 0).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    normals = torch.randn((n, 3), generator=gen, dtype=torch.float64)
    return (normals / normals.norm(dim=1, keepdim=True)).to(device)


def test_icp_point_to_plane_recovers_transform(device):
    target = random_cloud(2000, device, seed=0)
    normals = _random_normals(2000, device, seed=2)
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    result = icp(
        source,
        target,
        max_correspondence_distance=0.2,
        estimation=PointToPlane(),
        target_normals=normals,
    )
    assert result.converged
    assert result.fitness == pytest.approx(1.0)
    assert torch.allclose(result.transformation, gt, atol=1e-4)


def test_icp_disjoint_clouds(device):
    source = random_cloud(100, device, seed=0)
    target = random_cloud(100, device, seed=1) + 100.0

    result = icp(source, target, max_correspondence_distance=0.1)
    assert not result.converged
    assert result.fitness == 0.0
    assert result.num_iterations == 0
    assert torch.allclose(
        result.transformation, torch.eye(4, dtype=torch.float64, device=device)
    )
    assert (result.correspondences == -1).all()


def test_icp_max_iteration_respected(device):
    target = random_cloud(500, device, seed=0)
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    result = icp(
        source,
        target,
        max_correspondence_distance=0.1,
        criteria=ICPConvergenceCriteria(max_iteration=1),
    )
    assert result.num_iterations == 1
    assert not result.converged


def test_icp_with_init(device):
    target = random_cloud(500, device, seed=0)
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    result = icp(source, target, max_correspondence_distance=0.1, init=gt)
    assert result.converged
    assert torch.allclose(result.transformation, gt, atol=1e-5)


def test_icp_validates_normals_requirement(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="requires target_normals"):
        icp(points, points, 0.1, estimation=PointToPlane())


def test_icp_validates_distance(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="positive"):
        icp(points, points, 0.0)


def test_evaluate_registration_ground_truth(device):
    target = random_cloud(500, device, seed=0)
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    result = evaluate_registration(source, target, 0.05, gt)
    assert result.fitness == pytest.approx(1.0)
    assert result.inlier_rmse == pytest.approx(0.0, abs=1e-6)
    assert (result.correspondences == torch.arange(500, device=device)).all()


def test_evaluate_registration_identity_default(device):
    points = random_cloud(100, device, seed=0)
    result = evaluate_registration(points, points, 0.01)
    assert result.fitness == pytest.approx(1.0)


def test_icp_float32_input(device):
    target = random_cloud(500, device, seed=0).to(torch.float32)
    gt = random_rigid_transform(max_angle=0.05, max_translation=0.02, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt).to(torch.float32))

    result = icp(source, target, max_correspondence_distance=0.1)
    assert result.converged
    assert result.transformation.dtype == torch.float64
    assert torch.allclose(result.transformation, gt, atol=1e-3)
