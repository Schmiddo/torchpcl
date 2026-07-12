import pytest
import torch

import torchpcl as tp
from torchpcl.transforms import transform_points

from conftest import random_cloud, random_rigid_transform


def _random_normals(n, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    normals = torch.randn((n, 3), generator=generator, dtype=torch.float64)
    return (normals / normals.norm(dim=1, keepdim=True)).to(device)


def test_icp_point_to_point_recovers_transform(search_device):
    target = random_cloud(1000, search_device, seed=0)
    expected = random_rigid_transform(
        max_angle=0.05, max_translation=0.02, seed=1, device=search_device
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, max_distance=0.1)

    assert result.converged.tolist() == [True]
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
    torch.testing.assert_close(
        result.inlier_rmse, torch.zeros_like(result.inlier_rmse), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)


def test_icp_point_to_plane_recovers_transform(search_device):
    target = random_cloud(2000, search_device, seed=0)
    normals = _random_normals(2000, search_device, seed=2)
    expected = random_rigid_transform(
        max_angle=0.05, max_translation=0.02, seed=1, device=search_device
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(
        source,
        target,
        max_distance=0.2,
        method="point_to_plane",
        target_normals=normals,
    )

    assert result.converged.tolist() == [True]
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-4, rtol=1e-4)


def test_icp_disjoint_cloud_keeps_identity(search_device):
    source = random_cloud(100, search_device, seed=0)
    target = random_cloud(100, search_device, seed=1) + 100.0

    result = tp.icp(source, target, max_distance=0.1)

    assert result.converged.tolist() == [False]
    assert result.iterations.tolist() == [0]
    assert result.fitness.tolist() == [0.0]
    torch.testing.assert_close(
        result.transforms[0],
        torch.eye(4, dtype=source.dtype, device=search_device),
    )


def test_icp_respects_max_iterations(search_device):
    target = random_cloud(500, search_device, seed=0)
    expected = random_rigid_transform(
        max_angle=0.05, max_translation=0.02, seed=1, device=search_device
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, max_distance=0.1, max_iterations=1)

    assert result.iterations.tolist() == [1]
    assert result.converged.tolist() == [False]


def test_icp_accepts_initial_transform(search_device):
    target = random_cloud(500, search_device, seed=0)
    expected = random_rigid_transform(
        max_angle=0.05, max_translation=0.02, seed=1, device=search_device
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, max_distance=0.1, init=expected)

    assert result.converged.tolist() == [True]
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)


def test_icp_uses_attached_target_normals(search_device):
    target_points = random_cloud(500, search_device, seed=0)
    normals = _random_normals(500, search_device, seed=2)
    target = tp.PointCloud.from_points(target_points, normals=normals)

    result = tp.icp(
        target_points,
        target,
        max_distance=0.1,
        method="point_to_plane",
    )

    assert result.converged.tolist() == [True]


def test_icp_validates_inputs(search_device):
    points = random_cloud(10, search_device, seed=0)
    with pytest.raises(ValueError, match="target_normals"):
        tp.icp(points, points, 0.1, method="point_to_plane")
    with pytest.raises(ValueError, match="max_distance"):
        tp.icp(points, points, 0.0)
    with pytest.raises(ValueError, match="method"):
        tp.icp(points, points, 0.1, method="unknown")
    with pytest.raises(ValueError, match="non-empty"):
        tp.icp(points[:0], points, 0.1)


def test_evaluate_registration_batched_result(search_device):
    target = random_cloud(500, search_device, seed=0)
    expected = random_rigid_transform(
        max_angle=0.05, max_translation=0.02, seed=1, device=search_device
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.evaluate_registration(source, target, 0.05, expected)

    assert result.transforms.shape == (1, 4, 4)
    assert isinstance(result, tp.RegistrationMetrics)
    assert not hasattr(result, "iterations")
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
    torch.testing.assert_close(
        result.inlier_rmse, torch.zeros_like(result.inlier_rmse), atol=1e-6, rtol=0
    )


def test_evaluate_registration_packed_batch(search_device):
    first = random_cloud(30, search_device, seed=70)
    second = random_cloud(20, search_device, seed=71) + 2
    cloud = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor([0, len(first), len(first) + len(second)], device=search_device),
    )

    result = tp.evaluate_registration(cloud, cloud, max_distance=0.01)

    assert result.transforms.shape == (2, 4, 4)
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))


def test_icp_preserves_float32(search_device):
    target = random_cloud(500, search_device, seed=0).to(torch.float32)
    expected = random_rigid_transform(
        max_angle=0.05, max_translation=0.02, seed=1, device=search_device
    ).to(torch.float32)
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, max_distance=0.1)

    assert result.transforms.dtype == torch.float32
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-3, rtol=1e-3)


def test_icp_packed_batches_converge_independently(search_device):
    first = random_cloud(300, search_device, seed=10)
    second = random_cloud(250, search_device, seed=11) + 2.0
    target = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor([0, len(first), len(first) + len(second)], device=search_device),
    )
    expected = torch.stack(
        [
            random_rigid_transform(0.03, 0.01, seed=12, device=search_device),
            random_rigid_transform(0.03, 0.01, seed=13, device=search_device),
        ]
    )
    source = tp.transform(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, max_distance=0.1)

    assert result.converged.tolist() == [True, True]
    torch.testing.assert_close(result.transforms, expected, atol=1e-5, rtol=1e-5)


def test_icp_batch_failure_does_not_stop_other_entries(search_device):
    first = random_cloud(300, search_device, seed=20)
    second = random_cloud(100, search_device, seed=21) + 2.0
    target = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor([0, len(first), len(first) + len(second)], device=search_device),
    )
    expected = random_rigid_transform(
        0.03, 0.01, seed=22, device=search_device
    )
    source_points = torch.cat(
        [transform_points(first, torch.linalg.inv(expected)), second + 100.0]
    )
    source = tp.PointCloud(source_points, target.offsets)

    result = tp.icp(source, target, max_distance=0.1)

    assert result.converged.tolist() == [True, False]
    assert result.fitness[1] == 0
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result.transforms[1],
        torch.eye(4, dtype=source.dtype, device=search_device),
    )


def test_icp_supports_huber_weights(search_device):
    target = random_cloud(200, search_device, seed=30)
    source = target.clone()
    source[-1] += 0.05

    result = tp.icp(
        source,
        target,
        max_distance=0.1,
        robust_kernel="huber",
        robust_delta=0.01,
    )

    assert result.transforms.shape == (1, 4, 4)
    assert torch.isfinite(result.transforms).all()


def test_icp_result_does_not_contain_correspondences(search_device):
    points = random_cloud(20, search_device, seed=40)
    result = tp.icp(points, points, max_distance=0.1)

    assert not hasattr(result, "correspondences")
