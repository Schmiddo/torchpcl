import pytest
import torch

import torchpcl as tp
from torchpcl.registration import _materialize_level
from torchpcl.transforms import pose_to_matrix, transform_points

from conftest import random_cloud, random_rigid_transform


def _level(
    distance: float = 0.1,
    iterations: int = 30,
    voxel_size: float | None = None,
) -> tp.ICPLevel:
    return tp.ICPLevel(
        max_correspondence_distance=distance,
        max_iterations=iterations,
        voxel_size=voxel_size,
    )


def _random_normals(n, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    normals = torch.randn((n, 3), generator=generator, dtype=torch.float64)
    return (normals / normals.norm(dim=1, keepdim=True)).to(device)


def _asymmetric_cloud(device):
    generator = torch.Generator(device="cpu").manual_seed(123)
    parts = [
        torch.randn(700, 3, generator=generator)
        * torch.tensor([0.5, 0.15, 0.08]),
        torch.randn(400, 3, generator=generator)
        * torch.tensor([0.15, 0.35, 0.1])
        + torch.tensor([1.0, 0.5, 0.2]),
        torch.randn(300, 3, generator=generator)
        * torch.tensor([0.1, 0.1, 0.3])
        + torch.tensor([-0.5, 0.8, -0.1]),
    ]
    return torch.cat(parts).to(device=device, dtype=torch.float64)


def _multilevel_schedule():
    return [
        _level(distance=0.6, iterations=40, voxel_size=0.25),
        _level(distance=0.25, iterations=30, voxel_size=0.1),
        _level(distance=0.1, iterations=20, voxel_size=0.04),
    ]


def test_unvoxelized_level_uses_original_clouds(search_device):
    points = random_cloud(50, search_device, seed=0)
    cloud = tp.as_point_cloud(points)

    prepared = _materialize_level(
        cloud,
        cloud,
        _level(voxel_size=None),
        tp.PointToPoint(),
    )

    assert prepared.source is cloud
    assert prepared.target is cloud


def test_single_level_point_to_point_recovers_transform(search_device):
    target = random_cloud(1000, search_device, seed=0)
    expected = random_rigid_transform(
        max_angle=0.05,
        max_translation=0.02,
        seed=1,
        device=search_device,
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, [_level()])

    assert result.converged.tolist() == [True]
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
    torch.testing.assert_close(
        result.inlier_rmse,
        torch.zeros_like(result.inlier_rmse),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)


def test_single_level_point_to_plane_recovers_transform(search_device):
    target_points = random_cloud(2000, search_device, seed=0)
    target = tp.PointCloud.from_points(
        target_points,
        normals=_random_normals(2000, search_device, seed=2),
    )
    expected = random_rigid_transform(
        max_angle=0.05,
        max_translation=0.02,
        seed=1,
        device=search_device,
    )
    source = transform_points(target_points, torch.linalg.inv(expected))

    result = tp.icp(
        source,
        target,
        [_level(distance=0.2)],
        options=tp.ICPOptions(objective=tp.PointToPlane()),
    )

    assert result.converged.tolist() == [True]
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-4, rtol=1e-4)


def test_multilevel_recovers_outside_fine_level_basin(search_device):
    target = _asymmetric_cloud(search_device)
    expected = pose_to_matrix(
        torch.tensor([0.08, -0.05, 0.12, 0.28, -0.18, 0.12])
    ).to(search_device)
    source = transform_points(target, torch.linalg.inv(expected))

    fine = tp.icp(source, target, [_level(distance=0.08, iterations=40)])
    result = tp.icp(source, target, _multilevel_schedule())

    assert not fine.converged[0]
    assert result.converged[0]
    assert 0 < result.iterations[0] <= sum(
        level.max_iterations for level in _multilevel_schedule()
    )
    torch.testing.assert_close(
        result.transforms[0],
        expected,
        atol=5e-4,
        rtol=5e-4,
    )


def test_multilevel_can_finish_at_full_resolution(search_device):
    target = random_cloud(500, search_device, seed=3)
    expected = random_rigid_transform(
        0.05,
        0.03,
        seed=4,
        device=search_device,
    )
    source = transform_points(target, torch.linalg.inv(expected))
    levels = [
        _level(distance=0.25, iterations=20, voxel_size=0.15),
        _level(distance=0.1, iterations=20, voxel_size=None),
    ]

    result = tp.icp(source, target, levels)

    assert result.level_results[-1].level.voxel_size is None
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)


def test_voxelized_point_to_plane_reduces_and_normalizes_normals(search_device):
    points = random_cloud(100, search_device, seed=5)
    normals = torch.zeros_like(points)
    normals[:, 2] = 1
    cloud = tp.PointCloud.from_points(points, normals=normals)

    prepared = _materialize_level(
        cloud,
        cloud,
        _level(voxel_size=0.2),
        tp.PointToPlane(),
    )

    assert prepared.target_normals is not None
    torch.testing.assert_close(
        prepared.target_normals.norm(dim=1),
        torch.ones(
            prepared.target_normals.shape[0],
            dtype=points.dtype,
            device=search_device,
        ),
    )
    torch.testing.assert_close(
        prepared.target_normals[:, 2],
        torch.ones_like(prepared.target_normals[:, 2]),
    )


def test_huber_loss_produces_finite_result(search_device):
    target = random_cloud(200, search_device, seed=6)
    source = target.clone()
    source[-1] += 0.05

    result = tp.icp(
        source,
        target,
        [_level()],
        options=tp.ICPOptions(robust_loss=tp.HuberLoss(delta=0.01)),
    )

    assert torch.isfinite(result.transforms).all()
    assert torch.isfinite(result.inlier_rmse).all()


def test_initial_transform_reaches_first_level(search_device):
    target = random_cloud(500, search_device, seed=7)
    expected = random_rigid_transform(
        0.05,
        0.02,
        seed=8,
        device=search_device,
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, [_level()], init=expected)

    assert result.converged.tolist() == [True]
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)


def test_transforms_propagate_between_levels(search_device):
    target = random_cloud(500, search_device, seed=9)
    expected = random_rigid_transform(
        0.05,
        0.02,
        seed=10,
        device=search_device,
    )
    source = transform_points(target, torch.linalg.inv(expected))
    levels = [
        _level(distance=0.2, iterations=20, voxel_size=0.1),
        _level(distance=0.1, iterations=20),
    ]

    result = tp.icp(source, target, levels)

    assert len(result.level_results) == 2
    torch.testing.assert_close(
        result.level_results[0].transforms[0],
        expected,
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)


def test_packed_batches_converge_independently(search_device):
    first = random_cloud(300, search_device, seed=11)
    second = random_cloud(250, search_device, seed=12) + 2.0
    target = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor(
            [0, len(first), len(first) + len(second)],
            device=search_device,
        ),
    )
    expected = torch.stack(
        [
            random_rigid_transform(0.03, 0.01, seed=13, device=search_device),
            random_rigid_transform(0.03, 0.01, seed=14, device=search_device),
        ]
    )
    source = tp.transform(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, [_level()])

    assert result.converged.tolist() == [True, True]
    torch.testing.assert_close(result.transforms, expected, atol=1e-5, rtol=1e-5)


def test_batch_failure_does_not_stop_other_entries(search_device):
    first = random_cloud(300, search_device, seed=15)
    second = random_cloud(100, search_device, seed=16) + 2.0
    target = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor(
            [0, len(first), len(first) + len(second)],
            device=search_device,
        ),
    )
    expected = random_rigid_transform(
        0.03,
        0.01,
        seed=17,
        device=search_device,
    )
    source = tp.PointCloud(
        torch.cat(
            [
                transform_points(first, torch.linalg.inv(expected)),
                second + 100.0,
            ]
        ),
        target.offsets,
    )

    result = tp.icp(source, target, [_level()])

    assert result.converged.tolist() == [True, False]
    assert result.fitness[1] == 0
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result.transforms[1],
        torch.eye(4, dtype=source.dtype, device=search_device),
    )


def test_float32_inputs_produce_float32_results(search_device):
    target = random_cloud(500, search_device, seed=18).to(torch.float32)
    expected = random_rigid_transform(
        0.05,
        0.02,
        seed=19,
        device=search_device,
    ).to(torch.float32)
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.icp(source, target, [_level()])

    assert result.transforms.dtype == torch.float32
    assert result.fitness.dtype == torch.float32
    assert result.inlier_rmse.dtype == torch.float32
    torch.testing.assert_close(result.transforms[0], expected, atol=1e-3, rtol=1e-3)


def test_result_contains_per_level_diagnostics(search_device):
    points = random_cloud(200, search_device, seed=20)
    levels = [
        _level(distance=0.2, iterations=5, voxel_size=0.1),
        _level(distance=0.1, iterations=5),
    ]

    result = tp.icp(points, points, levels)

    assert tuple(level_result.level for level_result in result.level_results) == (
        *levels,
    )
    assert torch.equal(
        result.iterations,
        sum(
            (level.iterations for level in result.level_results),
            torch.zeros_like(result.iterations),
        ),
    )
    final = result.level_results[-1]
    assert result.transforms is final.transforms
    assert result.converged is final.converged
    assert result.fitness is final.fitness
    assert result.inlier_rmse is final.inlier_rmse
    assert not hasattr(result, "correspondences")


def test_icp_configuration_validation(search_device):
    points = random_cloud(10, search_device, seed=21)

    with pytest.raises(ValueError, match="at least one"):
        tp.icp(points, points, [])
    with pytest.raises(TypeError, match="ICPLevel"):
        tp.icp(points, points, [object()])
    with pytest.raises(ValueError, match="target normals"):
        tp.icp(
            points,
            points,
            [_level()],
            options=tp.ICPOptions(objective=tp.PointToPlane()),
        )
    with pytest.raises(ValueError, match="non-empty"):
        tp.icp(points[:0], points, [_level()])
    with pytest.raises(ValueError, match="max_correspondence_distance"):
        tp.ICPLevel(max_correspondence_distance=0)
    with pytest.raises(ValueError, match="max_iterations"):
        tp.ICPLevel(max_correspondence_distance=0.1, max_iterations=-1)
    with pytest.raises(ValueError, match="voxel_size"):
        tp.ICPLevel(max_correspondence_distance=0.1, voxel_size=0)
    with pytest.raises(ValueError, match="fitness_tolerance"):
        tp.ConvergenceCriteria(fitness_tolerance=-1)
    with pytest.raises(ValueError, match="rmse_tolerance"):
        tp.ConvergenceCriteria(rmse_tolerance=-1)
    with pytest.raises(ValueError, match="delta"):
        tp.HuberLoss(delta=0)


def test_legacy_registration_api_is_not_exported():
    for name in (
        "ICPScale",
        "PointCloudPyramid",
        "build_pyramid",
        "multiscale_icp",
    ):
        assert not hasattr(tp, name)


def test_evaluate_registration_batched_result(search_device):
    target = random_cloud(500, search_device, seed=22)
    expected = random_rigid_transform(
        0.05,
        0.02,
        seed=23,
        device=search_device,
    )
    source = transform_points(target, torch.linalg.inv(expected))

    result = tp.evaluate_registration(source, target, 0.05, expected)

    assert isinstance(result, tp.RegistrationMetrics)
    assert result.transforms.shape == (1, 4, 4)
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
    torch.testing.assert_close(
        result.inlier_rmse,
        torch.zeros_like(result.inlier_rmse),
        atol=1e-6,
        rtol=0,
    )


def test_evaluate_registration_packed_batch(search_device):
    first = random_cloud(30, search_device, seed=24)
    second = random_cloud(20, search_device, seed=25) + 2
    cloud = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor(
            [0, len(first), len(first) + len(second)],
            device=search_device,
        ),
    )

    result = tp.evaluate_registration(
        cloud,
        cloud,
        max_correspondence_distance=0.01,
    )

    assert result.transforms.shape == (2, 4, 4)
    torch.testing.assert_close(result.fitness, torch.ones_like(result.fitness))
