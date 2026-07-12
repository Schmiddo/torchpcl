import pytest
import torch

import torchpcl as tp
from torchpcl.transforms import pose_to_matrix, transform_points

from conftest import random_cloud, random_rigid_transform


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


def _scales():
    return [
        tp.ICPScale(voxel_size=0.25, max_distance=0.6, iterations=40),
        tp.ICPScale(voxel_size=0.1, max_distance=0.25, iterations=30),
        tp.ICPScale(voxel_size=0.04, max_distance=0.1, iterations=20),
    ]


def test_multiscale_recovers_outside_fine_scale_basin(search_device):
    target = _asymmetric_cloud(search_device)
    expected = pose_to_matrix(
        torch.tensor([0.08, -0.05, 0.12, 0.28, -0.18, 0.12])
    ).to(search_device)
    source = transform_points(target, torch.linalg.inv(expected))

    fine = tp.icp(source, target, max_distance=0.08, max_iterations=40)
    result = tp.multiscale_icp(source, target, _scales())

    assert not fine.converged[0]
    assert result.converged[0]
    assert 0 < result.iterations[0] <= sum(scale.iterations for scale in _scales())
    torch.testing.assert_close(
        result.transforms[0], expected, atol=5e-4, rtol=5e-4
    )


def test_multiscale_supports_packed_batches(search_device):
    first = random_cloud(400, search_device, seed=40)
    second = random_cloud(350, search_device, seed=41) + 2.0
    target = tp.PointCloud(
        torch.cat([first, second]),
        torch.tensor([0, len(first), len(first) + len(second)], device=search_device),
    )
    expected = torch.stack(
        [
            random_rigid_transform(0.08, 0.08, seed=42, device=search_device),
            random_rigid_transform(0.08, 0.08, seed=43, device=search_device),
        ]
    )
    source = tp.transform(target, torch.linalg.inv(expected))
    scales = [
        tp.ICPScale(0.2, 0.4, 30),
        tp.ICPScale(0.05, 0.15, 20),
    ]

    result = tp.multiscale_icp(source, target, scales)

    assert result.converged.tolist() == [True, True]
    torch.testing.assert_close(result.transforms, expected, atol=2e-3, rtol=2e-3)


def test_precomputed_pyramids_match_internal_preprocessing(search_device):
    target = random_cloud(500, search_device, seed=50)
    expected = random_rigid_transform(
        0.05, 0.03, seed=51, device=search_device
    )
    source = transform_points(target, torch.linalg.inv(expected))
    scales = [tp.ICPScale(0.15, 0.3, 20), tp.ICPScale(0.05, 0.1, 15)]
    sizes = [scale.voxel_size for scale in scales]
    source_pyramid = tp.build_pyramid(source, sizes)
    target_pyramid = tp.build_pyramid(target, sizes)

    direct = tp.multiscale_icp(source, target, scales)
    precomputed = tp.multiscale_icp(source_pyramid, target_pyramid, scales)

    torch.testing.assert_close(precomputed.transforms, direct.transforms)
    torch.testing.assert_close(precomputed.fitness, direct.fitness)
    assert torch.equal(precomputed.iterations, direct.iterations)


def test_point_to_plane_estimates_level_normals(search_device):
    target = _asymmetric_cloud(search_device)[:500]
    scales = [tp.ICPScale(0.15, 0.3, 10), tp.ICPScale(0.05, 0.1, 10)]

    result = tp.multiscale_icp(
        target,
        target,
        scales,
        method="point_to_plane",
        normal_k=12,
        normal_radius_factor=None,
    )

    assert result.converged[0]
    assert torch.isfinite(result.transforms).all()


def test_build_pyramid_reduces_attached_normals(search_device):
    points = random_cloud(100, search_device, seed=60)
    normals = torch.zeros_like(points)
    normals[:, 2] = 1
    cloud = tp.PointCloud.from_points(points, normals=normals)

    pyramid = tp.build_pyramid(cloud, [0.2, 0.1], normal_mode="reduce")

    assert len(pyramid.levels) == 2
    for level in pyramid.levels:
        assert level.normals is not None
        torch.testing.assert_close(
            level.normals[:, 2], torch.ones_like(level.normals[:, 2])
        )


def test_multiscale_validation():
    points = torch.randn(20, 3)
    with pytest.raises(ValueError, match="at least one"):
        tp.multiscale_icp(points, points, [])
    with pytest.raises(ValueError, match="voxel_sizes"):
        tp.multiscale_icp(
            tp.build_pyramid(points, [0.1]),
            tp.build_pyramid(points, [0.1]),
            [tp.ICPScale(0.2, 0.4, 5)],
        )
