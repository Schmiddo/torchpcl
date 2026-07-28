import pytest
import torch

import torchpcl as tp
from torchpcl.normals import _symmetric_eigh_3x3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_symmetric_eigh_3x3_matches_torch(dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(0)
    factors = torch.randn(512, 3, 3, generator=generator, dtype=dtype)
    matrices = factors @ factors.mT
    expected_values, expected_vectors = torch.linalg.eigh(matrices)
    matrices = matrices.to(device)

    values, smallest_vectors = _symmetric_eigh_3x3(matrices)

    tolerance = 5e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        values,
        expected_values.to(device),
        atol=tolerance,
        rtol=tolerance,
    )
    residual = matrices @ smallest_vectors[:, :, None]
    residual = residual - values[:, :1, None] * smallest_vectors[:, :, None]
    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        atol=tolerance,
        rtol=0,
    )
    alignment = (
        smallest_vectors * expected_vectors[:, :, 0].to(device)
    ).sum(dim=-1)
    assert (alignment.abs() > 0.999).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_symmetric_eigh_3x3_handles_repeated_eigenvalues(dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(1)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=dtype)
    )
    matrices = torch.stack(
        (
            torch.zeros(3, 3, dtype=dtype),
            torch.eye(3, dtype=dtype),
            rotation @ torch.diag(torch.tensor([1, 1, 2], dtype=dtype)) @ rotation.mT,
            rotation @ torch.diag(torch.tensor([0, 1, 1], dtype=dtype)) @ rotation.mT,
        )
    ).to(device)
    expected_values = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [1, 1, 2], [0, 1, 1]],
        dtype=dtype,
        device=device,
    )

    values, smallest_vectors = _symmetric_eigh_3x3(matrices)

    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        values,
        expected_values,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        smallest_vectors.norm(dim=-1),
        torch.ones(4, dtype=dtype, device=device),
        atol=tolerance,
        rtol=0,
    )
    residual = matrices @ smallest_vectors[:, :, None]
    residual = residual - values[:, :1, None] * smallest_vectors[:, :, None]
    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        atol=tolerance,
        rtol=0,
    )


def test_packed_normals_and_curvature(search_device):
    xy = torch.rand(80, 2, dtype=torch.float64, device=search_device)
    first = torch.cat([xy[:40], torch.zeros(40, 1, device=search_device)], dim=1)
    second = torch.cat([xy[40:], torch.ones(40, 1, device=search_device)], dim=1)
    cloud = tp.PointCloud(
        torch.cat([first, second]), torch.tensor([0, 40, 80], device=search_device)
    )

    result = tp.estimate_normals(cloud, k=10)

    assert result.valid.all()
    assert (result.normals[:, 2].abs() > 0.99).all()
    assert result.curvature.shape == (80,)
    assert (result.curvature < 1e-6).all()


def test_existing_normals_fix_the_sign(search_device):
    xy = torch.rand(50, 2, dtype=torch.float64, device=search_device)
    points = torch.cat([xy, torch.zeros(50, 1, device=search_device)], dim=1)
    signs = torch.where(
        torch.rand(50, device=search_device) < 0.5, 1.0, -1.0
    ).to(torch.float64)
    old = torch.zeros_like(points)
    old[:, 2] = signs
    cloud = tp.PointCloud.from_points(points, normals=old)

    result = tp.estimate_normals(cloud, k=10)

    assert (result.normals[:, 2] * signs > 0.99).all()


def test_normal_estimation_can_reuse_index(search_device):
    xy = torch.rand(30, 2, dtype=torch.float64, device=search_device)
    cloud = tp.PointCloud.from_points(
        torch.cat([xy, torch.zeros(30, 1, device=search_device)], dim=1)
    )
    index = tp.NeighborIndex(cloud)

    reused = tp.estimate_normals(cloud, k=8, index=index)
    direct = tp.estimate_normals(cloud, k=8)

    torch.testing.assert_close(reused.normals.abs(), direct.normals.abs())
