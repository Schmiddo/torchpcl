import pytest
import torch

import torchpcl as tp
from torchpcl.transforms import transform_points

from conftest import random_cloud, random_rigid_transform


def test_procrustes_recovers_rigid_transform(device):
    source = random_cloud(40, device, seed=100)
    expected = random_rigid_transform(0.8, 0.5, seed=101, device=device)
    target = transform_points(source, expected)

    result = tp.procrustes(source, target)

    assert result.rotation.shape == (1, 3, 3)
    assert result.translation.shape == (1, 3)
    assert result.scale.shape == (1,)
    torch.testing.assert_close(result.scale, torch.ones_like(result.scale))
    torch.testing.assert_close(
        result.transforms[0], expected, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        torch.linalg.det(result.rotation), torch.ones_like(result.scale)
    )


def test_procrustes_recovers_similarity_transform(device):
    source = random_cloud(50, device, seed=110)
    rigid = random_rigid_transform(0.8, 0.5, seed=111, device=device)
    expected_scale = source.new_tensor(2.5)
    target = expected_scale * (source @ rigid[:3, :3].T) + rigid[:3, 3]

    result = tp.procrustes(source, target, estimate_scale=True)

    torch.testing.assert_close(
        result.rotation[0], rigid[:3, :3], atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        result.translation[0], rigid[:3, 3], atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(result.scale[0], expected_scale)
    torch.testing.assert_close(
        result.transforms[0, :3, :3], expected_scale * rigid[:3, :3]
    )


def test_procrustes_zero_weight_ignores_outlier(device):
    source = random_cloud(20, device, seed=120)
    expected = random_rigid_transform(0.5, 0.3, seed=121, device=device)
    target = transform_points(source, expected)
    target[-1] = target[-1] + 100
    weights = torch.ones(20, dtype=source.dtype, device=device)
    weights[-1] = 0

    result = tp.procrustes(source, target, weights=weights)

    torch.testing.assert_close(
        result.transforms[0], expected, atol=1e-10, rtol=1e-10
    )


def test_procrustes_packed_ragged_batch(device):
    first = random_cloud(30, device, seed=130)
    second = random_cloud(20, device, seed=131) + 2
    offsets = torch.tensor([0, 30, 50], device=device)
    source = tp.PointCloud(torch.cat([first, second]), offsets)
    expected = torch.stack(
        [
            random_rigid_transform(0.5, 0.3, seed=132, device=device),
            random_rigid_transform(0.5, 0.3, seed=133, device=device),
        ]
    )
    target = tp.transform(source, expected)

    result = tp.procrustes(source, target)

    torch.testing.assert_close(
        result.transforms, expected, atol=1e-10, rtol=1e-10
    )


def test_procrustes_gradcheck():
    generator = torch.Generator().manual_seed(140)
    source = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    weights = torch.rand(9, generator=generator, dtype=torch.float64) + 0.5
    source.requires_grad_()
    target.requires_grad_()
    weights.requires_grad_()

    def align(s, t, w):
        return tp.procrustes(s, t, weights=w, estimate_scale=True).transforms

    assert torch.autograd.gradcheck(align, (source, target, weights))


def test_procrustes_validates_correspondences():
    source = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="equal lengths"):
        tp.procrustes(source, target)

    with pytest.raises(ValueError, match="nonnegative"):
        tp.procrustes(
            source,
            source,
            weights=torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float64),
        )

    line = torch.arange(4, dtype=torch.float64)[:, None] * torch.tensor(
        [[1.0, 2.0, 3.0]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="non-collinear"):
        tp.procrustes(line, line)

    weights = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 3"):
        tp.procrustes(source, source, weights=weights)
