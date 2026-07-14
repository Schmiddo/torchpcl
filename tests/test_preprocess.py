import pytest
import torch

from torchpcl import estimate_normals, voxelize

from conftest import random_cloud


def test_normals_unbounded_default(search_device):
    # Without a radius the neighborhood is true (unbounded) k-NN.
    gen = torch.Generator().manual_seed(0)
    xy = torch.rand((2000, 2), generator=gen, dtype=torch.float64)
    points = torch.cat([xy, torch.zeros(2000, 1, dtype=torch.float64)], dim=1).to(search_device)

    normals = estimate_normals(points, k=20).normals
    assert torch.allclose(
        normals[:, 2].abs(),
        torch.ones(2000, dtype=torch.float64, device=search_device),
        atol=1e-6,
    )


def test_normals_on_plane(search_device):
    gen = torch.Generator().manual_seed(0)
    xy = torch.rand((2000, 2), generator=gen, dtype=torch.float64)
    points = torch.cat([xy, torch.zeros(2000, 1, dtype=torch.float64)], dim=1).to(search_device)

    normals = estimate_normals(points, radius=0.1, k=20).normals
    assert normals.shape == points.shape
    assert torch.allclose(normals[:, 2].abs(), torch.ones(2000, dtype=torch.float64, device=search_device), atol=1e-6)
    assert torch.allclose(normals.norm(dim=1), torch.ones(2000, dtype=torch.float64, device=search_device), atol=1e-6)


def test_normals_on_sphere(search_device):
    gen = torch.Generator().manual_seed(1)
    points = torch.randn((5000, 3), generator=gen, dtype=torch.float64)
    points = (points / points.norm(dim=1, keepdim=True)).to(search_device)

    normals = estimate_normals(points, radius=0.2, k=30).normals
    # Normals of a unit sphere are the (possibly sign-flipped) positions.
    alignment = (normals * points).sum(dim=1).abs()
    assert float(alignment.min()) > 0.98


def test_normals_isolated_points_zero(search_device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float64, device=search_device
    )
    result = estimate_normals(points, radius=0.1, k=5)
    assert (result.normals == 0).all()
    assert not result.valid.any()


def test_normals_input_validation(search_device):
    points = random_cloud(10, search_device, seed=0)
    with pytest.raises(ValueError, match="radius"):
        estimate_normals(points, radius=0.0)
    with pytest.raises(ValueError, match="k must be"):
        estimate_normals(points, radius=0.1, k=2)


def test_voxelize_means(device):
    # Two clusters in distinct voxels: result must be the cluster means.
    points = torch.tensor(
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [5.10, 5.10, 5.10],
            [5.30, 5.10, 5.10],
        ],
        dtype=torch.float64,
        device=device,
    )
    down = voxelize(points, voxel_size=1.0).cloud.points
    assert down.shape == (2, 3)
    expected = torch.stack([points[:2].mean(dim=0), points[2:].mean(dim=0)])
    # Output voxel order is arbitrary; sort both by x.
    down = down[down[:, 0].argsort()]
    expected = expected[expected[:, 0].argsort()]
    assert torch.allclose(down, expected)


def test_voxelize_negative_coords(device):
    points = torch.tensor(
        [[-1.4, -1.4, -1.4], [-1.3, -1.3, -1.3]], dtype=torch.float64, device=device
    )
    down = voxelize(points, voxel_size=1.0).cloud.points
    assert down.shape == (1, 3)
    assert torch.allclose(down[0], points.mean(dim=0))


def test_voxelize_reduces_and_bounds(device):
    points = random_cloud(5000, device, seed=3)
    voxel = 0.1
    down = voxelize(points, voxel).cloud.points
    assert 0 < len(down) < len(points)
    # Each output point is a mean of points inside one voxel, so it stays
    # within the cloud's bounding box.
    assert (down >= points.amin(dim=0) - 1e-9).all()
    assert (down <= points.amax(dim=0) + 1e-9).all()


def test_voxelize_validation(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="voxel_size"):
        voxelize(points, 0.0)
