import pytest
import torch

from torchpcl import estimate_normals, voxel_downsample

from conftest import cubql_skip_reason, random_cloud


def _require_cubql():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if (reason := cubql_skip_reason()) is not None:
        pytest.skip(reason)


def test_normals_cubql_unbounded():
    _require_cubql()
    gen = torch.Generator().manual_seed(0)
    xy = torch.rand((2000, 2), generator=gen, dtype=torch.float64)
    points = torch.cat([xy, torch.zeros(2000, 1, dtype=torch.float64)], dim=1).cuda()

    normals = estimate_normals(points, k=20, backend="cubql")  # no radius
    assert torch.allclose(
        normals[:, 2].abs(), torch.ones(2000, dtype=torch.float64, device="cuda"), atol=1e-6
    )


def test_normals_backends_agree():
    _require_cubql()
    gen = torch.Generator().manual_seed(1)
    points = torch.randn((5000, 3), generator=gen, dtype=torch.float64)
    points = (points / points.norm(dim=1, keepdim=True)).cuda()

    warp_n = estimate_normals(points, radius=0.2, k=20, backend="warp")
    cubql_n = estimate_normals(points, radius=0.2, k=20, backend="cubql")
    alignment = (warp_n * cubql_n).sum(dim=1).abs()
    # Same hybrid semantics; tie-breaks may differ on a few neighborhoods.
    assert float(alignment.median()) > 0.9999
    assert float((alignment > 0.99).double().mean()) > 0.99


def test_normals_unknown_backend(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="unknown backend"):
        estimate_normals(points, radius=0.1, backend="kdtree")


def test_normals_warp_requires_radius(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="radius is required"):
        estimate_normals(points)


def test_normals_on_plane(device):
    gen = torch.Generator().manual_seed(0)
    xy = torch.rand((2000, 2), generator=gen, dtype=torch.float64)
    points = torch.cat([xy, torch.zeros(2000, 1, dtype=torch.float64)], dim=1).to(device)

    normals = estimate_normals(points, radius=0.1, k=20)
    assert normals.shape == points.shape
    assert torch.allclose(normals[:, 2].abs(), torch.ones(2000, dtype=torch.float64, device=device), atol=1e-6)
    assert torch.allclose(normals.norm(dim=1), torch.ones(2000, dtype=torch.float64, device=device), atol=1e-6)


def test_normals_on_sphere(device):
    gen = torch.Generator().manual_seed(1)
    points = torch.randn((5000, 3), generator=gen, dtype=torch.float64)
    points = (points / points.norm(dim=1, keepdim=True)).to(device)

    normals = estimate_normals(points, radius=0.2, k=30)
    # Normals of a unit sphere are the (possibly sign-flipped) positions.
    alignment = (normals * points).sum(dim=1).abs()
    assert float(alignment.min()) > 0.98


def test_normals_viewpoint_orientation(device):
    gen = torch.Generator().manual_seed(2)
    xy = torch.rand((500, 2), generator=gen, dtype=torch.float64)
    points = torch.cat([xy, torch.zeros(500, 1, dtype=torch.float64)], dim=1).to(device)

    viewpoint = torch.tensor([0.5, 0.5, 10.0], dtype=torch.float64, device=device)
    normals = estimate_normals(points, radius=0.1, k=20, viewpoint=viewpoint)
    assert (normals[:, 2] > 0.99).all()


def test_normals_isolated_points_zero(device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    normals = estimate_normals(points, radius=0.1, k=5)
    assert (normals == 0).all()


def test_normals_input_validation(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="radius"):
        estimate_normals(points, radius=0.0)
    with pytest.raises(ValueError, match="k must be"):
        estimate_normals(points, radius=0.1, k=2)


def test_voxel_downsample_means(device):
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
    down = voxel_downsample(points, voxel_size=1.0)
    assert down.shape == (2, 3)
    expected = torch.stack([points[:2].mean(dim=0), points[2:].mean(dim=0)])
    # Output voxel order is arbitrary; sort both by x.
    down = down[down[:, 0].argsort()]
    expected = expected[expected[:, 0].argsort()]
    assert torch.allclose(down, expected)


def test_voxel_downsample_negative_coords(device):
    points = torch.tensor(
        [[-1.4, -1.4, -1.4], [-1.3, -1.3, -1.3]], dtype=torch.float64, device=device
    )
    down = voxel_downsample(points, voxel_size=1.0)
    assert down.shape == (1, 3)
    assert torch.allclose(down[0], points.mean(dim=0))


def test_voxel_downsample_reduces_and_bounds(device):
    points = random_cloud(5000, device, seed=3)
    voxel = 0.1
    down = voxel_downsample(points, voxel)
    assert 0 < len(down) < len(points)
    # Each output point is a mean of points inside one voxel, so it stays
    # within the cloud's bounding box.
    assert (down >= points.amin(dim=0) - 1e-9).all()
    assert (down <= points.amax(dim=0) + 1e-9).all()


def test_voxel_downsample_validation(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="voxel_size"):
        voxel_downsample(points, 0.0)
