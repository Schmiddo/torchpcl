import pytest
import torch

from torchpcl.search import NearestNeighborSearch

from conftest import cubql_skip_reason, random_cloud


@pytest.fixture(params=["warp", "cubql"])
def nns_factory(request, device):
    if request.param == "cubql":
        if device.type != "cuda":
            pytest.skip("cubql backend is CUDA-only")
        if (reason := cubql_skip_reason()) is not None:
            pytest.skip(reason)
        from torchpcl.search_cubql import CuBQLNearestNeighborSearch

        return CuBQLNearestNeighborSearch
    return NearestNeighborSearch


def brute_force_nn(queries: torch.Tensor, points: torch.Tensor, radius: float):
    d = torch.cdist(queries.to(torch.float64), points.to(torch.float64))
    dist2, idx = (d**2).min(dim=1)
    valid = dist2 <= radius * radius
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    return idx, dist2, valid


def test_matches_brute_force(device, nns_factory):
    points = random_cloud(2000, device, seed=0)
    queries = random_cloud(500, device, seed=1)
    radius = 0.05

    nns = nns_factory(points, radius)
    idx, dist2 = nns.query(queries)
    ref_idx, ref_dist2, valid = brute_force_nn(queries, points, radius)

    assert torch.equal(idx >= 0, valid)
    # Indices match where the nearest neighbor is unique; squared
    # distances must match everywhere a neighbor was found.
    assert torch.equal(idx[valid], ref_idx[valid])
    assert torch.allclose(
        dist2[valid].to(torch.float64), ref_dist2[valid], atol=1e-5
    )


def test_out_of_radius_returns_minus_one(device, nns_factory):
    points = torch.zeros(1, 3, dtype=torch.float64, device=device)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.005, 0.0, 0.0]], dtype=torch.float64, device=device)
    nns = nns_factory(points, 0.01)
    idx, _ = nns.query(queries)
    assert idx.tolist() == [-1, 0]


def test_exact_match_found(device, nns_factory):
    points = random_cloud(100, device, seed=2)
    nns = nns_factory(points, 0.01)
    idx, dist2 = nns.query(points)
    assert (idx == torch.arange(100, device=device)).all()
    assert (dist2 == 0).all()


def test_backends_agree():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if (reason := cubql_skip_reason()) is not None:
        pytest.skip(reason)
    from torchpcl.search_cubql import CuBQLNearestNeighborSearch

    for seed, radius in [(0, 0.05), (1, 0.02), (2, 0.2)]:
        points = random_cloud(5000, "cuda", seed=seed)
        queries = random_cloud(2000, "cuda", seed=seed + 10)
        wi, wd = NearestNeighborSearch(points, radius).query(queries)
        ci, cd = CuBQLNearestNeighborSearch(points, radius).query(queries)

        found = wi >= 0
        assert torch.equal(found, ci >= 0)
        assert torch.allclose(wd[found], cd[found], atol=1e-6)
        # Compare indices only where the brute-force NN is unique (the
        # backends may tie-break differently).
        ref_idx, _, valid = brute_force_nn(queries, points, radius)
        d = torch.cdist(queries.to(torch.float64), points.to(torch.float64)) ** 2
        best = d.min(dim=1).values
        unique = valid & ((d <= best[:, None] + 1e-12).sum(dim=1) == 1)
        assert torch.equal(wi[unique], ci[unique])


def test_boundary_distance(device, nns_factory):
    # A point just inside the radius is found, just outside is not.
    points = torch.zeros(1, 3, dtype=torch.float64, device=device)
    radius = 0.1
    queries = torch.tensor(
        [[radius * 0.999, 0.0, 0.0], [radius * 1.001, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    nns = nns_factory(points, radius)
    idx, _ = nns.query(queries)
    assert idx.tolist() == [0, -1]
