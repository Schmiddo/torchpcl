import torch

from torchpcl.search import NearestNeighborSearch

from conftest import random_cloud


def brute_force_nn(queries: torch.Tensor, points: torch.Tensor, radius: float):
    d = torch.cdist(queries.to(torch.float64), points.to(torch.float64))
    dist2, idx = (d**2).min(dim=1)
    valid = dist2 <= radius * radius
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    return idx, dist2, valid


def test_matches_brute_force(device):
    points = random_cloud(2000, device, seed=0)
    queries = random_cloud(500, device, seed=1)
    radius = 0.05

    nns = NearestNeighborSearch(points, radius)
    idx, dist2 = nns.query(queries)
    ref_idx, ref_dist2, valid = brute_force_nn(queries, points, radius)

    assert torch.equal(idx >= 0, valid)
    # Indices match where the nearest neighbor is unique; squared
    # distances must match everywhere a neighbor was found.
    assert torch.equal(idx[valid], ref_idx[valid])
    assert torch.allclose(
        dist2[valid].to(torch.float64), ref_dist2[valid], atol=1e-5
    )


def test_out_of_radius_returns_minus_one(device):
    points = torch.zeros(1, 3, dtype=torch.float64, device=device)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.005, 0.0, 0.0]], dtype=torch.float64, device=device)
    nns = NearestNeighborSearch(points, 0.01)
    idx, _ = nns.query(queries)
    assert idx.tolist() == [-1, 0]


def test_exact_match_found(device):
    points = random_cloud(100, device, seed=2)
    nns = NearestNeighborSearch(points, 0.01)
    idx, dist2 = nns.query(points)
    assert (idx == torch.arange(100, device=device)).all()
    assert (dist2 == 0).all()


def test_boundary_distance(device):
    # A point just inside the radius is found, just outside is not.
    points = torch.zeros(1, 3, dtype=torch.float64, device=device)
    radius = 0.1
    queries = torch.tensor(
        [[radius * 0.999, 0.0, 0.0], [radius * 1.001, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    nns = NearestNeighborSearch(points, radius)
    idx, _ = nns.query(queries)
    assert idx.tolist() == [0, -1]
