import math

import pytest
import torch

from torchpcl.search import NearestNeighborSearch

from conftest import random_cloud


def brute_force_nn(queries: torch.Tensor, points: torch.Tensor, radius: float):
    d = torch.cdist(queries.to(torch.float64), points.to(torch.float64))
    dist2, idx = (d**2).min(dim=1)
    valid = dist2 <= radius * radius
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    return idx, dist2, valid


def test_matches_brute_force(cuda_device):
    points = random_cloud(2000, cuda_device, seed=0)
    queries = random_cloud(500, cuda_device, seed=1)
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


def test_out_of_radius_returns_minus_one(cuda_device):
    points = torch.zeros(1, 3, dtype=torch.float64, device=cuda_device)
    queries = torch.tensor(
        [[1.0, 0.0, 0.0], [0.005, 0.0, 0.0]], dtype=torch.float64, device=cuda_device
    )
    nns = NearestNeighborSearch(points, 0.01)
    idx, _ = nns.query(queries)
    assert idx.tolist() == [-1, 0]


def test_exact_match_found(cuda_device):
    points = random_cloud(100, cuda_device, seed=2)
    nns = NearestNeighborSearch(points, 0.01)
    idx, dist2 = nns.query(points)
    assert (idx == torch.arange(100, device=cuda_device)).all()
    assert (dist2 == 0).all()


def test_boundary_distance(cuda_device):
    # A point just inside the radius is found, just outside is not.
    points = torch.zeros(1, 3, dtype=torch.float64, device=cuda_device)
    radius = 0.1
    queries = torch.tensor(
        [[radius * 0.999, 0.0, 0.0], [radius * 1.001, 0.0, 0.0]],
        dtype=torch.float64,
        device=cuda_device,
    )
    nns = NearestNeighborSearch(points, radius)
    idx, _ = nns.query(queries)
    assert idx.tolist() == [0, -1]


def test_unbounded_query(cuda_device):
    points = random_cloud(100, cuda_device, seed=3) + 100.0
    queries = random_cloud(10, cuda_device, seed=4)
    idx, _ = NearestNeighborSearch(points, math.inf).query(queries)
    assert (idx >= 0).all()


def test_knn_query_matches_brute_force(cuda_device):
    points = random_cloud(1000, cuda_device, seed=0)
    queries = random_cloud(200, cuda_device, seed=1)
    radius, k = 0.1, 5

    idx, dist2 = NearestNeighborSearch(points, radius).knn_query(queries, k)

    d = torch.cdist(queries.to(torch.float64), points.to(torch.float64)) ** 2
    ref_d, _ = d.topk(k, dim=1, largest=False)
    ref_valid = ref_d <= radius * radius

    valid = idx >= 0
    assert torch.equal(valid, ref_valid)
    assert torch.allclose(
        dist2[valid].to(torch.float64), ref_d[ref_valid], atol=1e-5
    )
    # Rows come back sorted by distance (inf-padded; inf-inf diffs are
    # nan and carry no ordering information).
    padded = torch.where(valid, dist2, torch.full_like(dist2, float("inf")))
    diffs = padded.diff(dim=1)
    assert (diffs[torch.isfinite(diffs)] >= -1e-7).all()


def test_knn_unbounded(cuda_device):
    points = random_cloud(1000, cuda_device, seed=0)
    queries = random_cloud(200, cuda_device, seed=1)
    k = 8
    idx, dist2 = NearestNeighborSearch(points, math.inf).knn_query(queries, k)
    assert (idx >= 0).all()  # unbounded: k neighbors always found

    d = torch.cdist(queries.to(torch.float64), points.to(torch.float64)) ** 2
    ref_d, _ = d.topk(k, dim=1, largest=False)
    assert torch.allclose(dist2.to(torch.float64), ref_d, atol=1e-5)


def test_cpu_points_rejected():
    points = torch.zeros(10, 3)
    with pytest.raises(RuntimeError, match="CUDA-only"):
        NearestNeighborSearch(points, 0.1)
