import math

import pytest
import torch

from torchpcl import search
from torchpcl.search import BatchedNearestNeighborSearch, NearestNeighborSearch


def available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available() and search._bruteforce_cuda is not None:
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_bruteforce_knn_matches_reference(device):
    generator = torch.Generator().manual_seed(7)
    points = torch.rand(37, 3, generator=generator, device="cpu").to(device)
    queries = torch.rand(9, 3, generator=generator, device="cpu").to(device)
    idx, dist2 = NearestNeighborSearch(
        points, 0.45, backend="bruteforce"
    ).knn_query(queries, 5)

    ref_d2, _ = (torch.cdist(queries.double(), points.double()) ** 2).topk(
        5, dim=1, largest=False
    )
    valid = ref_d2 <= 0.45**2
    assert torch.equal(idx >= 0, valid)
    assert torch.allclose(dist2[idx >= 0].double(), ref_d2[valid], atol=1e-6)


def test_packed_batches_use_global_indices():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
    )
    queries = torch.tensor([[1.9, 0.0, 0.0], [10.1, 0.0, 0.0]])
    offsets = torch.tensor([0, 2, 4])
    query_offsets = torch.tensor([0, 1, 2])
    idx, dist2 = BatchedNearestNeighborSearch(
        points, offsets, math.inf
    ).knn_query(queries, query_offsets, 2)

    assert idx.tolist() == [[1, 0], [2, 3]]
    assert torch.allclose(dist2, torch.tensor([[0.01, 3.61], [0.01, 3.61]]))


def test_bruteforce_empty_queries():
    points = torch.zeros(1, 3)
    queries = torch.empty(0, 3)
    idx, dist2 = NearestNeighborSearch(
        points, math.inf, backend="bruteforce"
    ).knn_query(queries, 3)
    assert idx.shape == dist2.shape == (0, 3)
