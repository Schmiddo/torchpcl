import math

import pytest
import torch

from torchpcl import _C


def backend_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available() and _C.has_cuda():
        devices.append(torch.device("cuda"))
    return devices


def _selected_distances(points, queries, indices):
    valid = indices >= 0
    gathered = points[indices.to(torch.int64).clamp(min=0)]
    distances = (queries[:, None] - gathered).square().sum(dim=2)
    return distances.masked_fill(~valid, torch.inf).sort(dim=1).values


@pytest.mark.parametrize("device", backend_devices())
def test_bvh_knn_selects_reference_topk(device):
    generator = torch.Generator(device="cpu").manual_seed(7)
    points = torch.rand(37, 3, generator=generator).to(device)
    queries = torch.rand(9, 3, generator=generator).to(device)

    indices, distances = _C.BvhIndex(points).knn(queries, 5, math.inf)
    expected = torch.cdist(queries, points).square().topk(
        5, dim=1, largest=False
    ).values

    assert indices.dtype == torch.int32
    assert distances.dtype == torch.float32
    torch.testing.assert_close(
        _selected_distances(points, queries, indices), expected, atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("device", backend_devices())
def test_packed_knn_is_batch_isolated(device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        device=device,
    )
    offsets = torch.tensor([0, 2, 3], device=device)
    queries = torch.tensor(
        [[1.9, 0.0, 0.0], [10.1, 0.0, 0.0]], device=device
    )
    query_batch_ids = torch.tensor([0, 1], device=device)

    indices, _ = _C.packed_knn(
        points, offsets, queries, query_batch_ids, 2, math.inf
    )

    assert indices[0].sort().values.tolist() == [0, 1]
    assert indices[1].tolist().count(2) == 1
    assert indices[1].tolist().count(-1) == 1


@pytest.mark.parametrize("device", backend_devices())
def test_packed_knn_empty_reference_batch_is_invalid(device):
    points = torch.tensor([[10.0, 0.0, 0.0]], device=device)
    offsets = torch.tensor([0, 0, 1], device=device)
    queries = torch.tensor(
        [[0.0, 0.0, 0.0], [10.1, 0.0, 0.0]], device=device
    )
    query_batch_ids = torch.tensor([0, 1], device=device)

    indices, _ = _C.packed_knn(
        points, offsets, queries, query_batch_ids, 2, math.inf
    )

    assert indices[0].tolist() == [-1, -1]
    assert 0 in indices[1].tolist()


@pytest.mark.parametrize("device", backend_devices())
def test_native_radius_boundary_and_empty_queries(device):
    points = torch.zeros(1, 3, device=device)
    index = _C.BvhIndex(points)
    queries = torch.tensor(
        [[0.0999, 0.0, 0.0], [0.1001, 0.0, 0.0]], device=device
    )

    indices, _ = index.knn(queries, 1, 0.1)
    empty_indices, empty_distances = index.knn(torch.empty(0, 3, device=device), 3, 1.0)

    assert indices[:, 0].tolist() == [0, -1]
    assert empty_indices.shape == empty_distances.shape == (0, 3)


def test_native_argument_validation():
    points = torch.zeros(2, 3)

    with pytest.raises(RuntimeError, match=r"k must be in \[1, 64\]"):
        _C.BvhIndex(points).knn(points, 65, math.inf)
    with pytest.raises(RuntimeError, match="float32"):
        _C.BvhIndex(points.to(torch.float64))
    with pytest.raises(RuntimeError, match="contiguous"):
        _C.BvhIndex(torch.zeros(3, 2).t())


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _C.has_cuda()), reason="CUDA backend unavailable"
)
def test_cuda_backend_uses_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        points = torch.rand(100, 3, device="cuda")
        queries = torch.rand(10, 3, device="cuda")
        indices, _ = _C.BvhIndex(points).knn(queries, 1, math.inf)
        marker = indices.clone()
    stream.synchronize()
    assert (marker >= 0).all()
