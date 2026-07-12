import math

import pytest
import torch

import torchpcl as tp

from conftest import packed_ragged_batch


def _sorted_distances(result):
    return result.distances2.masked_fill(~result.valid, torch.inf).sort(dim=1).values


def test_knn_tensor_matches_cdist(search_device):
    reference = torch.rand(37, 3, device=search_device)
    queries = torch.rand(11, 3, device=search_device)

    result = tp.knn(reference, queries, 5)
    expected, _ = torch.cdist(queries, reference).square().topk(
        5, dim=1, largest=False
    )

    assert result.indices.shape == (11, 5)
    assert result.valid.all()
    torch.testing.assert_close(_sorted_distances(result), expected)


def test_packed_knn_is_batch_isolated_and_uses_global_indices():
    reference = tp.PointCloud(
        torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
        ),
        torch.tensor([0, 2, 4]),
    )
    queries = tp.PointCloud(
        torch.tensor([[1.9, 0.0, 0.0], [10.1, 0.0, 0.0]]),
        torch.tensor([0, 1, 2]),
    )

    result = tp.knn(reference, queries, 2)

    assert result.indices.sort(dim=1).values.tolist() == [[0, 1], [2, 3]]
    torch.testing.assert_close(
        _sorted_distances(result), torch.tensor([[0.01, 3.61], [0.01, 3.61]])
    )


def test_packed_one_shot_radius_and_hybrid():
    reference = tp.PointCloud(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ),
        torch.tensor([0, 2, 3]),
    )
    queries = tp.PointCloud(
        torch.tensor([[0.1, 0.0, 0.0], [10.2, 0.0, 0.0]]),
        torch.tensor([0, 1, 2]),
    )

    radius = tp.radius_neighbors(reference, queries, 0.5, max_neighbors=2)
    hybrid = tp.hybrid_neighbors(reference, queries, 0.5, 2)

    assert radius.valid.sum(dim=1).tolist() == [1, 1]
    assert radius.indices[radius.valid].tolist() == [0, 2]
    assert torch.equal(radius.indices.sort(dim=1).values, hybrid.indices.sort(dim=1).values)
    assert torch.equal(radius.valid, hybrid.valid)


def test_hybrid_has_explicit_invalid_slots(search_device):
    reference = torch.zeros(1, 3, dtype=torch.float64, device=search_device)
    queries = torch.tensor(
        [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=search_device,
    )

    result = tp.hybrid_neighbors(reference, queries, radius=0.5, k=2)

    assert result.indices.tolist() == [[0, -1], [-1, -1]]
    assert result.valid.tolist() == [[True, False], [False, False]]
    assert torch.isinf(result.distances2[~result.valid]).all()


def test_empty_query_batches_are_supported():
    points, offsets = packed_ragged_batch([2, 3], "cpu")
    queries, query_offsets = packed_ragged_batch([0, 0], "cpu")

    result = tp.knn(
        tp.PointCloud(points, offsets), tp.PointCloud(queries, query_offsets), 2
    )

    assert result.indices.shape == result.distances2.shape == (0, 2)


def test_empty_reference_returns_invalid_neighbors(search_device):
    reference = torch.empty(0, 3, device=search_device)
    queries = torch.randn(3, 3, device=search_device)

    result = tp.knn(reference, queries, 2)

    assert not result.valid.any()
    assert (result.indices == -1).all()
    assert torch.isinf(result.distances2).all()


def test_empty_packed_reference_batch_returns_invalid_row(search_device):
    reference = tp.PointCloud(
        torch.tensor([[10.0, 0.0, 0.0]], device=search_device),
        torch.tensor([0, 0, 1], device=search_device),
    )
    queries = tp.PointCloud(
        torch.tensor([[0.0, 0.0, 0.0], [10.1, 0.0, 0.0]], device=search_device),
        torch.tensor([0, 1, 2], device=search_device),
    )

    result = tp.knn(reference, queries, 1)

    assert result.valid[:, 0].tolist() == [False, True]
    assert result.indices[:, 0].tolist() == [-1, 0]


def test_neighbor_distances_have_gradients():
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True
    )
    queries = torch.tensor([[0.25, 0.0, 0.0]], requires_grad=True)

    distance = tp.knn(reference, queries, 1).distances2.sum()
    ref_grad, query_grad = torch.autograd.grad(distance, (reference, queries))

    torch.testing.assert_close(query_grad, torch.tensor([[0.5, 0.0, 0.0]]))
    torch.testing.assert_close(ref_grad[0], torch.tensor([-0.5, 0.0, 0.0]))
    assert torch.equal(ref_grad[1], torch.zeros(3))


def test_bvh_rejects_packed_reference():
    points, offsets = packed_ragged_batch([2, 3], "cpu")
    with pytest.raises(ValueError, match="bvh"):
        tp.NeighborIndex(tp.PointCloud(points, offsets), algorithm="bvh")


def test_knn_validation():
    points = torch.randn(3, 3)
    with pytest.raises(ValueError, match="same batch size"):
        tp.knn(
            tp.PointCloud.from_points(points),
            tp.PointCloud(points, torch.tensor([0, 1, 3])),
            1,
        )
    with pytest.raises(ValueError, match="positive"):
        tp.radius_neighbors(points, points, 0.0)
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        tp.knn(points, points, 65)


def test_unsupported_device_rejected():
    with pytest.raises(RuntimeError, match="unsupported device"):
        tp.NeighborIndex(torch.zeros(2, 3, device="meta"))


def test_unbounded_distance_is_finite():
    result = tp.NeighborIndex(torch.zeros(1, 3)).knn(
        torch.tensor([[1e10, 0.0, 0.0]]), 1
    )
    assert result.valid.item()
    assert result.distances2.item() == pytest.approx(1e20)
    assert math.isfinite(result.distances2.item())
