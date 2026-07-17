import pytest
import torch

from torchpcl import fscore, point_cloud_metrics

from conftest import random_cloud


def test_identical_clouds(search_device):
    points = random_cloud(1000, search_device, seed=0)
    m = point_cloud_metrics(points, points, threshold=0.01)
    zero = torch.zeros(1, dtype=points.dtype, device=points.device)
    torch.testing.assert_close(m.accuracy, zero, atol=1e-7, rtol=0)
    torch.testing.assert_close(m.completion, zero, atol=1e-7, rtol=0)
    torch.testing.assert_close(m.chamfer_distance, zero, atol=1e-7, rtol=0)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1_score == 1.0


def test_known_shift(search_device):
    # Grid with spacing 1, prediction shifted by 0.1 along x: every
    # nearest-neighbor distance is exactly 0.1 in both directions.
    axis = torch.arange(10, dtype=torch.float64, device=search_device)
    grid = torch.cartesian_prod(axis, axis, axis)
    shift = 0.1
    prediction = grid + torch.tensor([shift, 0.0, 0.0], dtype=torch.float64, device=search_device)

    m = point_cloud_metrics(prediction, grid, threshold=0.2)
    expected = torch.tensor([shift], dtype=grid.dtype, device=grid.device)
    torch.testing.assert_close(m.accuracy, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(m.completion, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(m.chamfer_distance, expected, atol=1e-7, rtol=0)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1_score == 1.0

    tight = point_cloud_metrics(prediction, grid, threshold=0.05)
    assert tight.precision == 0.0 and tight.recall == 0.0 and tight.f1_score == 0.0


def test_unbatched_fscore_retains_batch_dimension(search_device):
    points = random_cloud(100, search_device, seed=5)

    scores = fscore(points, points, threshold=0.01)

    assert scores.precision.shape == (1,)
    assert scores.precision.device == points.device
    torch.testing.assert_close(scores.f1_score, torch.ones_like(scores.f1_score))


def test_partial_prediction(search_device):
    reference = random_cloud(1000, search_device, seed=1)
    prediction = reference[:500]

    m = point_cloud_metrics(prediction, reference, threshold=0.01)
    torch.testing.assert_close(
        m.accuracy, torch.zeros_like(m.accuracy), atol=1e-7, rtol=0
    )
    assert m.precision == 1.0
    assert m.completion > 0.0
    assert m.recall < 1.0
    assert 0.0 < m.f1_score < 1.0


def test_direction_swap_symmetry(search_device):
    a = random_cloud(400, search_device, seed=2)
    b = random_cloud(600, search_device, seed=3) + 0.1

    ab = point_cloud_metrics(a, b, threshold=0.05)
    ba = point_cloud_metrics(b, a, threshold=0.05)
    torch.testing.assert_close(ab.accuracy, ba.completion)
    torch.testing.assert_close(ab.completion, ba.accuracy)
    torch.testing.assert_close(ab.precision, ba.recall)
    torch.testing.assert_close(ab.recall, ba.precision)
    torch.testing.assert_close(ab.chamfer_distance, ba.chamfer_distance)
    torch.testing.assert_close(ab.f1_score, ba.f1_score)


def test_validation(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="threshold"):
        point_cloud_metrics(points, points, threshold=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        point_cloud_metrics(points[:0], points, threshold=0.1)
