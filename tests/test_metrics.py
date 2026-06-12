import pytest
import torch

from torchpcl import point_cloud_metrics

from conftest import random_cloud


def test_identical_clouds(device):
    points = random_cloud(1000, device, seed=0)
    m = point_cloud_metrics(points, points, threshold=0.01)
    assert m.accuracy == pytest.approx(0.0, abs=1e-7)
    assert m.completion == pytest.approx(0.0, abs=1e-7)
    assert m.chamfer_distance == pytest.approx(0.0, abs=1e-7)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1_score == 1.0


def test_known_shift(device):
    # Grid with spacing 1, prediction shifted by 0.1 along x: every
    # nearest-neighbor distance is exactly 0.1 in both directions.
    axis = torch.arange(10, dtype=torch.float64, device=device)
    grid = torch.cartesian_prod(axis, axis, axis)
    shift = 0.1
    prediction = grid + torch.tensor([shift, 0.0, 0.0], dtype=torch.float64, device=device)

    m = point_cloud_metrics(prediction, grid, threshold=0.2)
    assert m.accuracy == pytest.approx(shift, abs=1e-9)
    assert m.completion == pytest.approx(shift, abs=1e-9)
    assert m.chamfer_distance == pytest.approx(2 * shift, abs=1e-9)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1_score == 1.0

    tight = point_cloud_metrics(prediction, grid, threshold=0.05)
    assert tight.precision == 0.0 and tight.recall == 0.0 and tight.f1_score == 0.0


def test_partial_prediction(device):
    reference = random_cloud(1000, device, seed=1)
    prediction = reference[:500]

    m = point_cloud_metrics(prediction, reference, threshold=0.01)
    assert m.accuracy == pytest.approx(0.0, abs=1e-7)
    assert m.precision == 1.0
    assert m.completion > 0.0
    assert m.recall < 1.0
    assert 0.0 < m.f1_score < 1.0


def test_direction_swap_symmetry(device):
    a = random_cloud(400, device, seed=2)
    b = random_cloud(600, device, seed=3) + 0.1

    ab = point_cloud_metrics(a, b, threshold=0.05)
    ba = point_cloud_metrics(b, a, threshold=0.05)
    assert ab.accuracy == pytest.approx(ba.completion)
    assert ab.completion == pytest.approx(ba.accuracy)
    assert ab.precision == pytest.approx(ba.recall)
    assert ab.recall == pytest.approx(ba.precision)
    assert ab.chamfer_distance == pytest.approx(ba.chamfer_distance)
    assert ab.f1_score == pytest.approx(ba.f1_score)


def test_validation(device):
    points = random_cloud(10, device, seed=0)
    with pytest.raises(ValueError, match="threshold"):
        point_cloud_metrics(points, points, threshold=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        point_cloud_metrics(points[:0], points, threshold=0.1)
    with pytest.raises(ValueError, match="unknown backend"):
        point_cloud_metrics(points, points, threshold=0.1, backend="kdtree")


def test_cubql_backend_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from conftest import cubql_skip_reason

    if (reason := cubql_skip_reason()) is not None:
        pytest.skip(reason)

    prediction = random_cloud(3000, "cuda", seed=4)
    reference = random_cloud(4000, "cuda", seed=5) + 0.02

    ref = point_cloud_metrics(prediction, reference, threshold=0.05)
    fast = point_cloud_metrics(prediction, reference, threshold=0.05, backend="cubql")
    assert fast.accuracy == pytest.approx(ref.accuracy, rel=1e-6)
    assert fast.completion == pytest.approx(ref.completion, rel=1e-6)
    assert fast.chamfer_distance == pytest.approx(ref.chamfer_distance, rel=1e-6)
    assert fast.precision == pytest.approx(ref.precision)
    assert fast.recall == pytest.approx(ref.recall)
    assert fast.f1_score == pytest.approx(ref.f1_score)
