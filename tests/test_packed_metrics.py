import pytest
import torch

import torchpcl as tp


def _packed_metric_clouds(device):
    prediction = tp.PointCloud(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        ),
        torch.tensor([0, 2, 3], device=device),
    )
    reference = tp.PointCloud(
        torch.tensor(
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [10.2, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        ),
        torch.tensor([0, 2, 3], device=device),
    )
    return prediction, reference


def test_packed_chamfer_returns_per_batch_values(search_device):
    prediction, reference = _packed_metric_clouds(search_device)

    values = tp.chamfer_distance(
        prediction, reference, squared=False, reduction="none"
    )

    torch.testing.assert_close(
        values, torch.tensor([0.1, 0.2], dtype=torch.float64, device=search_device)
    )


def test_fscore_supports_one_threshold(search_device):
    prediction, reference = _packed_metric_clouds(search_device)

    scores = tp.fscore(prediction, reference, 0.15)

    assert scores.precision.shape == (2,)
    torch.testing.assert_close(
        scores.f1_score,
        torch.tensor([1.0, 0.0], dtype=torch.float64, device=search_device),
    )

    with pytest.raises(ValueError, match="scalar"):
        tp.fscore(prediction, reference, torch.tensor([0.1, 0.2]))


def test_dense_tensor_and_point_cloud_are_interchangeable(search_device):
    points = torch.randn(2, 6, 3, device=search_device)
    cloud = tp.as_point_cloud(points)

    distance = tp.chamfer_distance(points, cloud, reduction="none")

    torch.testing.assert_close(distance, torch.zeros_like(distance))


def test_packed_chamfer_gradients_match_dense_pairs(search_device):
    prediction, reference = _packed_metric_clouds(search_device)
    prediction_points = prediction.points.clone().requires_grad_()
    reference_points = reference.points.clone().requires_grad_()
    prediction = tp.PointCloud(prediction_points, prediction.offsets)
    reference = tp.PointCloud(reference_points, reference.offsets)

    loss = tp.chamfer_distance(prediction, reference, reduction="sum")
    gradients = torch.autograd.grad(loss, (prediction_points, reference_points))

    expected = sum(
        tp.chamfer_distance(
            prediction_points[prediction.offsets[b] : prediction.offsets[b + 1]],
            reference_points[reference.offsets[b] : reference.offsets[b + 1]],
        )
        for b in range(2)
    )
    expected_gradients = torch.autograd.grad(
        expected, (prediction_points, reference_points)
    )
    for actual, wanted in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, wanted)


def test_unsquared_chamfer_has_finite_zero_gradient(search_device):
    points = torch.rand(10, 3, dtype=torch.float64, device=search_device)
    prediction = points.clone().requires_grad_()

    loss = tp.chamfer_distance(prediction, points, squared=False)
    gradient = torch.autograd.grad(loss, prediction)[0]

    assert loss == 0
    assert torch.isfinite(gradient).all()


def test_point_cloud_metrics_are_device_tensors(search_device):
    prediction, reference = _packed_metric_clouds(search_device)

    metrics = tp.point_cloud_metrics(prediction, reference, threshold=0.15)

    assert metrics.accuracy.shape == (2,)
    assert metrics.accuracy.device == prediction.device
    assert metrics.precision.device == prediction.device
