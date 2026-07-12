import pytest
import torch

from torchpcl import PointCloud, chamfer_distance, point_cloud_metrics

from conftest import random_cloud


def bruteforce_chamfer(prediction, reference, squared):
    d2 = ((prediction[:, None] - reference[None]) ** 2).sum(-1)
    fwd = d2.min(dim=1).values
    bwd = d2.min(dim=0).values
    if not squared:
        fwd, bwd = fwd.sqrt(), bwd.sqrt()
    return (fwd.mean() + bwd.mean()) / 2


@pytest.mark.parametrize("squared", [True, False])
def test_matches_bruteforce_values_and_grads(search_device, squared):
    prediction = random_cloud(300, search_device, seed=0).requires_grad_()
    reference = (random_cloud(400, search_device, seed=1) + 0.1).requires_grad_()

    loss = chamfer_distance(prediction, reference, squared=squared)
    grads = torch.autograd.grad(loss, (prediction, reference))

    expected = bruteforce_chamfer(prediction, reference, squared)
    expected_grads = torch.autograd.grad(expected, (prediction, reference))

    assert loss.item() == pytest.approx(expected.item())
    for got, want in zip(grads, expected_grads):
        torch.testing.assert_close(got, want)


def test_unsquared_matches_metric(search_device):
    prediction = random_cloud(500, search_device, seed=2)
    reference = random_cloud(600, search_device, seed=3)
    loss = chamfer_distance(prediction, reference, squared=False)
    m = point_cloud_metrics(prediction, reference, threshold=0.05)
    torch.testing.assert_close(loss, m.chamfer_distance)


def test_identical_clouds_zero_loss_finite_grad(search_device):
    points = random_cloud(200, search_device, seed=4)
    prediction = points.clone().requires_grad_()
    loss = chamfer_distance(prediction, points, squared=False)
    assert loss.item() == 0.0
    # sqrt'(0) is infinite; the safe sqrt must keep the gradient finite.
    (grad,) = torch.autograd.grad(loss, prediction)
    assert torch.isfinite(grad).all()


def test_batched_matches_per_sample(search_device):
    pred_lengths = torch.tensor([200, 150, 300])
    ref_lengths = torch.tensor([250, 300, 100])
    # Padding holds garbage that must not influence the result.
    prediction = torch.full((3, 300, 3), 1e6, dtype=torch.float64, device=search_device)
    reference = torch.full((3, 300, 3), -1e6, dtype=torch.float64, device=search_device)
    for b in range(3):
        prediction[b, : pred_lengths[b]] = random_cloud(int(pred_lengths[b]), search_device, seed=10 + b)
        reference[b, : ref_lengths[b]] = random_cloud(int(ref_lengths[b]), search_device, seed=20 + b)

    prediction_cloud = PointCloud.from_padded(prediction, pred_lengths)
    reference_cloud = PointCloud.from_padded(reference, ref_lengths)
    losses = chamfer_distance(prediction_cloud, reference_cloud, reduction="none")
    expected = torch.stack([
        chamfer_distance(
            prediction[b, : pred_lengths[b]], reference[b, : ref_lengths[b]]
        )
        for b in range(3)
    ])
    torch.testing.assert_close(losses, expected)

    mean = chamfer_distance(prediction_cloud, reference_cloud)
    assert mean.item() == pytest.approx(expected.mean().item())
