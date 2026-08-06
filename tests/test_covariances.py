import torch

import torchpcl as tp


def test_covariance_from_neighbors_is_analytic_and_differentiable(device):
    points = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    indices = torch.arange(3, device=device).expand(3, -1)
    neighbors = tp.Neighbors(
        indices=indices,
        distances2=torch.zeros(3, 3, dtype=points.dtype, device=device),
        valid=torch.ones(3, 3, dtype=torch.bool, device=device),
    )

    sample = tp.covariance_from_neighbors(points, neighbors)
    population = tp.covariance_from_neighbors(points, neighbors, correction=0)
    sample.covariances.sum().backward()

    torch.testing.assert_close(sample.means, torch.zeros_like(points))
    torch.testing.assert_close(
        sample.covariances[:, 0, 0],
        torch.ones(3, device=device, dtype=points.dtype),
    )
    torch.testing.assert_close(
        population.covariances[:, 0, 0],
        torch.full((3,), 2 / 3, device=device, dtype=points.dtype),
    )
    assert sample.valid.all()
    assert torch.isfinite(points.grad).all()


def test_estimate_covariances_matches_centered_knn_reference(search_device):
    generator = torch.Generator().manual_seed(4)
    points = torch.randn(8, 3, generator=generator, dtype=torch.float64).to(
        search_device
    )
    result = tp.estimate_covariances(points, k=4)
    neighbors = tp.knn(points, points, k=4)
    gathered = points[neighbors.indices]
    expected_means = gathered.mean(dim=1)
    centered = gathered - expected_means[:, None]
    expected_covariances = centered.transpose(1, 2) @ centered / 3

    torch.testing.assert_close(result.means, expected_means)
    torch.testing.assert_close(result.covariances, expected_covariances)
    assert result.counts.tolist() == [4] * 8
    assert result.valid.all()


def test_hybrid_covariances_keep_empty_batches_and_zero_invalid_rows(search_device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [5.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=search_device,
    )
    cloud = tp.PointCloud(
        points, torch.tensor([0, 0, 3, 4], device=search_device)
    )

    result = tp.estimate_covariances(cloud, radius=0.2, k=4)

    assert result.counts.tolist() == [3, 3, 3, 1]
    assert result.valid.tolist() == [True, True, True, False]
    assert torch.count_nonzero(result.means[-1]) == 0
    assert torch.count_nonzero(result.covariances[-1]) == 0
