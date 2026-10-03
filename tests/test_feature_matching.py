import pytest
import torch

import torchpcl as tp


def test_match_features_returns_mutual_point_rows(search_device):
    source = tp.PointCloud.from_points(
        torch.randn(4, 3, device=search_device),
        features=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [8.0, 0.0]],
            device=search_device,
        ),
    )
    target = tp.PointCloud.from_points(
        torch.randn(3, 3, device=search_device),
        features=torch.tensor(
            [[1.1, 0.0], [0.1, 0.0], [8.2, 0.0]],
            device=search_device,
        ),
    )

    result = tp.match_features(source, target)

    assert result.source_indices.tolist() == [0, 1, 3]
    assert result.target_indices.tolist() == [1, 0, 2]
    assert torch.allclose(
        result.distances2,
        torch.tensor([0.01, 0.01, 0.04], device=search_device),
        atol=1e-6,
    )


def test_match_features_matches_dense_search_for_float64(search_device):
    generator = torch.Generator().manual_seed(4)
    source_features = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    target_features = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    source = tp.PointCloud.from_points(
        torch.randn(7, 3, generator=generator).to(search_device),
        features=source_features.to(search_device),
    )
    target = tp.PointCloud.from_points(
        torch.randn(6, 3, generator=generator).to(search_device),
        features=target_features.to(search_device),
    )

    result = tp.match_features(source, target, mutual=False)
    dense = torch.cdist(source.features, target.features).square()

    assert torch.equal(result.target_indices, dense.argmin(dim=1))
    assert torch.allclose(result.distances2, dense.min(dim=1).values)


def test_cpu_feature_matching_matches_dense_for_fpfh_dimension():
    source_features = torch.randn(73, 33)
    target_features = torch.randn(61, 33)
    source = tp.PointCloud.from_points(torch.randn(73, 3), features=source_features)
    target = tp.PointCloud.from_points(torch.randn(61, 3), features=target_features)
    result = tp.match_features(source, target, mutual=False)
    dense = torch.cdist(source_features, target_features).square()

    assert torch.equal(result.target_indices, dense.argmin(dim=1))
    assert torch.allclose(result.distances2, dense.min(dim=1).values, atol=1e-4)


@pytest.mark.parametrize("dimension", [33, 7])
def test_mutual_matching_breaks_ties_toward_lower_rows(search_device, dimension):
    # Small integers make exact distance ties common and every distance exact.
    generator = torch.Generator().manual_seed(5)
    source_features = torch.randint(0, 3, (300, dimension), generator=generator).float()
    target_features = torch.randint(0, 3, (257, dimension), generator=generator).float()
    source = tp.PointCloud.from_points(
        torch.zeros(300, 3, device=search_device),
        features=source_features.to(search_device),
    )
    target = tp.PointCloud.from_points(
        torch.zeros(257, 3, device=search_device),
        features=target_features.to(search_device),
    )
    result = tp.match_features(source, target, mutual=True)

    differences = source_features[:, None] - target_features[None]
    dense = differences.square().sum(dim=2)  # exact for small integers
    forward = dense.argmin(dim=1)  # first minimum: the lower row wins ties
    reverse = dense.argmin(dim=0)
    rows = torch.arange(300)
    keep = reverse[forward] == rows
    assert torch.equal(result.source_indices.cpu(), rows[keep])
    assert torch.equal(result.target_indices.cpu(), forward[keep])
    assert torch.equal(result.distances2.cpu(), dense.min(dim=1).values[keep])
