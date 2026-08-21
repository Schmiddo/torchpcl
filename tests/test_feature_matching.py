import torch

import torchpcl as tp


def test_match_features_returns_mutual_point_rows(device):
    source = tp.PointCloud.from_points(
        torch.randn(4, 3, device=device),
        features=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [8.0, 0.0]],
            device=device,
        ),
    )
    target = tp.PointCloud.from_points(
        torch.randn(3, 3, device=device),
        features=torch.tensor(
            [[1.1, 0.0], [0.1, 0.0], [8.2, 0.0]],
            device=device,
        ),
    )

    result = tp.match_features(source, target)

    assert result.source_indices.tolist() == [0, 1, 3]
    assert result.target_indices.tolist() == [1, 0, 2]
    assert torch.allclose(
        result.distances2,
        torch.tensor([0.01, 0.01, 0.04], device=device),
        atol=1e-6,
    )


def test_match_features_matches_dense_search_for_float64(device):
    generator = torch.Generator().manual_seed(4)
    source_features = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    target_features = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    source = tp.PointCloud.from_points(
        torch.randn(7, 3, generator=generator).to(device),
        features=source_features.to(device),
    )
    target = tp.PointCloud.from_points(
        torch.randn(6, 3, generator=generator).to(device),
        features=target_features.to(device),
    )

    result = tp.match_features(source, target, mutual=False)
    dense = torch.cdist(source.features, target.features).square()

    assert torch.equal(result.target_indices, dense.argmin(dim=1))
    assert torch.allclose(result.distances2, dense.min(dim=1).values)
