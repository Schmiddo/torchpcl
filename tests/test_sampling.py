import pytest
import torch

import torchpcl as tp


def test_farthest_point_sample_exact_sequence_on_a_line(device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)], dtype=torch.float64, device=device
    )

    result = tp.farthest_point_sample(points, 4)

    # Selection order (not input order): 0 (start), then the farthest point,
    # then whichever point maximizes the distance to both prior selections.
    assert result.points.tolist() == [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert result.offsets.tolist() == [0, 4]


def test_farthest_point_sample_clamps_to_all_points(device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)], dtype=torch.float64, device=device
    )

    result = tp.farthest_point_sample(points, 10)

    assert result.points.tolist() == points.tolist()


def test_farthest_point_sample_preserves_normals_and_features(device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)], dtype=torch.float64, device=device
    )
    normals = torch.zeros(6, 3, dtype=torch.float64, device=device)
    normals[:, 2] = 1.0
    features = torch.arange(12, dtype=torch.float64, device=device).reshape(6, 2)
    cloud = tp.PointCloud.from_points(points, normals=normals, features=features)

    result = tp.farthest_point_sample(cloud, 4)

    selected = torch.tensor([0, 5, 2, 1], device=device)
    assert result.normals.tolist() == normals[selected].tolist()
    assert result.features.tolist() == features[selected].tolist()


def test_farthest_point_sample_rejects_batched_input(device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)], dtype=torch.float64, device=device
    )
    batched = points.unsqueeze(0).repeat(2, 1, 1)

    with pytest.raises(ValueError):
        tp.farthest_point_sample(batched, 2)


def test_farthest_point_sample_rejects_nonpositive_num_samples(device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64, device=device
    )

    with pytest.raises(ValueError):
        tp.farthest_point_sample(points, 0)
