import pytest
import torch

import torchpcl as tp


def test_farthest_point_sample_exact_sequence_on_a_line(search_device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)],
        dtype=torch.float64,
        device=search_device,
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


def test_farthest_point_sample_matches_across_devices(search_device):
    if search_device.type == "cpu":
        pytest.skip("compares the CUDA kernel against the CPU reference")
    # Wide enough that the CUDA grid spans several blocks, and half-duplicated
    # so the lower-index tie-break has to hold across block boundaries.
    points = torch.rand(20_000, 3, dtype=torch.float64, generator=
                        torch.Generator().manual_seed(0))
    points[10_000:] = points[:10_000]

    expected = tp.farthest_point_sample(points, 1_000).points
    result = tp.farthest_point_sample(points.to(search_device), 1_000).points

    assert torch.equal(result.cpu(), expected)


def test_farthest_point_sample_clamps_to_all_points(device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)], dtype=torch.float64, device=device
    )

    result = tp.farthest_point_sample(points, 10)

    assert result.points.tolist() == points.tolist()


def test_farthest_point_sample_preserves_normals_and_features(search_device):
    points = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(6)],
        dtype=torch.float64,
        device=search_device,
    )
    normals = torch.zeros(6, 3, dtype=torch.float64, device=search_device)
    normals[:, 2] = 1.0
    features = torch.arange(12, dtype=torch.float64, device=search_device).reshape(6, 2)
    cloud = tp.PointCloud.from_points(points, normals=normals, features=features)

    result = tp.farthest_point_sample(cloud, 4)

    selected = torch.tensor([0, 5, 2, 1], device=search_device)
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
