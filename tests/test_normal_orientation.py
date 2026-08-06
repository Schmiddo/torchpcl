import pytest
import torch

import torchpcl as tp


def test_orient_normals_toward_and_away_from_viewpoint(device):
    points = torch.tensor(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        device=device,
    )
    normals = torch.tensor(
        [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device=device,
        requires_grad=True,
    )
    cloud = tp.PointCloud.from_points(points, normals=normals)
    viewpoint = torch.zeros(3, device=device)

    cloud = tp.orient_normals_toward_viewpoint(cloud, viewpoint)
    torch.testing.assert_close(
        cloud.normals,
        torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=device,
        ),
    )
    cloud = tp.orient_normals_toward_viewpoint(cloud, viewpoint, toward=False)
    torch.testing.assert_close(
        cloud.normals,
        torch.tensor(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=device,
        ),
    )
    cloud.normals.sum().backward()
    assert torch.equal(normals.grad.abs(), torch.ones_like(normals))


def test_per_cloud_direction_orientation_is_batch_isolated(device):
    points = torch.zeros(3, 3, device=device)
    normals = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        device=device,
    )
    cloud = tp.PointCloud(
        points,
        torch.tensor([0, 2, 2, 3], device=device),
        normals=normals,
    )
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
    )

    aligned = tp.orient_normals_to_direction(cloud, directions)
    opposed = tp.orient_normals_to_direction(cloud, directions, align=False)

    assert (aligned.normals[:2, 0] >= 0).all()
    assert aligned.normals[2, 1] > 0
    assert (opposed.normals[:2, 0] <= 0).all()
    assert opposed.normals[2, 1] < 0


def test_orientation_requires_attached_normals_and_nonzero_direction(device):
    cloud = tp.PointCloud.from_points(torch.zeros(1, 3, device=device))
    with pytest.raises(ValueError, match="normals"):
        tp.orient_normals_toward_viewpoint(cloud, torch.zeros(3, device=device))

    cloud = tp.PointCloud.from_points(cloud.points, normals=torch.ones(1, 3, device=device))
    with pytest.raises(ValueError, match="nonzero"):
        tp.orient_normals_to_direction(cloud, torch.zeros(3, device=device))
