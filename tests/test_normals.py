import torch

import torchpcl as tp


def test_packed_normals_and_per_batch_viewpoints(search_device):
    xy = torch.rand(80, 2, dtype=torch.float64, device=search_device)
    first = torch.cat([xy[:40], torch.zeros(40, 1, device=search_device)], dim=1)
    second = torch.cat([xy[40:], torch.ones(40, 1, device=search_device)], dim=1)
    cloud = tp.PointCloud(
        torch.cat([first, second]), torch.tensor([0, 40, 80], device=search_device)
    )
    viewpoints = torch.tensor(
        [[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]],
        dtype=torch.float64,
        device=search_device,
    )

    result = tp.estimate_normals(
        cloud, k=10, orient="viewpoint", viewpoint=viewpoints,
        return_curvature=True,
    )

    assert result.valid.all()
    assert (result.normals[:40, 2] > 0.99).all()
    assert (result.normals[40:, 2] < -0.99).all()
    assert result.curvature.shape == (80,)


def test_normal_estimation_can_reuse_index(search_device):
    xy = torch.rand(30, 2, dtype=torch.float64, device=search_device)
    cloud = tp.PointCloud.from_points(
        torch.cat([xy, torch.zeros(30, 1, device=search_device)], dim=1)
    )
    index = tp.NeighborIndex(cloud)

    reused = tp.estimate_normals(cloud, k=8, index=index)
    direct = tp.estimate_normals(cloud, k=8)

    torch.testing.assert_close(reused.normals.abs(), direct.normals.abs())


def test_excluding_self_still_returns_k_neighbors(search_device):
    points = torch.rand(20, 3, device=search_device)
    result = tp.estimate_normals(points, k=5, include_self=False)
    assert result.normals.shape == points.shape
    assert result.valid.all()
