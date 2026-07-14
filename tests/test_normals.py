import torch

import torchpcl as tp


def test_packed_normals_and_curvature(search_device):
    xy = torch.rand(80, 2, dtype=torch.float64, device=search_device)
    first = torch.cat([xy[:40], torch.zeros(40, 1, device=search_device)], dim=1)
    second = torch.cat([xy[40:], torch.ones(40, 1, device=search_device)], dim=1)
    cloud = tp.PointCloud(
        torch.cat([first, second]), torch.tensor([0, 40, 80], device=search_device)
    )

    result = tp.estimate_normals(cloud, k=10)

    assert result.valid.all()
    assert (result.normals[:, 2].abs() > 0.99).all()
    assert result.curvature.shape == (80,)
    assert (result.curvature < 1e-6).all()


def test_existing_normals_fix_the_sign(search_device):
    xy = torch.rand(50, 2, dtype=torch.float64, device=search_device)
    points = torch.cat([xy, torch.zeros(50, 1, device=search_device)], dim=1)
    signs = torch.where(
        torch.rand(50, device=search_device) < 0.5, 1.0, -1.0
    ).to(torch.float64)
    old = torch.zeros_like(points)
    old[:, 2] = signs
    cloud = tp.PointCloud.from_points(points, normals=old)

    result = tp.estimate_normals(cloud, k=10)

    assert (result.normals[:, 2] * signs > 0.99).all()


def test_normal_estimation_can_reuse_index(search_device):
    xy = torch.rand(30, 2, dtype=torch.float64, device=search_device)
    cloud = tp.PointCloud.from_points(
        torch.cat([xy, torch.zeros(30, 1, device=search_device)], dim=1)
    )
    index = tp.NeighborIndex(cloud)

    reused = tp.estimate_normals(cloud, k=8, index=index)
    direct = tp.estimate_normals(cloud, k=8)

    torch.testing.assert_close(reused.normals.abs(), direct.normals.abs())
