import torch

from torchpcl import PointToPlane, PointToPoint
from torchpcl.transforms import transform_points

from conftest import random_cloud, random_rigid_transform


def test_point_to_point_recovers_exact_transform(device):
    source = random_cloud(200, device, seed=0)
    gt = random_rigid_transform(max_angle=0.5, max_translation=1.0, seed=1, device=device)
    target = transform_points(source, gt)

    delta = PointToPoint().compute_transformation(source, target, None)
    assert torch.allclose(delta, gt, atol=1e-9)


def test_point_to_point_requires_no_normals():
    assert PointToPoint.requires_normals is False
    assert PointToPlane.requires_normals is True


def _cloud_with_normals(n, device, seed):
    # Independent random normals keep the 6x6 system well conditioned.
    # (Surface-consistent normals, e.g. a sphere with radial normals,
    # put rotation in the Jacobian's null space: cross(s, n) ≈ 0.)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    points = torch.randn((n, 3), generator=gen, dtype=torch.float64)
    normals = torch.randn((n, 3), generator=gen, dtype=torch.float64)
    normals = normals / normals.norm(dim=1, keepdim=True)
    return points.to(device), normals.to(device)


def test_point_to_plane_recovers_small_transform(device):
    target, normals = _cloud_with_normals(500, device, seed=0)
    gt = random_rigid_transform(max_angle=0.01, max_translation=0.01, seed=1, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    delta = PointToPlane().compute_transformation(source, target, normals)
    assert torch.allclose(delta, gt, atol=1e-4)


def test_point_to_plane_reduces_error(device):
    target, normals = _cloud_with_normals(500, device, seed=2)
    gt = random_rigid_transform(max_angle=0.1, max_translation=0.1, seed=3, device=device)
    source = transform_points(target, torch.linalg.inv(gt))

    def p2pl_error(points):
        return (((points - target) * normals).sum(-1) ** 2).sum()

    delta = PointToPlane().compute_transformation(source, target, normals)
    assert p2pl_error(transform_points(source, delta)) < p2pl_error(source)


def test_point_to_plane_raises_without_normals(device):
    source = random_cloud(10, device, seed=0)
    try:
        PointToPlane().compute_transformation(source, source, None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
