import pytest
import torch

from torchpcl import PointCloud, transform
from torchpcl.transforms import pose_to_matrix, transform_points


def _rot_x(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)


def _rot_y(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)


def _rot_z(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def test_pose_to_matrix_matches_zyx_composition():
    pose = torch.tensor([0.3, -0.2, 0.5, 1.0, -2.0, 3.0], dtype=torch.float64)
    t = pose_to_matrix(pose)
    expected_r = _rot_z(pose[2]) @ _rot_y(pose[1]) @ _rot_x(pose[0])
    assert torch.allclose(t[:3, :3], expected_r, atol=1e-12)
    assert torch.allclose(t[:3, 3], pose[3:], atol=1e-12)
    assert torch.allclose(t[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))


def test_pose_to_matrix_small_angle_approximation():
    pose = torch.tensor([1e-6, -2e-6, 3e-6, 0.0, 0.0, 0.0], dtype=torch.float64)
    t = pose_to_matrix(pose)
    skew = torch.tensor(
        [[0, -pose[2], pose[1]], [pose[2], 0, -pose[0]], [-pose[1], pose[0], 0]],
        dtype=torch.float64,
    )
    assert torch.allclose(t[:3, :3], torch.eye(3, dtype=torch.float64) + skew, atol=1e-11)


def test_pose_to_matrix_is_rigid():
    pose = torch.tensor([1.1, 0.7, -2.0, 0.5, 0.0, -0.3], dtype=torch.float64)
    r = pose_to_matrix(pose)[:3, :3]
    assert torch.allclose(r @ r.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(torch.linalg.det(r), torch.tensor(1.0, dtype=torch.float64))


def test_transform_points_round_trip(device):
    points = torch.randn(100, 3, dtype=torch.float64, device=device)
    t = pose_to_matrix(torch.tensor([0.2, 0.1, -0.3, 1.0, 2.0, 3.0], dtype=torch.float64)).to(device)
    transformed = transform_points(points, t)
    restored = transform_points(transformed, torch.linalg.inv(t))
    assert torch.allclose(restored, points, atol=1e-10)


def test_transform_packed_batch_and_normals(device):
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    normals = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    features = torch.arange(3, device=device)
    cloud = PointCloud(
        points,
        torch.tensor([0, 2, 3], device=device),
        normals=normals,
        features=features,
    )
    transforms = torch.eye(4, dtype=torch.float64, device=device).repeat(2, 1, 1)
    transforms[0, 0, 3] = 10.0
    transforms[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )

    moved = transform(cloud, transforms)

    expected_points = torch.tensor(
        [[11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    expected_normals = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    assert torch.allclose(moved.points, expected_points)
    assert torch.allclose(moved.normals, expected_normals)
    assert moved.offsets is cloud.offsets
    assert moved.features is cloud.features


def test_transform_broadcasts_one_matrix_over_empty_batches(device):
    cloud = PointCloud(
        torch.tensor([[1.0, 2.0, 3.0]], device=device),
        torch.tensor([0, 0, 1, 1], device=device),
    )
    matrix = torch.eye(4, device=device)
    matrix[:3, 3] = torch.tensor([2.0, 3.0, 4.0], device=device)

    moved = transform(cloud, matrix)

    assert moved.offsets.tolist() == [0, 0, 1, 1]
    assert torch.equal(moved.points, torch.tensor([[3.0, 5.0, 7.0]], device=device))


def test_transform_promotes_geometry_dtype(device):
    points = torch.randn(4, 3, dtype=torch.float32, device=device)
    matrix = torch.eye(4, dtype=torch.float64, device=device)

    moved = transform(points, matrix)

    assert moved.dtype == torch.float64


def test_transform_restores_dense_batch_shape(device):
    points = torch.zeros(2, 3, 3, device=device)
    matrices = torch.eye(4, device=device).repeat(2, 1, 1)
    matrices[:, :3, 3] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device
    )

    moved = transform(points, matrices)

    assert moved.shape == points.shape
    torch.testing.assert_close(moved[:, 0], matrices[:, :3, 3])

    single = transform(points[0], matrices[:1])
    torch.testing.assert_close(single[0], matrices[0, :3, 3])


def test_transform_rejects_half_precision_points():
    with pytest.raises(ValueError, match="float32 or float64"):
        transform(torch.randn(3, 3, dtype=torch.float16), torch.eye(4))


def test_transform_gradcheck():
    points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    matrix = torch.eye(4, dtype=torch.float64).requires_grad_()

    assert torch.autograd.gradcheck(transform, (points, matrix))


def test_packed_transform_gradcheck():
    points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 1, 3])
    matrices = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()

    def apply(p, t):
        return transform(PointCloud(p, offsets), t).points

    assert torch.autograd.gradcheck(apply, (points, matrices))
