import torch

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
