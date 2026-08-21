import pytest
import torch

import torchpcl as tp
from torchpcl.transforms import pose_to_matrix


def _descriptor_fixture(device):
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [-1.0, -0.5, 0.25],
            [0.5, -1.0, 1.5],
            [100.0, 100.0, 100.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-0.2, 0.8, 0.5],
            [0.7, -0.1, 0.7],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    return tp.PointCloud.from_points(points, normals=normals)


def test_fpfh_small_fixture_and_diagnostics(search_device):
    result = tp.fpfh(_descriptor_fixture(search_device), 10.0, max_neighbors=6)

    assert result.descriptors.shape == (7, 33)
    assert result.valid.tolist() == [True] * 6 + [False]
    assert result.neighbor_limit_reached.tolist() == [True] * 6 + [False]
    assert torch.allclose(
        result.descriptors[:6].reshape(6, 3, 11).sum(dim=2),
        torch.full((6, 3), 200.0, dtype=torch.float64, device=search_device),
    )
    assert result.descriptors[0, 5].item() == pytest.approx(84.8435, abs=1e-4)
    assert result.descriptors[0, 11].item() == pytest.approx(38.9786, abs=1e-4)
    assert torch.count_nonzero(result.descriptors[6]) == 0


def test_fpfh_is_rigid_transform_invariant(search_device):
    generator = torch.Generator(device="cpu").manual_seed(9)
    points = torch.randn(16, 3, generator=generator, dtype=torch.float64).to(
        search_device
    )
    normals = torch.randn(16, 3, generator=generator, dtype=torch.float64).to(
        search_device
    )
    cloud = tp.PointCloud.from_points(points, normals=normals)
    transformation = pose_to_matrix(
        torch.tensor(
            [0.2, -0.1, 0.15, 2.0, -1.0, 0.5],
            dtype=torch.float64,
            device=search_device,
        )
    )

    source = tp.fpfh(cloud, 10.0, max_neighbors=16)
    target = tp.fpfh(tp.transform(cloud, transformation), 10.0, max_neighbors=16)

    assert torch.equal(source.valid, target.valid)
    assert torch.allclose(source.descriptors, target.descriptors, atol=1e-10)
