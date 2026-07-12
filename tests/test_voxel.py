import torch

import torchpcl as tp


def test_voxelize_packed_batches_and_signed_coordinates(device):
    points = torch.tensor(
        [
            [-3_000_000.2, 0.0, 0.0],
            [-3_000_000.1, 0.0, 0.0],
            [-3_000_000.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    cloud = tp.PointCloud(points, torch.tensor([0, 2, 4], device=device))

    result = tp.voxelize(cloud, 1.0)

    assert result.cloud.offsets.tolist() == [0, 1, 3]
    assert result.coordinates.tolist() == [
        [0, -3_000_001, 0, 0],
        [1, -3_000_001, 0, 0],
        [1, 1, 0, 0],
    ]
    assert result.point_to_voxel.tolist() == [0, 0, 1, 2]
    assert result.counts.tolist() == [2, 1, 1]


def test_voxel_reductions():
    points = torch.tensor(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.1, 0.0, 0.0]]
    )
    values = torch.tensor([[2.0, 5.0], [4.0, 3.0], [8.0, 1.0]])
    result = tp.voxelize(points, 1.0)

    torch.testing.assert_close(
        result.reduce(values, "sum"), torch.tensor([[6.0, 8.0], [8.0, 1.0]])
    )
    torch.testing.assert_close(
        result.reduce(values, "mean"), torch.tensor([[3.0, 4.0], [8.0, 1.0]])
    )
    torch.testing.assert_close(
        result.reduce(values, "min"), torch.tensor([[2.0, 3.0], [8.0, 1.0]])
    )
    torch.testing.assert_close(
        result.reduce(values, "max"), torch.tensor([[4.0, 5.0], [8.0, 1.0]])
    )
    torch.testing.assert_close(
        result.reduce(values, "first"), torch.tensor([[2.0, 5.0], [8.0, 1.0]])
    )


def test_voxel_mean_and_sum_gradients():
    points = torch.tensor(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=torch.float64,
    )
    values = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    result = tp.voxelize(points, 1.0)

    mean_grad = torch.autograd.grad(result.reduce(values, "mean").sum(), values)[0]
    sum_grad = torch.autograd.grad(result.reduce(values, "sum").sum(), values)[0]

    torch.testing.assert_close(
        mean_grad,
        torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float64
        ),
    )
    torch.testing.assert_close(sum_grad, torch.ones_like(values))


def test_voxelize_empty_batch_entries(device):
    cloud = tp.PointCloud(
        torch.tensor([[0.1, 0.0, 0.0]], device=device),
        torch.tensor([0, 0, 1, 1], device=device),
    )

    result = tp.voxelize(cloud, 1.0)

    assert result.cloud.offsets.tolist() == [0, 0, 1, 1]


def test_voxel_downsample_preserves_input_style():
    points = torch.randn(5, 3)
    assert isinstance(tp.voxel_downsample(points, 1.0), torch.Tensor)
    cloud = tp.PointCloud.from_points(points)
    assert isinstance(tp.voxel_downsample(cloud, 1.0), tp.PointCloud)
