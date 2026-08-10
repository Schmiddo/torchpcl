import pytest
import torch

import torchpcl as tp


def test_from_points_properties_and_storage_sharing(device):
    points = torch.randn(4, 3, dtype=torch.float64, device=device)
    normals = torch.randn_like(points)
    features = torch.arange(8, device=device).reshape(4, 2)

    cloud = tp.PointCloud.from_points(
        points, normals=normals, features=features
    )

    assert cloud.batch_size == 1
    assert cloud.lengths.tolist() == [4]
    assert cloud.device == points.device
    assert cloud.dtype == torch.float64
    assert cloud.points is points
    assert cloud.normals is normals
    assert cloud.features is features


def test_padded_round_trip_with_empty_batch_entry(device):
    padded = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [99.0, 99.0, 99.0]],
            [[99.0, 99.0, 99.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            [[7.0, 8.0, 9.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    lengths = torch.tensor([2, 0, 1], device=device)
    normals = torch.ones_like(padded)
    features = torch.arange(18, device=device).reshape(3, 3, 2)

    cloud = tp.PointCloud.from_padded(
        padded, lengths, normals=normals, features=features
    )
    restored, restored_lengths = cloud.to_padded(pad_value=-1.0)

    assert cloud.offsets.tolist() == [0, 2, 2, 3]
    assert cloud.normals.shape == (3, 3)
    assert cloud.features.shape == (3, 2)
    assert torch.equal(restored_lengths, lengths.to(torch.int64))
    for batch, length in enumerate(lengths.tolist()):
        assert torch.equal(restored[batch, :length], padded[batch, :length])
        assert torch.all(restored[batch, length:] == -1)


def test_to_casts_geometry_but_preserves_feature_dtype(device):
    points = torch.randn(3, 3, dtype=torch.float64, device=device)
    normals = torch.randn_like(points)
    features = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
    cloud = tp.PointCloud.from_points(
        points, normals=normals, features=features
    )

    converted = cloud.to(torch.float32)

    assert converted.dtype == torch.float32
    assert converted.normals.dtype == torch.float32
    assert converted.features.dtype == torch.int64
    assert converted.offsets.data_ptr() == cloud.offsets.data_ptr()


def test_clone_has_independent_storage(device):
    cloud = tp.PointCloud.from_points(
        torch.randn(3, 3, device=device),
        features=torch.randn(3, 2, device=device),
    )

    cloned = cloud.clone()

    assert torch.equal(cloned.points, cloud.points)
    assert cloned.points.data_ptr() != cloud.points.data_ptr()
    assert cloned.offsets.data_ptr() != cloud.offsets.data_ptr()
    assert cloned.features.data_ptr() != cloud.features.data_ptr()


@pytest.mark.parametrize(
    "offsets",
    [
        torch.tensor([1, 3]),
        torch.tensor([0, 4]),
        torch.tensor([0, 2, 1, 3]),
    ],
)
def test_rejects_invalid_offsets(offsets):
    points = torch.randn(3, 3)
    with pytest.raises(ValueError, match="offsets"):
        tp.PointCloud(points, offsets)


def test_rejects_invalid_attributes():
    points = torch.randn(3, 3)
    offsets = torch.tensor([0, 3])
    with pytest.raises(ValueError, match="normals"):
        tp.PointCloud(points, offsets, normals=torch.randn(2, 3))
    with pytest.raises(ValueError, match="features"):
        tp.PointCloud(points, offsets, features=torch.randn(2, 4))


def test_as_point_cloud_normalizes_dense_tensor_without_copy(device):
    points = torch.randn(2, 4, 3, device=device)

    cloud = tp.as_point_cloud(points)

    assert cloud.offsets.tolist() == [0, 4, 8]
    assert cloud.points.shape == (8, 3)
    assert cloud.points.data_ptr() == points.data_ptr()


def test_as_point_cloud_rejects_other_tensor_layouts():
    with pytest.raises(ValueError, match="from_padded"):
        tp.as_point_cloud(torch.randn(2, 3, 4))


def _marked_cloud(values, lengths, device):
    markers = torch.tensor(values, dtype=torch.float64, device=device)
    points = markers[:, None].expand(-1, 3).clone()
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    return tp.PointCloud(
        points,
        offsets,
        normals=points + 10.0,
        features=markers.to(torch.int64)[:, None],
    )


def test_cat_concatenates_corresponding_ragged_batches(device):
    first = _marked_cloud([1, 2, 3], [2, 0, 1], device)
    second = _marked_cloud([4, 5, 6], [1, 2, 0], device)

    result = tp.cat([first, second])

    expected = torch.tensor([1, 2, 4, 5, 6, 3], device=device)
    assert result.offsets.tolist() == [0, 3, 5, 6]
    assert torch.equal(result.points[:, 0], expected.to(torch.float64))
    assert torch.equal(result.normals[:, 0], expected.to(torch.float64) + 10.0)
    assert torch.equal(result.features[:, 0], expected)


def test_cat_batch_appends_complete_batches(device):
    first = _marked_cloud([1, 2, 3], [2, 0, 1], device)
    second = _marked_cloud([4, 5, 6], [1, 2, 0], device)

    result = tp.cat_batch([first, second])

    assert result.offsets.tolist() == [0, 2, 2, 3, 4, 6, 6]
    assert torch.equal(result.points, torch.cat([first.points, second.points]))
    assert torch.equal(result.normals, torch.cat([first.normals, second.normals]))
    assert torch.equal(
        result.features, torch.cat([first.features, second.features])
    )


@pytest.mark.parametrize("operation", [tp.cat, tp.cat_batch])
def test_cat_preserves_empty_clouds_and_single_input(operation, device):
    empty = tp.PointCloud(
        torch.empty(0, 3, device=device),
        torch.tensor([0], dtype=torch.int64, device=device),
        features=torch.empty(0, 2, dtype=torch.int64, device=device),
    )

    assert operation([empty]) is empty
    result = operation([empty, empty])
    assert result.batch_size == 0
    assert result.points.shape == (0, 3)
    assert result.features.shape == (0, 2)
    assert result.offsets.tolist() == [0]


@pytest.mark.parametrize("operation", [tp.cat, tp.cat_batch])
def test_cat_rejects_empty_lists(operation):
    with pytest.raises(ValueError, match="not be empty"):
        operation([])


def test_cat_requires_equal_batch_sizes():
    one_batch = tp.PointCloud.from_points(torch.randn(2, 3))
    two_batches = tp.PointCloud(
        torch.randn(2, 3), torch.tensor([0, 1, 2], dtype=torch.int64)
    )

    with pytest.raises(ValueError, match="same batch size"):
        tp.cat([one_batch, two_batches])


@pytest.mark.parametrize("operation", [tp.cat, tp.cat_batch])
def test_cat_propagates_gradients_to_every_input(operation, device):
    first_points = torch.randn(3, 3, device=device, requires_grad=True)
    second_points = torch.randn(3, 3, device=device, requires_grad=True)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    first = tp.PointCloud(first_points, offsets)
    second = tp.PointCloud(second_points, offsets)

    operation([first, second]).points.sum().backward()

    assert torch.equal(first_points.grad, torch.ones_like(first_points))
    assert torch.equal(second_points.grad, torch.ones_like(second_points))
