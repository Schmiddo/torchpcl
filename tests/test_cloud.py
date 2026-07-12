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


def test_from_padded_rejects_boolean_lengths():
    with pytest.raises(ValueError, match="integer dtype"):
        tp.PointCloud.from_padded(
            torch.randn(2, 3, 3), torch.tensor([True, False])
        )
