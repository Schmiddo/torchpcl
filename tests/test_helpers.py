import pytest
import torch

from conftest import packed_ragged_batch


def test_packed_ragged_batch_supports_empty_entries(device):
    points, offsets = packed_ragged_batch([2, 0, 3], device, seed=7)

    assert points.shape == (5, 3)
    expected_device = torch.empty(0, device=device).device
    assert points.device == expected_device
    assert offsets.tolist() == [0, 2, 2, 5]
    assert offsets.device == expected_device


def test_packed_ragged_batch_is_device_independent(device):
    points, _ = packed_ragged_batch([1, 2], device, dtype=torch.float32, seed=9)
    expected, _ = packed_ragged_batch([1, 2], "cpu", dtype=torch.float32, seed=9)

    assert torch.equal(points.cpu(), expected)


def test_packed_ragged_batch_rejects_negative_lengths():
    with pytest.raises(ValueError, match="nonnegative"):
        packed_ragged_batch([2, -1], "cpu")
