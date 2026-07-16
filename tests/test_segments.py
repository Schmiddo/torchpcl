import torch

from torchpcl._segments import segment_sum


def test_segment_sum_preserves_missing_segments_and_trailing_shape(device):
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        device=device,
    )
    ids = torch.tensor([0, 2, 0], device=device)

    result = segment_sum(values, ids, 4)

    expected = torch.stack(
        [
            values[0] + values[2],
            torch.zeros_like(values[0]),
            values[1],
            torch.zeros_like(values[0]),
        ]
    )
    torch.testing.assert_close(result, expected)


def test_segment_sum_single_empty_segment(device):
    values = torch.empty((0, 3), device=device)
    ids = torch.empty(0, dtype=torch.int64, device=device)

    result = segment_sum(values, ids, 1)

    torch.testing.assert_close(result, torch.zeros((1, 3), device=device))


def test_segment_sum_gradcheck():
    values = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 1, 0, 2, 1])

    assert torch.autograd.gradcheck(lambda value: segment_sum(value, ids, 3), (values,))
