import math

import pytest
import torch

import torchpcl as tp


def test_radius_and_statistical_outliers_on_cluster_with_boundary(search_device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=search_device,
    )

    radius = tp.radius_outliers(points, radius=1.0, k=2)
    statistical = tp.statistical_outliers(points, k=2, std_ratio=1.0)

    assert radius.tolist() == [False, False, False, True]
    assert statistical.tolist() == [False, False, False, True]
