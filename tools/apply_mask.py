#!/usr/bin/env python3
"""Apply a binary NumPy mask to the vertices of a PLY point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_mask(path: Path, num_points: int) -> np.ndarray:
    """Load and validate a one-dimensional binary point mask."""
    mask = np.load(path, allow_pickle=False)
    if mask.ndim != 1:
        raise ValueError(f"Mask must be one-dimensional, but has shape {mask.shape}")
    if len(mask) != num_points:
        raise ValueError(
            f"Mask contains {len(mask):,} entries, but the point cloud contains "
            f"{num_points:,} points"
        )
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Mask must contain only binary values (0 or 1)")
    return mask.astype(bool, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_cloud", type=Path, help="input point cloud in PLY format")
    parser.add_argument("mask", type=Path, help="binary point mask in NPY format")
    parser.add_argument("output_cloud", type=Path, help="output point cloud in PLY format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cloud = PlyData.read(args.input_cloud)

    try:
        vertices = cloud["vertex"]
    except KeyError as error:
        raise ValueError(
            f"Input PLY file {args.input_cloud} has no vertex element"
        ) from error

    mask = load_mask(args.mask, len(vertices.data))
    vertices.data = vertices.data[~mask]
    cloud.write(args.output_cloud)

    print(
        f"Kept {mask.sum():,} of {len(mask):,} points from {args.input_cloud} "
        f"in {args.output_cloud}"
    )


if __name__ == "__main__":
    main()
