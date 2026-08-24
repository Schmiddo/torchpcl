"""PLY input and output for single point clouds."""

from __future__ import annotations

from collections.abc import Sequence
import math
from os import PathLike
from typing import Any

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .cloud import PointCloud, PointCloudLike, as_point_cloud


_POINT_FIELDS = ("x", "y", "z")
_NORMAL_FIELDS = ("nx", "ny", "nz")

_DTYPES = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.uint8: np.uint8,
    torch.uint16: np.uint16,
    torch.uint32: np.uint32,
}


@torch.no_grad()
def load(
    path: str | PathLike[str],
    *,
    dtype: torch.dtype = torch.float32,
    feature_names: Sequence[str] = (),
) -> PointCloud:
    """Load one point cloud from a PLY file.

    Vertex properties ``x``, ``y``, and ``z`` are required. Normals are loaded
    when all of ``nx``, ``ny``, and ``nz`` are present. Columns named by
    ``features`` are stacked in the given order, and attached as an ``(N, F)``
    feature tensor. They must have the same dtype. Other elements and properties
    are ignored.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    ply = PlyData.read(path)

    try:
        vertices = ply["vertex"]
    except KeyError as error:
        raise ValueError("PLY file does not contain a vertex element") from error

    property_names = {prop.name for prop in vertices.properties}
    missing_coordinates = {"x", "y", "z"} - property_names
    if missing_coordinates:
        missing = ", ".join(sorted(missing_coordinates))
        raise ValueError(f"PLY vertex element is missing properties: {missing}")

    normal_names = {"nx", "ny", "nz"}
    present_normals = normal_names & property_names
    if present_normals and present_normals != normal_names:
        missing = ", ".join(sorted(normal_names - present_normals))
        raise ValueError(f"PLY vertex normals are incomplete; missing: {missing}")

    missing_features = set(feature_names) - property_names
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"PLY vertex element is missing feature properties: {missing}")

    def tensor_from_columns(names: Sequence[str], dtype=None) -> torch.Tensor:
        array = np.ascontiguousarray(
            np.column_stack([vertices[name] for name in names])
        )
        return torch.as_tensor(array, dtype=dtype)

    points = tensor_from_columns(("x", "y", "z"), dtype)
    normals = tensor_from_columns(("nx", "ny", "nz"), dtype) if present_normals else None
    features = tensor_from_columns(feature_names) if feature_names else None

    return PointCloud.from_points(points, normals=normals, features=features)


@torch.no_grad()
def save(
    path: str | PathLike[str],
    cloud: PointCloudLike,
    *,
    feature_names: Sequence[str] = (),
) -> None:
    """Save one point cloud to a binary PLY file.

    When the cloud has features, ``feature_names`` must provide a name for each
    flattened feature column in order. Feature values must use a scalar dtype
    supported by PLY.
    """
    packed = as_point_cloud(cloud, "cloud")
    if packed.batch_size != 1:
        raise ValueError(
            f"cloud must contain exactly one batch entry, got {packed.batch_size}"
        )

    flat_features = None
    if packed.features is None:
        if feature_names:
            raise ValueError("feature names were provided but cloud has no features")
    else:
        if packed.features.dtype not in _DTYPES:
            supported = ", ".join(str(dtype) for dtype in _DTYPES)
            raise ValueError(
                "features must have a PLY-compatible dtype "
                f"({supported}), got {packed.features.dtype}"
            )
        feature_count = math.prod(packed.features.shape[1:])
        flat_features = packed.features.reshape(
            packed.features.shape[0], feature_count
        )
        if len(feature_names) != feature_count:
            raise ValueError(
                f"number of feature names ({len(feature_names)}) does not match feature "
                f"shape {tuple(packed.features.shape)}"
            )

    geometry_dtype = np.float32 if packed.dtype == torch.float32 else np.float64
    vertex_dtype: list[tuple[str, Any]] = [
        (name, geometry_dtype) for name in ("x", "y", "z")
    ]
    if packed.normals is not None:
        vertex_dtype.extend((name, geometry_dtype) for name in ("nx", "ny", "nz"))
    if flat_features is not None:
        vertex_dtype.extend(
            (fname, _DTYPES[flat_features.dtype]) for fname in feature_names
        )
    vertices = np.empty(packed.points.shape[0], dtype=vertex_dtype)

    def _serialize(tensor, names):
        tensor = tensor.detach().cpu().numpy()
        for idx, name in enumerate(names):
            vertices[name] = tensor[:, idx]

    _serialize(packed.points, ("x", "y", "z"))
    if packed.normals is not None:
        _serialize(packed.normals, ("nx", "ny", "nz"))
    if flat_features is not None:
        _serialize(flat_features, feature_names)

    element = PlyElement.describe(vertices, "vertex")
    PlyData([element], byte_order="<").write(path)


load_ply = load
save_ply = save


__all__ = ["load", "load_ply", "save", "save_ply"]
