import numpy as np
import pytest
import torch

plyfile = pytest.importorskip("plyfile")

import torchpcl as tp
import torchpcl.io


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ply_round_trip_preserves_points_and_normals(tmp_path, dtype):
    points = torch.tensor([[1.25, -2.5, 3.75], [4.5, 5.25, -6.0]], dtype=dtype)
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.25, -0.5, 0.75]], dtype=dtype)
    cloud = tp.PointCloud.from_points(points, normals=normals)
    path = tmp_path / "cloud.ply"

    torchpcl.io.save(path, cloud)
    restored = torchpcl.io.load(path, dtype=dtype)

    assert restored.batch_size == 1
    assert restored.dtype == dtype
    assert torch.equal(restored.points, points)
    assert torch.equal(restored.normals, normals)
    assert restored.features is None


def test_top_level_ply_functions_accept_a_tensor(tmp_path):
    points = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    path = tmp_path / "points.ply"

    tp.save_ply(path, points)
    restored = tp.load_ply(path)

    assert torch.equal(restored.points, points)
    assert restored.normals is None


def test_load_ignores_unrelated_vertex_properties_and_elements(tmp_path):
    vertices = np.array(
        [(1.0, 2.0, 3.0, 255), (4.0, 5.0, 6.0, 128)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1")],
    )
    faces = np.array([([0, 1, 1],)], dtype=[("vertex_indices", "i4", (3,))])
    path = tmp_path / "with_extras.ply"
    plyfile.PlyData(
        [
            plyfile.PlyElement.describe(vertices, "vertex"),
            plyfile.PlyElement.describe(faces, "face"),
        ]
    ).write(path)

    cloud = torchpcl.io.load(path)

    assert torch.equal(
        cloud.points,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_feature_columns_round_trip(tmp_path):
    columns = ["red", "intensity"]
    cloud = tp.PointCloud.from_points(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        features=torch.tensor([[0.0, 1.25], [1.0, 42.0]], dtype=torch.float32),
    )
    path = tmp_path / "features.ply"

    torchpcl.io.save(path, cloud, feature_names=columns)
    restored = torchpcl.io.load(path, feature_names=columns)
    assert restored.features.dtype == torch.float32
    assert torch.equal(restored.features, cloud.features)


@pytest.mark.parametrize(
    ("vertex_dtype", "message"),
    [
        ([("x", "f4"), ("y", "f4")], "missing properties"),
        (
            [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4")],
            "normals are incomplete",
        ),
    ],
)
def test_load_rejects_missing_geometry_properties(tmp_path, vertex_dtype, message):
    path = tmp_path / "invalid.ply"
    vertices = np.zeros(1, dtype=vertex_dtype)
    plyfile.PlyData([plyfile.PlyElement.describe(vertices, "vertex")]).write(path)

    with pytest.raises(ValueError, match=message):
        torchpcl.io.load(path)


def test_load_rejects_ply_without_vertices(tmp_path):
    path = tmp_path / "no_vertices.ply"
    edges = np.zeros(1, dtype=[("first", "i4"), ("second", "i4")])
    plyfile.PlyData([plyfile.PlyElement.describe(edges, "edge")]).write(path)

    with pytest.raises(ValueError, match="vertex element"):
        torchpcl.io.load(path)


def test_save_rejects_batches_and_generic_features(tmp_path):
    batched = tp.PointCloud(
        torch.randn(2, 3), torch.tensor([0, 1, 2], dtype=torch.int64)
    )
    featured = tp.PointCloud.from_points(torch.randn(2, 3), features=torch.randn(2, 4))

    with pytest.raises(ValueError, match="exactly one"):
        torchpcl.io.save(tmp_path / "batch.ply", batched)


def test_ply_io_validates_feature_columns(tmp_path):
    path = tmp_path / "cloud.ply"
    cloud = tp.PointCloud.from_points(torch.randn(2, 3), features=torch.randn(2, 2))

    with pytest.raises(ValueError, match="does not match"):
        torchpcl.io.save(path, cloud, feature_names=["value"])
