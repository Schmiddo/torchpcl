from types import SimpleNamespace

import numpy as np
import torch

import torchpcl as tp
from torchpcl.tools.register import _combine_with_colors, main
from torchpcl.transforms import pose_to_matrix, transform_points


def test_register_tool_runs_global_icp_and_saves_combined_ply(tmp_path, capsys):
    generator = torch.Generator().manual_seed(7)
    points = torch.randn(48, 3, generator=generator, dtype=torch.float64)
    points[:, 0] *= 2.0
    points[:, 1] *= 0.7
    transformation = pose_to_matrix(
        torch.tensor([0.08, -0.12, 0.05, 0.4, -0.2, 0.3])
    )
    source_path = tmp_path / "source.ply"
    target_path = tmp_path / "target.ply"
    output_path = tmp_path / "results" / "combined.ply"
    source_colors = torch.tensor([220, 40, 30], dtype=torch.uint8).expand(48, -1)
    target_colors = torch.tensor([20, 80, 230], dtype=torch.uint8).expand(48, -1)
    color_names = ("red", "green", "blue")
    tp.save_ply(
        source_path,
        tp.PointCloud.from_points(points, features=source_colors),
        feature_names=color_names,
    )
    tp.save_ply(
        target_path,
        tp.PointCloud.from_points(
            transform_points(points, transformation),
            features=target_colors,
        ),
        feature_names=color_names,
    )

    exit_code = main(
        [
            str(source_path),
            str(target_path),
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--voxel-size",
            "0.01",
            "--normal-radius",
            "10",
            "--feature-radius",
            "10",
            "--fgr-distance",
            "0.05",
            "--icp-distance",
            "0.05",
            "--max-neighbors",
            "48",
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    combined = tp.load_ply(
        output_path,
        dtype=torch.float64,
        feature_names=color_names,
    )
    assert exit_code == 0
    assert "Global registration: success=True" in output
    assert "Final transformation (source -> target):" in output
    assert "Inlier RMSE (registration error):" in output
    assert combined.points.shape == (96, 3)
    assert torch.equal(combined.features[:48], source_colors)
    assert torch.equal(combined.features[48:], target_colors)
    with np.load(output_path.with_suffix(".npz")) as masks:
        assert masks.files == ["source", "target"]
        assert masks["source"].dtype == np.bool_
        assert masks["source"].sum() == 48
        assert masks["target"].sum() == 48
        assert not np.any(masks["source"] & masks["target"])


def test_combined_cloud_assigns_distinct_solid_colors_to_uncolored_inputs():
    source = tp.PointCloud.from_points(torch.randn(2, 3))
    target = tp.PointCloud.from_points(torch.randn(3, 3))

    combined = _combine_with_colors(source, target)

    assert torch.equal(combined.features[:2], combined.features[:1].expand(2, -1))
    assert torch.equal(combined.features[2:], combined.features[2:3].expand(3, -1))
    assert not torch.equal(combined.features[0], combined.features[2])


def test_register_tool_uses_all_pairs_then_icp_spanning_tree(
    tmp_path, capsys, monkeypatch
):
    directories = [tmp_path / "one", tmp_path / "two", tmp_path / "three"]
    paths = [directories[0] / "scan.ply", directories[1] / "scan.ply"]
    paths.append(directories[2] / "other.ply")
    color_names = ("red", "green", "blue")
    for index, path in enumerate(paths):
        path.parent.mkdir()
        points = torch.arange(36, dtype=torch.float32).reshape(12, 3)
        tp.save_ply(path, tp.PointCloud.from_points(points + index))

    monkeypatch.setattr(
        "torchpcl.tools.register._describe",
        lambda cloud, **_: cloud,
    )
    fgr_pairs = []
    scores = {(0, 1): 0.9, (0, 2): 0.8, (1, 2): 0.1}

    def fake_fgr(source, target, *, options):
        source_index = round(float(source.points[0, 0].item()))
        target_index = round(float(target.points[0, 0].item()))
        fgr_pairs.append((source_index, target_index))
        return SimpleNamespace(
            transforms=torch.eye(4).unsqueeze(0),
            success=torch.tensor([True]),
            correspondence_counts=torch.tensor([12]),
            fitness=torch.tensor([scores[source_index, target_index]]),
            inlier_rmse=torch.tensor([0.01]),
        )

    icp_pairs = []

    def fake_icp(source, target, levels, *, init, options):
        source_index = round(float(source.points[0, 0].item()))
        target_index = round(float(target.points[0, 0].item()))
        icp_pairs.append((source_index, target_index))
        return SimpleNamespace(
            transforms=init.unsqueeze(0) if init.ndim == 2 else init,
            fitness=torch.tensor([1.0]),
            inlier_rmse=torch.tensor([0.0]),
        )

    monkeypatch.setattr(tp, "fast_global_registration", fake_fgr)
    monkeypatch.setattr(tp, "icp", fake_icp)
    output_path = tmp_path / "combined.ply"

    assert main([*(str(path) for path in paths), "-o", str(output_path)]) == 0

    assert fgr_pairs == [(0, 1), (0, 2), (1, 2)]
    assert icp_pairs == [(0, 1), (0, 2)]
    combined = tp.load_ply(output_path, feature_names=color_names)
    assert combined.points.shape == (36, 3)
    with np.load(output_path.with_suffix(".npz")) as masks:
        assert masks.files == ["scan", "scan-1", "other"]
        assert [masks[key].sum() for key in masks.files] == [12, 12, 12]
        assert np.all(sum((masks[key] for key in masks.files)) == 1)
    assert "Final transformations" in capsys.readouterr().out
