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


def test_combined_cloud_assigns_red_and_blue_to_uncolored_inputs():
    source = tp.PointCloud.from_points(torch.randn(2, 3))
    target = tp.PointCloud.from_points(torch.randn(3, 3))

    combined = _combine_with_colors(source, target)

    assert torch.equal(
        combined.features[:2],
        torch.tensor([[255, 0, 0], [255, 0, 0]], dtype=torch.uint8),
    )
    assert torch.equal(
        combined.features[2:],
        torch.tensor(
            [[0, 0, 255], [0, 0, 255], [0, 0, 255]], dtype=torch.uint8
        ),
    )
