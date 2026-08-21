import torch

import torchpcl as tp
from torchpcl.transforms import pose_to_matrix, transform_points


def _asymmetric_points(device, count=32):
    generator = torch.Generator(device="cpu").manual_seed(3)
    points = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    points[:, 0] *= 2.0
    points[:, 1] *= 0.7
    return points.to(device)


def _known_transform(device):
    pose = torch.tensor(
        [0.08, -0.12, 0.05, 0.4, -0.2, 0.3],
        dtype=torch.float64,
        device=device,
    )
    return pose_to_matrix(pose)


def test_fgr_recovers_transform_from_correspondences(search_device):
    points = _asymmetric_points(search_device)
    expected = _known_transform(search_device)
    source = tp.PointCloud.from_points(points)
    target = tp.PointCloud.from_points(transform_points(points, expected))
    rows = torch.arange(points.shape[0], device=search_device)
    correspondences = tp.CorrespondenceSet(rows, rows.clone())

    result = tp.fast_global_registration_from_correspondences(
        source,
        target,
        correspondences,
        options=tp.FGROptions(
            max_correspondence_distance=0.05,
            tuple_scale=None,
        ),
    )

    assert result.success.tolist() == [True]
    assert result.correspondence_counts.tolist() == [points.shape[0]]
    assert torch.allclose(result.transforms[0], expected, atol=1e-8)
    assert result.fitness.item() == 1.0


def test_fpfh_to_fgr_workflow_improves_alignment(search_device):
    generator = torch.Generator(device="cpu").manual_seed(123)
    shared = torch.randn(40, 3, generator=generator, dtype=torch.float64)
    shared[:, 0] *= 2.0
    source_only = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    target_only = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    source_center = torch.tensor([12.0, 0.0, 0.0], dtype=torch.float64)
    target_center = torch.tensor([-12.0, 0.0, 0.0], dtype=torch.float64)
    source_only += source_center
    target_only += target_center
    source_points = torch.cat((shared, source_only)).to(search_device)
    target_points = torch.cat((shared, target_only)).to(search_device)
    shared_normals = shared / shared.norm(dim=1, keepdim=True)
    source_only_normals = source_only - source_center
    source_only_normals /= source_only_normals.norm(dim=1, keepdim=True)
    target_only_normals = target_only - target_center
    target_only_normals /= target_only_normals.norm(dim=1, keepdim=True)
    source_normals = torch.cat((shared_normals, source_only_normals)).to(
        search_device
    )
    target_normals = torch.cat((shared_normals, target_only_normals)).to(
        search_device
    )
    expected = _known_transform(search_device)
    source = tp.PointCloud.from_points(source_points, normals=source_normals)
    target = tp.transform(
        tp.PointCloud.from_points(target_points, normals=target_normals),
        expected,
    )
    assert isinstance(target, tp.PointCloud)

    source_fpfh = tp.fpfh(source, 5.0, max_neighbors=40)
    target_fpfh = tp.fpfh(target, 5.0, max_neighbors=40)
    source = source.with_features(source_fpfh.descriptors)
    target = target.with_features(target_fpfh.descriptors)
    options = tp.FGROptions(max_correspondence_distance=0.05)
    identity = tp.evaluate_registration(
        source, target, options.max_correspondence_distance
    )

    result = tp.fast_global_registration(source, target, options=options)

    assert result.success.tolist() == [True]
    assert result.correspondence_counts.item() >= shared.shape[0]
    assert result.fitness.item() > identity.fitness.item()
    assert result.fitness.item() > 0.7
    assert torch.allclose(result.transforms[0], expected, atol=1e-6)
