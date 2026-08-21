"""Minimal normals -> FPFH -> FGR -> ICP registration workflow."""

import torch

import torchpcl as tp
from torchpcl.transforms import pose_to_matrix, transform_points


def describe(
    cloud: tp.PointCloud,
    viewpoint: torch.Tensor,
) -> tp.PointCloud:
    normals = tp.estimate_normals(cloud, radius=10.0, k=30)
    cloud = cloud.with_normals(normals.normals)
    cloud = tp.orient_normals_toward_viewpoint(cloud, viewpoint)
    features = tp.fpfh(cloud, radius=10.0, max_neighbors=48)
    return cloud.select_points(features.valid).with_features(
        features.descriptors[features.valid]
    )


def main() -> None:
    generator = torch.Generator().manual_seed(7)
    points = torch.randn(48, 3, generator=generator, dtype=torch.float64)
    points[:, 0] *= 2.0
    points[:, 1] *= 0.7
    expected = pose_to_matrix(
        torch.tensor([0.08, -0.12, 0.05, 0.4, -0.2, 0.3])
    )
    target_points = transform_points(points, expected)

    viewpoint = torch.tensor([0.0, 0.0, 8.0], dtype=torch.float64)
    target_viewpoint = transform_points(viewpoint[None], expected)[0]
    source = describe(tp.PointCloud.from_points(points), viewpoint)
    target = describe(tp.PointCloud.from_points(target_points), target_viewpoint)

    coarse = tp.fast_global_registration(
        source,
        target,
        options=tp.FGROptions(max_correspondence_distance=0.05),
    )
    refined = tp.icp(
        source,
        target,
        [tp.ICPLevel(max_correspondence_distance=0.05, max_iterations=20)],
        init=coarse.transforms,
    )

    print("FGR transform:\n", coarse.transforms[0])
    print("FGR fitness/RMSE:", coarse.fitness.item(), coarse.inlier_rmse.item())
    print("ICP fitness/RMSE:", refined.fitness.item(), refined.inlier_rmse.item())


if __name__ == "__main__":
    main()
