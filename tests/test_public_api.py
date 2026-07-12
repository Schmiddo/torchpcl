import torchpcl


def test_public_api_is_intentionally_small():
    assert set(torchpcl.__all__) == {
        "FScoreResult",
        "ICPResult",
        "ICPScale",
        "NeighborIndex",
        "Neighbors",
        "NormalResult",
        "PointCloud",
        "PointCloudMetrics",
        "PointCloudPyramid",
        "RegistrationMetrics",
        "Voxelization",
        "build_pyramid",
        "chamfer_distance",
        "estimate_normals",
        "evaluate_registration",
        "fscore",
        "hybrid_neighbors",
        "icp",
        "knn",
        "multiscale_icp",
        "point_cloud_metrics",
        "radius_neighbors",
        "transform",
        "voxelize",
        "__version__",
    }


def test_removed_compatibility_symbols_are_not_exported():
    for name in (
        "BatchedNearestNeighborSearch",
        "ICPConvergenceCriteria",
        "NearestNeighborSearch",
        "PointToPlane",
        "PointToPoint",
        "chamfer_loss",
        "voxel_downsample",
    ):
        assert not hasattr(torchpcl, name)
