# Migration to 0.2

Version 0.2 intentionally replaces the initial Open3D-shaped API. Compatibility
wrappers are not retained.

## Point Clouds and Search

- Replace separate `NearestNeighborSearch` and `BatchedNearestNeighborSearch`
  usage with `NeighborIndex`.
- Represent ragged data with `PointCloud(points, offsets)`.
- Search results now include `indices`, exact `distances2`, and `valid`.
- Packed indices are global, not local to a batch entry.

## Voxelization and Normals

- Replace `voxel_downsample(points, size)` with
  `voxelize(points, size).cloud.points` for tensors or
  `voxelize(cloud, size).cloud` for packed input.
- `voxelize` also exposes coordinates, counts, point-to-voxel mapping, and
  reusable feature reductions.
- `estimate_normals` returns `NormalResult`; use `.normals` and `.valid`.
- The compatibility module `torchpcl.preprocess` was removed.

## Metrics

- Replace `chamfer_loss` with `chamfer_distance`.
- Convert padded inputs explicitly with `PointCloud.from_padded`.
- Metric results are tensors on the input device rather than Python floats.
- `point_cloud_metrics` preserves accuracy/completion direction conventions but
  returns tensor-valued fields.

## ICP

- Replace estimator objects with `method="point_to_point"` or
  `method="point_to_plane"`.
- Replace `ICPConvergenceCriteria` with direct `max_iterations`,
  `relative_fitness`, and `relative_rmse` arguments.
- Results are always batched. Use `result.transforms[0]` for an unbatched call.
- `transformation` became `transforms`, and `num_iterations` became
  `iterations`.
- Correspondences are no longer returned.
- `evaluate_registration` returns `RegistrationMetrics`, not `ICPResult`.

Multi-scale registration uses `ICPScale` and `multiscale_icp`. Reuse
preprocessing with `build_pyramid` and pass `PointCloudPyramid` objects directly
to subsequent calls.

## Behavior Changes

- float32 ICP now returns float32 transforms instead of forcing float64.
- Failed ICP entries retain their last valid transform.
- Empty cloud pairs are rejected by metrics and registration.
- CPU and CUDA behavior is compared by distances and transforms rather than
  exact tie-broken neighbor indices.
