# Migration to 0.2

Version 0.2 intentionally replaces the initial Open3D-shaped API. Compatibility
wrappers are not retained.

## Point Clouds and Search

- Replace separate `NearestNeighborSearch` and `BatchedNearestNeighborSearch`
  usage with `NeighborIndex`.
- Represent ragged data with `PointCloud(points, offsets)`.
- Search results now include `indices`, `distances2` recomputed in the input
  dtype, and `valid`. Candidate selection uses float32 coordinates.
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
- Equal-length `(B, N, 3)` tensors are accepted directly and interpreted as
  `B` contiguous clouds.
- Metric results are tensors on the input device rather than Python floats.
- F-score accepts one scalar threshold per call. F-score and combined metric
  fields always retain their `(B,)` batch dimension.
- `point_cloud_metrics` preserves accuracy/completion direction conventions but
  returns tensor-valued fields.

## ICP

- Replace separate single- and multi-scale entry points with `icp` and a
  nonempty sequence of `ICPLevel` objects.
- Use one level with `voxel_size=None` for single-scale registration and
  several levels for multi-scale registration.
- Replace method strings with `PointToPoint()` or `PointToPlane()` in
  `ICPOptions`.
- Replace direct convergence arguments with `ConvergenceCriteria` and replace
  robust-kernel strings with `HuberLoss`.
- Attach point-to-plane normals to the target `PointCloud`. ICP does not
  estimate them automatically.
- Results are always batched. Use `result.transforms[0]` for an unbatched call.
- Total iterations and final-level metrics are available directly; inspect
  `result.level_results` for per-level diagnostics.
- Correspondences are no longer returned.
- `evaluate_registration` returns `RegistrationMetrics`, not `ICPResult`.
- Reusable registration pyramids and public neighbor-index parameters were
  removed.

## Behavior Changes

- float32 ICP now returns float32 transforms instead of forcing float64.
- Failed ICP entries retain their last valid transform.
- Empty cloud pairs are rejected by metrics and registration.
- CPU and CUDA behavior is compared by distances and transforms rather than
  tie-broken neighbor indices.
