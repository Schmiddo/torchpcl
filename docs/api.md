# API Reference

All public objects are exported from `torchpcl`. Native backend modules are
implementation details and are intentionally excluded from this interface.

## Data

### `PointCloud(points, offsets, normals=None, features=None)`

Packed ragged point-cloud batch. Points are `(P, 3)` float32 or float64;
offsets are `(B + 1,)` int64 on the same device. Optional normals have shape
`(P, 3)`, while features may have shape `(P, ...)`.

Construction and conversion helpers:

- `PointCloud.from_points(points, normals=None, features=None)`
- `PointCloud.from_padded(points, lengths, normals=None, features=None)`
- `cloud.to_padded(pad_value=0)`
- `cloud.to(device=None, dtype=None, non_blocking=False, copy=False)`
- `cloud.clone()`

### `PointCloudPyramid`

Ordered tuple of packed clouds and matching voxel sizes. Build one with
`build_pyramid` or pass a previously built pyramid to `multiscale_icp`.

## Geometry

### `transform(cloud, transforms)`

Applies one `(4, 4)` transform to a tensor, or `(B, 4, 4)` transforms to a
packed cloud. Attached normals are rotated; features are shared.

### `voxelize(cloud, voxel_size)`

Returns `Voxelization(cloud, coordinates, point_to_voxel, counts)`. Coordinates
have columns `(batch, x, y, z)`. `Voxelization.reduce(values, reduction)`
supports `mean`, `sum`, `min`, `max`, and `first`.

### `estimate_normals(...)`

```text
estimate_normals(cloud, radius=None, *, k=30, viewpoint=None,
                 orient="none", include_self=True,
                 return_curvature=False, index=None) -> NormalResult
```

Invalid normals are zero and identified by `NormalResult.valid`. Orientation is
`"none"` or `"viewpoint"`. This operation does not provide gradients.

## Search

### `NeighborIndex(reference, algorithm="auto")`

Reusable exact index with `knn`, `radius`, and `hybrid` methods. `algorithm` is
`"auto"`, `"bvh"`, or `"bruteforce"`. Packed references use brute force;
explicit BVH currently requires one cloud.

One-shot equivalents:

- `knn(reference, queries, k, algorithm="auto")`
- `radius_neighbors(reference, queries, radius, max_neighbors=64, algorithm="auto")`
- `hybrid_neighbors(reference, queries, radius, k, algorithm="auto")`

Every call returns `Neighbors(indices, distances2, valid)`.

## Metrics

### `chamfer_distance(...)`

```text
chamfer_distance(source, target, *, squared=True, directional="both",
                 combine="mean", point_reduction="mean",
                 reduction="mean") -> Tensor
```

`directional` is `"both"`, `"source_to_target"`, or `"target_to_source"`.
`reduction="none"` returns one value per packed batch entry.

### `fscore(prediction, reference, threshold)`

Returns tensor-valued precision, recall, and F-score. Threshold may be scalar
or one-dimensional, producing `(B,)` or `(B, T)` packed outputs.

### `point_cloud_metrics(prediction, reference, threshold)`

Returns accuracy, completion, unsquared Chamfer distance, precision, recall,
and F-score. Accuracy is prediction-to-reference; completion is the reverse.

## Registration

### `icp(...)`

```text
icp(source, target, max_distance, *, init=None,
    method="point_to_point", target_normals=None, max_iterations=30,
    relative_fitness=1e-6, relative_rmse=1e-6,
    robust_kernel=None, robust_delta=1.0, index=None) -> ICPResult
```

Methods are `"point_to_point"` and `"point_to_plane"`. The optional robust
kernel is `"huber"`. `ICPResult` contains transforms, convergence flags,
iteration counts, fitness, and inlier RMSE; it never contains correspondences.

### `evaluate_registration(source, target, max_distance, transforms=None, index=None)`

Returns `RegistrationMetrics(transforms, fitness, inlier_rmse)` without ICP
updates.

### `build_pyramid(...)`

```text
build_pyramid(cloud, voxel_sizes, *, normal_mode="none", normal_k=30,
              normal_radius_factor=2.5) -> PointCloudPyramid
```

`normal_mode` is `"none"`, `"reduce"`, or `"estimate"`.

### `multiscale_icp(...)`

```text
multiscale_icp(source, target, scales, *, init=None,
               method="point_to_point", target_normals=None,
               normal_k=30, normal_radius_factor=2.5,
               relative_fitness=1e-6, relative_rmse=1e-6,
               robust_kernel=None, robust_delta=1.0) -> ICPResult
```

Each `ICPScale` defines voxel size, maximum correspondence distance, and
iteration budget. Transforms flow from coarse to fine. Iterations are summed
across levels; convergence and error metrics describe the final level.
