# API Reference

All public objects are exported from `torchpcl`. Native backend modules are
implementation details and are intentionally excluded from this interface.

## Data

### `PointCloud(points, offsets, normals=None, features=None)`

Packed ragged point-cloud batch. Points are `(P, 3)` float32 or float64;
offsets are `(B + 1,)` int64 on the same device. Optional normals have shape
`(P, 3)`, while features may have shape `(P, ...)`.

Construction and conversion helpers:

- `as_point_cloud(points_or_cloud)`
- `PointCloud.from_points(points, normals=None, features=None)`
- `PointCloud.from_padded(points, lengths, normals=None, features=None)`
- `cloud.to_padded(pad_value=0)`
- `cloud.to(device=None, dtype=None, non_blocking=False, copy=False)`
- `cloud.clone()`

### `PointCloudPyramid`

Ordered tuple of packed clouds and matching voxel sizes. Build one with
`build_pyramid` or pass a previously built pyramid to `multiscale_icp`.

## Geometry

Public geometry inputs accept `(N, 3)` tensors, equal-length `(B, N, 3)` tensor
batches, or `PointCloud`. Variable-length padded tensors must be converted with
`PointCloud.from_padded` before being passed to an operation.

Point-aligned arguments match the input leading shape: `(N, ...)` for a
single tensor, `(B, N, ...)` for a dense tensor batch, or `(P, ...)` for a
packed cloud.

### `transform(cloud, transforms)`

Applies one `(4, 4)` transform to every batch entry, or corresponding
`(B, 4, 4)` transforms. Tensor inputs retain their `(N, 3)` or `(B, N, 3)`
shape. Packed inputs return a packed cloud. Attached normals are rotated;
features are shared.

### `procrustes(source, target, *, weights=None, estimate_scale=False)`

Differentiably aligns corresponding point rows using weighted SVD. Tensor
inputs may represent one cloud or an equal-length batch; packed inputs require
equal source and target lengths in every batch entry. Weights match the input
leading shape. The orientation-preserving result contains batched `rotation`,
`translation`, `scale`, and homogeneous `transforms`.

The default solves rigid Procrustes/Kabsch. `estimate_scale=True` solves the
Umeyama similarity problem and places `scale * rotation` in the upper-left of
`transforms`. Inputs and optional `(P,)` weights support ordinary PyTorch
autograd. Every batch needs at least three positively weighted non-collinear
points; gradients are undefined where the optimal rotation is not locally
unique.

### `voxelize(cloud, voxel_size)`

Returns `Voxelization(cloud, coordinates, point_to_voxel, counts)`. Coordinates
have columns `(batch, x, y, z)`. `Voxelization.reduce(values, reduction)`
supports `mean`, `sum`, `min`, `max`, and `first`.

### `estimate_normals(...)`

```text
estimate_normals(cloud, radius=None, *, k=30, index=None) -> NormalResult
```

Returns unit normals, a validity mask, and curvature (smallest eigenvalue over
the eigenvalue sum). If the cloud has normals attached, estimated normals are
sign-aligned to them; otherwise signs are arbitrary. The neighborhood includes
the query point. Invalid normals are zero and identified by
`NormalResult.valid`. This operation does not provide gradients.

## Search

### `NeighborIndex(reference, algorithm="auto")`

Reusable exact index with `knn`, `radius`, and `hybrid` methods. `algorithm` is
`"auto"`, `"bvh"`, or `"bruteforce"`. Multi-cloud references use brute force;
explicit BVH currently requires one cloud.

One-shot equivalents:

- `knn(reference, queries, k, algorithm="auto")`
- `radius_neighbors(reference, queries, radius, max_neighbors=64, algorithm="auto")`
- `hybrid_neighbors(reference, queries, radius, k, algorithm="auto")`

Every call returns `Neighbors(indices, distances2, valid)`. Valid candidates in
each row are returned in nondecreasing distance order (nearest first). The
ordering of candidates at equal distances is unspecified.

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

Returns `(B,)` tensor-valued precision, recall, and F-score at one scalar
threshold.

### `point_cloud_metrics(prediction, reference, threshold)`

Returns accuracy, completion, unsquared Chamfer distance, precision, recall,
and F-score. Every field retains its `(B,)` batch dimension. Accuracy is
prediction-to-reference; completion is the reverse.

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
