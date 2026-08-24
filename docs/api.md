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
- `cloud.with_normals(normals)`
- `cloud.with_features(features)`
- `cloud.select_points(mask)`
- `cat(clouds)` concatenates corresponding batches along the point dimension
- `cat_batch(clouds)` appends complete batch entries

`select_points` accepts a packed `(P,)` boolean mask and retains the batch
count and order, attached normals and features, and ordinary gradients through
the selected tensors. Batch entries may become empty.

### PLY I/O

`torchpcl.io.load(path)` loads one PLY vertex cloud. The `x`, `y`, and `z`
properties are required; complete `nx`/`ny`/`nz` normals are preserved when present.
The geometry output dtype can be selected with `dtype=torch.float32` or
`dtype=torch.float64`.

`torchpcl.io.save(path, cloud)` saves one tensor or one-entry `PointCloud` as binary
PLY. Additional scalar vertex properties can be mapped to the `PointCloud.features`
tensor by passing the same ordered sequence of strings to `feature_names` on load
and save. Feature trailing dimensions are flattened when saving and load back as
`(P, F)`. Supported feature dtypes are `int8`, `int16`, `int32`, `uint8`, `uint16`,
`uint32`, `float32`, and `float64`.
The equivalent `load_ply` and `save_ply` names are also exported from `torchpcl`.

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

### Normal orientation

```text
orient_normals_toward_viewpoint(cloud, viewpoint, *, toward=True) -> PointCloud
orient_normals_to_direction(cloud, direction, *, align=True) -> PointCloud
```

Both operations require a `PointCloud` with attached normals and return a new
PointCloud. Viewpoints and directions may be shared `(3,)` tensors or per-cloud
`(B, 3)` tensors with matching dtype and device. Viewpoint orientation leaves
normals unchanged when a point coincides with the viewpoint; direction vectors
must be nonzero.

### `estimate_covariances(...)`

```text
estimate_covariances(cloud, *, radius=inf, k=30) -> CovarianceResult
```

Returns packed `(P, 3)` neighborhood means, `(P, 3, 3)` covariance matrices,
integer counts, and a validity mask. Uses hybrid k-NN/radius search. Rows with fewer
than three neighbors are invalid and have zero means and covariances.

`covariance_from_neighbors(cloud, neighbors)` reuses a fixed-width `Neighbors` result
with one row per packed point. Neighborhood identity is discrete, while means and
covariance entries retain gradients to the gathered coordinates.

## Search

### `NeighborIndex(reference, algorithm="auto")`

Reusable index with `knn`, `radius`, and `hybrid` methods. `algorithm` is
`"auto"`, `"bvh"`, or `"bruteforce"`. Multi-cloud references use brute force;
explicit BVH currently requires one cloud. Candidate selection uses float32
coordinates for both float32 and float64 inputs. Squared distances are
recomputed from the original points and queries in their input dtype.

One-shot equivalents:

- `knn(reference, queries, k, algorithm="auto")`
- `radius_neighbors(reference, queries, radius, max_neighbors=64, algorithm="auto")`
- `hybrid_neighbors(reference, queries, radius, k, algorithm="auto")`

Every call returns `Neighbors(indices, distances2, valid)`. Candidate order and
distance-tie resolution are unspecified.

## Filtering

### `radius_outliers(...)`

```text
radius_outliers(cloud, radius, k) -> torch.Tensor
```

Outliers are points with less than `k` neighbors in the given radius.

### `statistical_outliers(...)`

```text
statistical_outliers(cloud, k, std_ratio=2.0) -> torch.Tensor
```

Computes each point's available mean unsquared k-NN distance, then classifies
against `cloud_mean + std_ratio * cloud_std` independently per batch entry.
The standard deviation is the sample standard deviation; one finite score has
standard deviation zero. Missing point scores and empty-cloud summaries are
NaN, and rows without scores are outliers.

Both outlier functions run without gradient tracking and return packed outlier masks.

## Features

### `fpfh(...)`

```text
fpfh(cloud, radius, *, max_neighbors=64) -> FPFHResult
```

Computes point-aligned 33-value fast point feature histograms from a
`PointCloud` with attached, consistently oriented normals. The result contains
`descriptors`, a per-row `valid` mask, and `neighbor_limit_reached`, which
signals that the bounded search filled all available slots. Invalid descriptor
rows are zero. This inference-only implementation supports at most 64 neighbors.

For feature matching, select valid rows before attaching the descriptors:

```python
features = fpfh(cloud, radius=0.25)
cloud = cloud.select_points(features.valid).with_features(
    features.descriptors[features.valid]
)
```

## Metrics

### `chamfer_distance(...)`

```text
chamfer_distance(source, target, *, squared=True, bidirectional=True,
                 reduction="mean") -> Tensor
```

The forward nearest-neighbor distances are averaged within each cloud. When
`bidirectional=True`, the reverse mean is included and the two directions are
averaged. `reduction="none"` returns one value per packed batch entry;
`"mean"` and `"sum"` reduce those batch values.

### `fscore(prediction, reference, threshold)`

Returns `(B,)` tensor-valued precision, recall, and F-score at one scalar
threshold.

### `point_cloud_metrics(prediction, reference, threshold)`

Returns accuracy, completion, unsquared Chamfer distance, precision, recall,
and F-score. Every field retains its `(B,)` batch dimension. Accuracy is
prediction-to-reference; completion is the reverse.

## Registration

### Feature matching and fast global registration

```text
match_features(source, target, *, mutual=True) -> CorrespondenceSet
fast_global_registration(source, target, *, options) -> FGRResult
fast_global_registration_from_correspondences(
    source, target, correspondences, *, options
) -> FGRResult
```

`match_features` uses exact squared-L2 matching over point-aligned
`PointCloud.features`. `CorrespondenceSet` stores source and target point-row
indices plus optional squared descriptor distances.

The current FGR prototype supports one source/target cloud pair. The
feature-driven entry point performs mutual matching and delegates to the
correspondence-driven solver. `FGROptions` requires
`max_correspondence_distance` and provides an iteration budget plus optional
tuple filtering. Results contain batch-shaped source-to-target transforms,
success flags, iteration and correspondence counts, fitness, and inlier RMSE.
Success means a finite optimization completed; fitness and RMSE determine
whether the alignment is useful.

See `examples/global_registration.py` for the complete normals, FPFH, FGR, and
ICP workflow.

### `icp(...)`

```text
icp(source, target, levels, *, init=None, options=None) -> ICPResult
```

`levels` is a nonempty sequence of `ICPLevel` objects. Every level defines a
maximum correspondence distance, an iteration budget, and an optional voxel
size. `voxel_size=None` uses the original clouds. Other levels are voxelized
independently from the original inputs and execute in the supplied order.

`ICPOptions` selects a `PointToPoint` or `PointToPlane` objective, convergence
tolerances, and an optional `HuberLoss`. Point-to-plane requires normals
attached to the target `PointCloud`; ICP never estimates normals implicitly.

The result contains final transforms, final-level convergence and metrics,
total iteration counts, and a tuple of `ICPLevelResult` diagnostics. It never
contains correspondences.

Configuration example:

```python
levels = [
    ICPLevel(
        voxel_size=0.2,
        max_correspondence_distance=0.4,
        max_iterations=30,
    ),
    ICPLevel(
        voxel_size=None,
        max_correspondence_distance=0.1,
        max_iterations=20,
    ),
]
options = ICPOptions(
    objective=PointToPlane(),
    convergence=ConvergenceCriteria(
        fitness_tolerance=1e-6,
        rmse_tolerance=1e-6,
    ),
    robust_loss=HuberLoss(delta=0.02),
)
result = icp(source, target, levels, options=options)
```

### `evaluate_registration(...)`

```text
evaluate_registration(source, target, max_correspondence_distance,
                      transforms=None) -> RegistrationMetrics
```

Returns transforms, fitness, and inlier RMSE without ICP updates.
