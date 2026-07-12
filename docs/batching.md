# Packed Batching

Point clouds are ragged: storing every cloud at a shared padded length wastes
memory and complicates geometric operations. torchpcl therefore uses packed
points and offsets as its canonical batch representation.

## Representation

For lengths `[2, 0, 3]`, offsets are `[0, 2, 2, 5]` and points have shape
`(5, 3)`. Batch `b` occupies `points[offsets[b]:offsets[b + 1]]`. Empty batch
entries are represented by repeated offsets.

Offsets must:

- be one-dimensional int64 tensors;
- share the point device;
- start at zero;
- be nondecreasing;
- end at the total point count.

Neighbor indices are global packed row indices. Use the returned `valid` mask;
invalid slots contain index `-1` and infinite squared distance.

## Padded Conversion

```python
cloud = torchpcl.PointCloud.from_padded(points, lengths)
points, lengths = cloud.to_padded(pad_value=0.0)
```

Padding beyond each input length is ignored. `to_padded` pads to the longest
cloud and returns points and lengths; attached feature padding must be handled
separately.

## Operation Semantics

- `transform`, `voxelize`, search, normals, metrics, and registration share the
  same offsets convention.
- Voxel keys include batch identity, so points from different clouds never
  share a voxel.
- Search pairs queries and references by batch position.
- ICP tracks active, converged, and failed entries independently.
- `reduction="none"` returns one metric value per batch entry.

Storage, transforms, voxelization, and empty query batches allow empty entries.
Metrics and registration reject pairs where either corresponding cloud is
empty because their reductions and fitness are undefined.

## Dtypes and Devices

Geometry supports float32 and float64. Paired clouds must have identical dtype
and device; torchpcl does not silently cast one geometry input to another.
Offsets and point-aligned attributes must share the point device.

CUDA results stay on CUDA. Testing and application code should compare them
with `torch.testing.assert_close` rather than NumPy-based helpers.

## Gradients

Transforms, voxel reductions, and Chamfer distances use ordinary PyTorch
autograd. Neighbor identity, voxel membership, normal estimation, and ICP are
discrete or inference-only. Chamfer gradients treat the selected neighbor as
piecewise constant.

Exact distance ties may choose different indices across CPU, CUDA, BVH, and
brute-force backends. Distances and aggregate metrics should remain comparable.
