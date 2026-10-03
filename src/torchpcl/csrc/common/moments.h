#pragma once

#include <cstdint>

namespace torchpcl {

#ifdef __CUDACC__
#define TORCHPCL_HOST_DEVICE __host__ __device__
#else
#define TORCHPCL_HOST_DEVICE
#endif

// The mean and centered 3x3 scatter matrix of the points a fixed-width
// neighborhood row names (-1 marks an empty slot), and how many it names.
// An empty row has zero mean and scatter.
template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline int64_t neighbor_moments_one(
    const scalar_t* points,
    const int64_t* row,
    int64_t k,
    scalar_t* mean,
    scalar_t* scatter) {
  int64_t count = 0;
  scalar_t sum[3] = {0, 0, 0};
  for (int64_t slot = 0; slot < k; ++slot) {
    const int64_t neighbor = row[slot];
    if (neighbor < 0) continue;
    ++count;
    for (int axis = 0; axis < 3; ++axis) sum[axis] += points[neighbor * 3 + axis];
  }
  const scalar_t denominator = static_cast<scalar_t>(count > 0 ? count : 1);
  for (int axis = 0; axis < 3; ++axis) mean[axis] = sum[axis] / denominator;

  scalar_t xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (int64_t slot = 0; slot < k; ++slot) {
    const int64_t neighbor = row[slot];
    if (neighbor < 0) continue;
    const scalar_t x = points[neighbor * 3 + 0] - mean[0];
    const scalar_t y = points[neighbor * 3 + 1] - mean[1];
    const scalar_t z = points[neighbor * 3 + 2] - mean[2];
    xx += x * x;
    xy += x * y;
    xz += x * z;
    yy += y * y;
    yz += y * z;
    zz += z * z;
  }
  scatter[0] = xx;
  scatter[1] = xy;
  scatter[2] = xz;
  scatter[3] = xy;
  scatter[4] = yy;
  scatter[5] = yz;
  scatter[6] = xz;
  scatter[7] = yz;
  scatter[8] = zz;
  return count;
}

#undef TORCHPCL_HOST_DEVICE

}  // namespace torchpcl
