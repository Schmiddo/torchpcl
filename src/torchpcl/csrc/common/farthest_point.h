#pragma once

#include <cstdint>

namespace torchpcl {

#ifdef __CUDACC__
#define TORCHPCL_HOST_DEVICE __host__ __device__
#else
#define TORCHPCL_HOST_DEVICE
#endif

// One point's running distance to the nearest already-selected center, and
// its packed row index. Shared by the CPU and CUDA farthest-point-sample
// kernels, which both reduce these to find the next farthest point.
template <typename scalar_t>
struct FarthestCandidate {
  scalar_t value;
  int64_t index;
};

// Prefers the larger value; ties break toward the lower index, so a
// reduction's result is independent of how work is partitioned across
// threads/chunks.
template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline FarthestCandidate<scalar_t> farther_candidate(
    FarthestCandidate<scalar_t> a, FarthestCandidate<scalar_t> b) {
  if (b.value > a.value) return b;
  if (b.value == a.value && b.index < a.index) return b;
  return a;
}

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline scalar_t squared_distance(
    const scalar_t* points, int64_t first, int64_t second) {
  const scalar_t dx = points[first * 3 + 0] - points[second * 3 + 0];
  const scalar_t dy = points[first * 3 + 1] - points[second * 3 + 1];
  const scalar_t dz = points[first * 3 + 2] - points[second * 3 + 2];
  return dx * dx + dy * dy + dz * dz;
}

#undef TORCHPCL_HOST_DEVICE

}  // namespace torchpcl
