#pragma once

#include "cuBQL/math/vec.h"

#include <cmath>
#include <limits>

namespace torchpcl {

#ifdef __CUDACC__
#define TORCHPCL_HOST_DEVICE __host__ __device__
#else
#define TORCHPCL_HOST_DEVICE
#endif

// Squared distance in float32: three rounded products added left to right,
// the same on every device, so CPU and CUDA agree bit for bit. The products
// must not be fused into the additions (nvcc fuses by default). This is also
// how torch's CPU `(a - b).square().sum(-1)` rounds; its CUDA reduction adds
// the terms in another order and can differ in the last bit.
TORCHPCL_HOST_DEVICE inline float exact_distance2(
    float ax, float ay, float az, float bx, float by, float bz) {
  const float dx = ax - bx;
  const float dy = ay - by;
  const float dz = az - bz;
#ifdef __CUDA_ARCH__
  return __fmul_rn(dx, dx) + __fmul_rn(dy, dy) + __fmul_rn(dz, dz);
#else
  return dx * dx + dy * dy + dz * dz;
#endif
}

TORCHPCL_HOST_DEVICE inline float exact_distance2(
    cuBQL::vec3f a, cuBQL::vec3f b) {
  return exact_distance2(a.x, a.y, a.z, b.x, b.y, b.z);
}

// The squared search radius, which is inclusive: a point is a neighbor when
// its exact_distance2 is at most this.
inline float inclusive_bound2(double radius) {
  return static_cast<float>(radius * radius);
}

// The bound BVH traversal culls with. cuBQL culls strictly (`<`) and against
// distances rounded down, so the inclusive bound is widened by a few float32
// ULPs to keep every point on the boundary; exact_distance2 then drops the
// extra candidates this lets through.
inline float cull_bound2(float bound2) {
  return std::nextafter(
      bound2 * (1.0f + 1e-6f), std::numeric_limits<float>::infinity());
}

#undef TORCHPCL_HOST_DEVICE

}  // namespace torchpcl
