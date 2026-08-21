#include "backend.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <limits>

namespace torchpcl {
namespace {

constexpr int kBlock = 256;
constexpr int kWarp = 32;

template <typename scalar_t>
__global__ void feature_knn1_kernel(
    int64_t* output_indices,
    scalar_t* output_distances2,
    const scalar_t* reference,
    const scalar_t* queries,
    int64_t reference_count,
    int64_t query_count,
    int64_t dimension,
    scalar_t initial_distance2) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int warps_per_block = blockDim.x / kWarp;
  const int64_t query =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (query >= query_count) {
    return;
  }

  scalar_t best_distance2 = initial_distance2;
  int64_t best_index = -1;
  const scalar_t* query_row = queries + query * dimension;
  for (int64_t candidate = 0; candidate < reference_count; ++candidate) {
    const scalar_t* reference_row = reference + candidate * dimension;
    scalar_t distance2 = 0;
    for (int64_t component = lane; component < dimension; component += kWarp) {
      const scalar_t delta = query_row[component] - reference_row[component];
      distance2 += delta * delta;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      distance2 += __shfl_down_sync(0xffffffff, distance2, offset);
    }
    if (lane == 0 &&
        (distance2 < best_distance2 ||
         (distance2 == best_distance2 &&
          (best_index < 0 || candidate < best_index)))) {
      best_distance2 = distance2;
      best_index = candidate;
    }
  }
  if (lane == 0) {
    output_indices[query] = best_index;
    output_distances2[query] = best_distance2;
  }
}

}  // namespace

SearchResult feature_knn1_cuda(
    const at::Tensor& reference, const at::Tensor& queries) {
  const c10::cuda::CUDAGuard guard(reference.get_device());
  auto indices = at::empty({queries.size(0)}, queries.options().dtype(at::kLong));
  auto distances2 = at::empty({queries.size(0)}, queries.options());
  if (queries.size(0) == 0) {
    return {indices, distances2};
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int warps_per_block = kBlock / kWarp;
  const auto blocks =
      (queries.size(0) + warps_per_block - 1) / warps_per_block;
  AT_DISPATCH_FLOATING_TYPES(
      reference.scalar_type(), "feature_knn1_cuda", [&] {
        feature_knn1_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            distances2.data_ptr<scalar_t>(),
            reference.data_ptr<scalar_t>(),
            queries.data_ptr<scalar_t>(),
            reference.size(0),
            queries.size(0),
            reference.size(1),
            std::numeric_limits<scalar_t>::infinity());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, distances2};
}

}  // namespace torchpcl
