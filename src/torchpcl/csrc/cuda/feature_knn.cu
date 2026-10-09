#include "backend.h"
#include "common/cuda.h"


#include <cstdint>
#include <limits>

namespace torchpcl {
namespace {

constexpr int kBlock = 128;
// Reference rows per tile, and components per chunk of a row.
constexpr int kRows = 32;
constexpr int kChunk = 32;

// One thread per query. The block stages a tile of kRows reference rows in
// shared memory one kChunk-component chunk at a time, which every thread
// reads in lockstep (a broadcast); each thread holds its query's chunk and
// the tile's kRows partial distances in registers. Ties keep the lower index.
template <typename scalar_t>
__global__ void feature_knn1_kernel(
    int64_t* output_indices,
    scalar_t* output_distances2,
    const scalar_t* reference,
    const scalar_t* queries,
    int64_t reference_count,
    int64_t query_count,
    int64_t dimension) {
  __shared__ scalar_t tile[kRows][kChunk];
  const int64_t query = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = query < query_count;
  // Inactive threads still help stage tiles; they read row 0 and discard it.
  const scalar_t* query_row = queries + (active ? query : 0) * dimension;

  scalar_t best_distance2 = std::numeric_limits<scalar_t>::infinity();
  int64_t best_index = -1;
  for (int64_t start = 0; start < reference_count; start += kRows) {
    const int rows = static_cast<int>(
        min(static_cast<int64_t>(kRows), reference_count - start));
    scalar_t distances2[kRows];
#pragma unroll
    for (int row = 0; row < kRows; ++row) distances2[row] = scalar_t(0);

    for (int64_t first = 0; first < dimension; first += kChunk) {
      const int components = static_cast<int>(
          min(static_cast<int64_t>(kChunk), dimension - first));
      __syncthreads();
      // Padding rows and components are zero on both sides, adding nothing.
      for (int offset = threadIdx.x; offset < kRows * kChunk; offset += blockDim.x) {
        const int row = offset / kChunk;
        const int component = offset % kChunk;
        tile[row][component] = row < rows && component < components
            ? reference[(start + row) * dimension + first + component]
            : scalar_t(0);
      }
      __syncthreads();
      scalar_t chunk[kChunk];
#pragma unroll
      for (int component = 0; component < kChunk; ++component) {
        chunk[component] = component < components
            ? __ldg(query_row + first + component)
            : scalar_t(0);
      }
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
#pragma unroll
        for (int component = 0; component < kChunk; ++component) {
          const scalar_t delta = chunk[component] - tile[row][component];
          distances2[row] += delta * delta;
        }
      }
    }
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      if (row < rows && distances2[row] < best_distance2) {
        best_distance2 = distances2[row];
        best_index = start + row;
      }
    }
  }
  if (active) {
    output_indices[query] = best_index;
    output_distances2[query] = best_distance2;
  }
}

}  // namespace

SearchResult feature_knn1_cuda(
    const Tensor& reference, const Tensor& queries) {
  const DeviceGuard guard(reference.get_device_index());
  auto indices = torch::stable::new_empty(queries, {queries.size(0)}, kLong);
  auto distances2 = torch::stable::new_empty(queries, {queries.size(0)});
  if (queries.size(0) == 0) {
    return {indices, distances2};
  }

  const auto stream = current_stream(reference);
  const auto blocks = (queries.size(0) + kBlock - 1) / kBlock;
  THO_DISPATCH_V2(reference.scalar_type(), "feature_knn1_cuda", ([&] {
        feature_knn1_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
            indices.mutable_data_ptr<int64_t>(),
            distances2.mutable_data_ptr<scalar_t>(),
            reference.const_data_ptr<scalar_t>(),
            queries.const_data_ptr<scalar_t>(),
            reference.size(0),
            queries.size(0),
            reference.size(1));
      }), AT_FLOATING_TYPES);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, distances2};
}

}  // namespace torchpcl
