#include "backend.h"
#include "common/cuda.h"
#include "common/candidate.h"
#include "common/distance.h"

#include <math_constants.h>

#include <cstdint>

namespace torchpcl {
namespace {

constexpr int kBlock = 128;

template <int Capacity>
__global__ void packed_knn_kernel(
    int64_t* output_indices,
    float* output_distances,
    const float* points,
    const int64_t* point_offsets,
    const float* queries,
    const int64_t* query_batch_ids,
    int64_t query_count,
    int k,
    float bound2) {
  __shared__ float point_tile[kBlock * 3];
  __shared__ int64_t shared_batch;
  __shared__ bool homogeneous;

  const int64_t query =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = query < query_count;
  if (threadIdx.x == 0) {
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t block_end = first + blockDim.x;
    const int64_t last = (block_end < query_count ? block_end : query_count) - 1;
    shared_batch = query_batch_ids[first];
    homogeneous = query_batch_ids[last] == shared_batch;
  }
  __syncthreads();

  float best_distances[Capacity];
  int32_t best_indices[Capacity];
  for (int rank = 0; rank < k; ++rank) {
    best_distances[rank] = bound2;
    best_indices[rank] = -1;
  }
  float query_x = 0.0f;
  float query_y = 0.0f;
  float query_z = 0.0f;
  if (active) {
    query_x = queries[query * 3];
    query_y = queries[query * 3 + 1];
    query_z = queries[query * 3 + 2];
  }

  if (homogeneous) {
    const int64_t point_begin = point_offsets[shared_batch];
    const int64_t point_end = point_offsets[shared_batch + 1];
    for (int64_t tile = point_begin; tile < point_end; tile += blockDim.x) {
      const int64_t point = tile + threadIdx.x;
      if (point < point_end) {
        point_tile[threadIdx.x * 3] = points[point * 3];
        point_tile[threadIdx.x * 3 + 1] = points[point * 3 + 1];
        point_tile[threadIdx.x * 3 + 2] = points[point * 3 + 2];
      }
      __syncthreads();
      if (active) {
        const int64_t remaining = point_end - tile;
        const int tile_size = remaining < blockDim.x
            ? static_cast<int>(remaining)
            : static_cast<int>(blockDim.x);
        for (int offset = 0; offset < tile_size; ++offset) {
          const float distance2 = exact_distance2(
              point_tile[offset * 3], point_tile[offset * 3 + 1],
              point_tile[offset * 3 + 2], query_x, query_y, query_z);
          insert_candidate(
              best_distances,
              best_indices,
              k,
              distance2,
              static_cast<int32_t>(tile + offset));
        }
      }
      __syncthreads();
    }
  } else if (active) {
    const int64_t batch = query_batch_ids[query];
    const int64_t point_begin = point_offsets[batch];
    const int64_t point_end = point_offsets[batch + 1];
    for (int64_t point = point_begin; point < point_end; ++point) {
      const float distance2 = exact_distance2(
          points[point * 3], points[point * 3 + 1], points[point * 3 + 2],
          query_x, query_y, query_z);
      insert_candidate(
          best_distances,
          best_indices,
          k,
          distance2,
          static_cast<int32_t>(point));
    }
  }

  if (active) {
    for (int rank = 0; rank < k; ++rank) {
      const bool found = best_indices[rank] >= 0;
      output_indices[query * k + rank] = best_indices[rank];
      output_distances[query * k + rank] = found ? best_distances[rank] : CUDART_INF_F;
    }
  }
}

}  // namespace

SearchResult packed_knn_cuda(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids,
    int64_t k,
    double radius) {
  STD_TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
  const DeviceGuard guard(points.get_device_index());
  auto indices = torch::stable::new_empty(queries, {queries.size(0), k}, kLong);
  auto distances = torch::stable::new_empty(queries, {queries.size(0), k});
  if (queries.size(0) == 0) {
    return {indices, distances};
  }
  const auto stream = current_stream(points);
  const auto blocks = (queries.size(0) + kBlock - 1) / kBlock;
  if (k == 1) {
    packed_knn_kernel<1><<<blocks, kBlock, 0, stream>>>(
        indices.mutable_data_ptr<int64_t>(),
        distances.mutable_data_ptr<float>(),
        points.const_data_ptr<float>(),
        point_offsets.const_data_ptr<int64_t>(),
        queries.const_data_ptr<float>(),
        query_batch_ids.const_data_ptr<int64_t>(),
        queries.size(0),
        1,
        inclusive_bound2(radius));
  } else {
    packed_knn_kernel<kMaxK><<<blocks, kBlock, 0, stream>>>(
        indices.mutable_data_ptr<int64_t>(),
        distances.mutable_data_ptr<float>(),
        points.const_data_ptr<float>(),
        point_offsets.const_data_ptr<int64_t>(),
        queries.const_data_ptr<float>(),
        query_batch_ids.const_data_ptr<int64_t>(),
        queries.size(0),
        static_cast<int>(k),
        inclusive_bound2(radius));
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, distances};
}

}  // namespace torchpcl
