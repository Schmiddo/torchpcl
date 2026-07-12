#include "backend.h"
#include "common/candidate.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cstdint>

namespace torchpcl {
namespace {

constexpr int kBlock = 128;

template <int Capacity>
__global__ void packed_knn_kernel(
    int32_t* output_indices,
    float* output_distances,
    const float* points,
    const int64_t* point_offsets,
    const float* queries,
    const int64_t* query_batch_ids,
    int64_t query_count,
    int k,
    float max_distance2) {
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
    best_distances[rank] = max_distance2;
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
          const float dx = query_x - point_tile[offset * 3];
          const float dy = query_y - point_tile[offset * 3 + 1];
          const float dz = query_z - point_tile[offset * 3 + 2];
          const float distance2 = dx * dx + dy * dy + dz * dz;
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
      const float dx = query_x - points[point * 3];
      const float dy = query_y - points[point * 3 + 1];
      const float dz = query_z - points[point * 3 + 2];
      const float distance2 = dx * dx + dy * dy + dz * dz;
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
      output_indices[query * k + rank] = best_indices[rank];
      output_distances[query * k + rank] = best_distances[rank];
    }
  }
}

}  // namespace

SearchResult packed_knn_cuda(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius) {
  TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
  const c10::cuda::CUDAGuard guard(points.get_device());
  auto indices = at::empty(
      {queries.size(0), k}, queries.options().dtype(at::kInt));
  auto distances = at::empty({queries.size(0), k}, queries.options());
  if (queries.size(0) == 0) {
    return {indices, distances};
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto blocks = (queries.size(0) + kBlock - 1) / kBlock;
  if (k == 1) {
    packed_knn_kernel<1><<<blocks, kBlock, 0, stream>>>(
        indices.data_ptr<int32_t>(),
        distances.data_ptr<float>(),
        points.data_ptr<float>(),
        point_offsets.data_ptr<int64_t>(),
        queries.data_ptr<float>(),
        query_batch_ids.data_ptr<int64_t>(),
        queries.size(0),
        1,
        static_cast<float>(radius * radius));
  } else {
    packed_knn_kernel<kMaxK><<<blocks, kBlock, 0, stream>>>(
        indices.data_ptr<int32_t>(),
        distances.data_ptr<float>(),
        points.data_ptr<float>(),
        point_offsets.data_ptr<int64_t>(),
        queries.data_ptr<float>(),
        query_batch_ids.data_ptr<int64_t>(),
        queries.size(0),
        static_cast<int>(k),
        static_cast<float>(radius * radius));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, distances};
}

}  // namespace torchpcl
