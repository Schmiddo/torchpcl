#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kBlock = 128;
constexpr int kMaxK = 64;

__device__ inline bool better(
    float d2, int32_t index, float other_d2, int32_t other_index) {
  return d2 < other_d2 ||
      (d2 == other_d2 && (other_index < 0 || index < other_index));
}

template <int Capacity>
__global__ void tiled_knn(
    int32_t* out_idx,
    float* out_d2,
    const float* points,
    const int64_t* point_offsets,
    const float* queries,
    const int64_t* query_offsets,
    int k,
    float max_d2) {
  __shared__ float point_tile[kBlock * 3];
  const int batch = blockIdx.y;
  const int64_t query = query_offsets[batch] +
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t query_end = query_offsets[batch + 1];
  const bool active = query < query_end;
  float best_d2[Capacity];
  int32_t best_idx[Capacity];
  for (int rank = 0; rank < k; ++rank) {
    best_d2[rank] = max_d2;
    best_idx[rank] = -1;
  }
  float qx = 0.0f, qy = 0.0f, qz = 0.0f;
  if (active) {
    qx = queries[query * 3];
    qy = queries[query * 3 + 1];
    qz = queries[query * 3 + 2];
  }

  const int64_t point_begin = point_offsets[batch];
  const int64_t point_end = point_offsets[batch + 1];
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
        const float dx = qx - point_tile[offset * 3];
        const float dy = qy - point_tile[offset * 3 + 1];
        const float dz = qz - point_tile[offset * 3 + 2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        const auto index = static_cast<int32_t>(tile + offset);
        if (!better(d2, index, best_d2[k - 1], best_idx[k - 1])) continue;
        int rank = k - 1;
        while (rank > 0 && better(d2, index, best_d2[rank - 1], best_idx[rank - 1])) {
          best_d2[rank] = best_d2[rank - 1];
          best_idx[rank] = best_idx[rank - 1];
          --rank;
        }
        best_d2[rank] = d2;
        best_idx[rank] = index;
      }
    }
    __syncthreads();
  }
  if (active) {
    for (int rank = 0; rank < k; ++rank) {
      out_idx[query * k + rank] = best_idx[rank];
      out_d2[query * k + rank] = best_d2[rank];
    }
  }
}

void check_points(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == 3,
              name, " must have shape (N, 3)");
}

std::vector<int64_t> checked_offsets(
    const at::Tensor& offsets, int64_t size, int device, const char* name) {
  TORCH_CHECK(offsets.is_cuda() && offsets.get_device() == device,
              name, " must be on the same CUDA device as points");
  TORCH_CHECK(offsets.scalar_type() == at::kLong, name, " must be int64");
  TORCH_CHECK(offsets.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() >= 2,
              name, " must have shape (B + 1,)");
  // The small synchronization keeps validation and ragged launch sizing simple.
  auto cpu = offsets.to(at::kCPU);
  const auto* values = cpu.data_ptr<int64_t>();
  std::vector<int64_t> result(values, values + offsets.numel());
  TORCH_CHECK(result.front() == 0, name, " must start at zero");
  for (size_t i = 1; i < result.size(); ++i) {
    TORCH_CHECK(result[i] >= result[i - 1], name, " must be nondecreasing");
  }
  TORCH_CHECK(result.back() == size, name, " must end at the tensor row count");
  return result;
}

std::tuple<at::Tensor, at::Tensor> knn(
    at::Tensor points,
    at::Tensor point_offsets,
    at::Tensor queries,
    at::Tensor query_offsets,
    int64_t k,
    double radius) {
  check_points(points, "points");
  check_points(queries, "queries");
  TORCH_CHECK(queries.get_device() == points.get_device(),
              "queries must be on the same CUDA device as points");
  TORCH_CHECK(k >= 1 && k <= kMaxK, "k must be in [1, ", kMaxK, "]");
  TORCH_CHECK(radius > 0.0 && !std::isnan(radius), "radius must be positive");
  TORCH_CHECK(points.size(0) <= INT32_MAX, "points has too many rows");

  const c10::cuda::CUDAGuard guard(points.get_device());
  const auto po = checked_offsets(
      point_offsets, points.size(0), points.get_device(), "point_offsets");
  const auto qo = checked_offsets(
      query_offsets, queries.size(0), points.get_device(), "query_offsets");
  TORCH_CHECK(po.size() == qo.size(),
              "point_offsets and query_offsets must describe the same batch count");
  TORCH_CHECK(po.size() - 1 <= 65535, "batch count exceeds the CUDA grid limit");
  int64_t max_queries = 0;
  for (size_t batch = 0; batch + 1 < po.size(); ++batch) {
    TORCH_CHECK(po[batch] < po[batch + 1] || qo[batch] == qo[batch + 1],
                "a batch with queries must contain at least one point");
    max_queries = std::max(max_queries, qo[batch + 1] - qo[batch]);
  }

  auto indices = at::empty({queries.size(0), k}, queries.options().dtype(at::kInt));
  auto distances = at::empty({queries.size(0), k}, queries.options());
  if (max_queries > 0) {
    const dim3 grid(
        static_cast<unsigned>((max_queries + kBlock - 1) / kBlock),
        static_cast<unsigned>(po.size() - 1));
    if (k == 1) {
      tiled_knn<1><<<grid, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          indices.data_ptr<int32_t>(), distances.data_ptr<float>(),
          points.data_ptr<float>(), point_offsets.data_ptr<int64_t>(),
          queries.data_ptr<float>(), query_offsets.data_ptr<int64_t>(),
          1, static_cast<float>(radius * radius));
    } else {
      tiled_knn<kMaxK><<<grid, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          indices.data_ptr<int32_t>(), distances.data_ptr<float>(),
          points.data_ptr<float>(), point_offsets.data_ptr<int64_t>(),
          queries.data_ptr<float>(), query_offsets.data_ptr<int64_t>(),
          static_cast<int>(k), static_cast<float>(radius * radius));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {indices, distances};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("knn", &knn); }
