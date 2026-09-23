#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"

#include "backend.h"
#include "common/checks.h"
#include "common/distance.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <math_constants.h>

#include <cstdint>
#include <memory>

namespace torchpcl {
namespace {

constexpr int kBlock = 256;

class DeviceAsyncMemoryResource final : public cuBQL::GpuMemoryResource {
 public:
  explicit DeviceAsyncMemoryResource(c10::DeviceIndex device)
      : device_(device) {
    const c10::cuda::CUDAGuard guard(device_);
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memory_pool_, device_));
  }

  void malloc(void** ptr, size_t size, cudaStream_t stream) override {
    const c10::cuda::CUDAGuard guard(device_);
    C10_CUDA_CHECK(
        cudaMallocFromPoolAsync(ptr, size, memory_pool_, stream));
  }

  void free(void* ptr, cudaStream_t stream) override {
    const c10::cuda::CUDAGuard guard(device_);
    C10_CUDA_CHECK(cudaFreeAsync(ptr, stream));
  }

 private:
  c10::DeviceIndex device_;
  cudaMemPool_t memory_pool_{};
};

__global__ void make_boxes(
    cuBQL::box3f* boxes, const cuBQL::vec3f* points, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    boxes[index] = cuBQL::box3f().including(points[index]);
  }
}

// One output slot: the candidate and its exact distance, or empty.
__device__ inline void write_slot(
    int64_t* output_indices,
    float* output_distances,
    int64_t slot,
    int index,
    const cuBQL::vec3f* points,
    cuBQL::vec3f query,
    float bound2) {
  const float distance2 =
      index >= 0 ? exact_distance2(points[index], query) : CUDART_INF_F;
  const bool kept = index >= 0 && distance2 <= bound2;
  output_indices[slot] = kept ? index : -1;
  output_distances[slot] = kept ? distance2 : CUDART_INF_F;
}

__global__ void nearest_query(
    int64_t* output_indices,
    float* output_distances,
    cuBQL::bvh3f bvh,
    const cuBQL::vec3f* points,
    const cuBQL::vec3f* queries,
    int count,
    float cull2,
    float bound2) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= count) {
    return;
  }
  write_slot(
      output_indices, output_distances, query,
      cuBQL::points::findClosest(bvh, points, queries[query], cull2),
      points, queries[query], bound2);
}

__global__ void knn_query(
    int64_t* output_indices,
    float* output_distances,
    cuBQL::bvh3f bvh,
    const cuBQL::vec3f* points,
    const cuBQL::vec3f* queries,
    int count,
    int k,
    float cull2,
    float bound2) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= count) {
    return;
  }
  cuBQL::knn::Candidate candidates[kMaxK];
  const auto result = cuBQL::points::findKNN(
      candidates, k, bvh, points, queries[query], cull2);
  for (int rank = 0; rank < k; ++rank) {
    write_slot(
        output_indices, output_distances,
        static_cast<int64_t>(query) * k + rank,
        rank < result.numFound ? candidates[rank].primID : -1,
        points, queries[query], bound2);
  }
}

class CudaBvh final : public BvhImpl {
 public:
  explicit CudaBvh(const at::Tensor& points)
      : points_(points),
        device_(points.get_device()),
        memory_resource_(device_) {
    TORCH_CHECK(points_.is_cuda(), "points must be a CUDA tensor");
    const c10::cuda::CUDAGuard guard(device_);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto count = static_cast<int>(points_.size(0));
    auto boxes = at::empty(
        {points_.size(0), 6}, points_.options().dtype(at::kFloat));
    make_boxes<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        reinterpret_cast<cuBQL::box3f*>(boxes.data_ptr<float>()),
        reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cuBQL::gpuBuilder(
        bvh_,
        reinterpret_cast<const cuBQL::box3f*>(boxes.data_ptr<float>()),
        static_cast<uint32_t>(count),
        cuBQL::BuildConfig(kCudaBvhLeafSize),
        stream,
        memory_resource_);
  }

  ~CudaBvh() override {
    try {
      const c10::cuda::CUDAGuard guard(device_);
      cuBQL::cuda::free(
          bvh_, at::cuda::getCurrentCUDAStream(), memory_resource_);
    } catch (...) {
    }
  }

  SearchResult knn(
      const at::Tensor& queries, int64_t k, double radius) override {
    check_points(queries, "queries");
    TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
    TORCH_CHECK(
        queries.get_device() == device_,
        "queries must be on the same device as the indexed points");
    const c10::cuda::CUDAGuard guard(device_);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto count = queries.size(0);
    auto indices = at::empty(
        {count, k}, queries.options().dtype(at::kLong));
    auto distances = at::empty({count, k}, queries.options());
    if (count == 0) {
      return {indices, distances};
    }
    const float bound2 = inclusive_bound2(radius);
    const float cull2 = cull_bound2(bound2);
    if (k == 1) {
      nearest_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
          indices.data_ptr<int64_t>(),
          distances.data_ptr<float>(),
          bvh_,
          reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
          reinterpret_cast<const cuBQL::vec3f*>(queries.data_ptr<float>()),
          static_cast<int>(count),
          cull2,
          bound2);
    } else {
      knn_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
          indices.data_ptr<int64_t>(),
          distances.data_ptr<float>(),
          bvh_,
          reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
          reinterpret_cast<const cuBQL::vec3f*>(queries.data_ptr<float>()),
          static_cast<int>(count),
          static_cast<int>(k),
          cull2,
          bound2);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {indices, distances};
  }

 private:
  at::Tensor points_;
  c10::DeviceIndex device_;
  DeviceAsyncMemoryResource memory_resource_;
  cuBQL::bvh3f bvh_{};
};

}  // namespace

std::shared_ptr<BvhImpl> make_cuda_bvh(const at::Tensor& points) {
  return std::make_shared<CudaBvh>(points);
}

}  // namespace torchpcl
