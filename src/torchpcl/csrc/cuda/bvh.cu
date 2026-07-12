#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"

#include "backend.h"
#include "common/checks.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <memory>

namespace torchpcl {
namespace {

constexpr int kBlock = 256;

__global__ void make_boxes(
    cuBQL::box3f* boxes, const cuBQL::vec3f* points, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    boxes[index] = cuBQL::box3f().including(points[index]);
  }
}

__global__ void nearest_query(
    int32_t* output_indices,
    float* output_distances,
    cuBQL::bvh3f bvh,
    const cuBQL::vec3f* points,
    const cuBQL::vec3f* queries,
    int count,
    float max_distance2) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= count) {
    return;
  }
  const int index =
      cuBQL::points::findClosest(bvh, points, queries[query], max_distance2);
  output_indices[query] = index;
  output_distances[query] = index >= 0
      ? cuBQL::sqrDistance(points[index], queries[query])
      : max_distance2;
}

__global__ void knn_query(
    int32_t* output_indices,
    float* output_distances,
    cuBQL::bvh3f bvh,
    const cuBQL::vec3f* points,
    const cuBQL::vec3f* queries,
    int count,
    int k,
    float max_distance2) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= count) {
    return;
  }
  cuBQL::knn::Candidate candidates[kMaxK];
  const auto result = cuBQL::points::findKNN(
      candidates, k, bvh, points, queries[query], max_distance2);
  for (int rank = 0; rank < k; ++rank) {
    const bool found = rank < result.numFound;
    output_indices[static_cast<int64_t>(query) * k + rank] =
        found ? candidates[rank].primID : -1;
    output_distances[static_cast<int64_t>(query) * k + rank] =
        found ? candidates[rank].sqrDist : max_distance2;
  }
}

class CudaBvh final : public BvhImpl {
 public:
  explicit CudaBvh(const at::Tensor& points)
      : points_(points), device_(points.get_device()) {
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
        cuBQL::BuildConfig{},
        stream);
  }

  ~CudaBvh() override {
    try {
      const c10::cuda::CUDAGuard guard(device_);
      cuBQL::cuda::free(bvh_, at::cuda::getCurrentCUDAStream());
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
        {count, k}, queries.options().dtype(at::kInt));
    auto distances = at::empty({count, k}, queries.options());
    if (count == 0) {
      return {indices, distances};
    }
    const float max_distance2 = static_cast<float>(radius * radius);
    if (k == 1) {
      nearest_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
          indices.data_ptr<int32_t>(),
          distances.data_ptr<float>(),
          bvh_,
          reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
          reinterpret_cast<const cuBQL::vec3f*>(queries.data_ptr<float>()),
          static_cast<int>(count),
          max_distance2);
    } else {
      knn_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
          indices.data_ptr<int32_t>(),
          distances.data_ptr<float>(),
          bvh_,
          reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
          reinterpret_cast<const cuBQL::vec3f*>(queries.data_ptr<float>()),
          static_cast<int>(count),
          static_cast<int>(k),
          max_distance2);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {indices, distances};
  }

 private:
  at::Tensor points_;
  cuBQL::bvh3f bvh_{};
  c10::DeviceIndex device_;
};

}  // namespace

std::shared_ptr<BvhImpl> make_cuda_bvh(const at::Tensor& points) {
  return std::make_shared<CudaBvh>(points);
}

}  // namespace torchpcl
