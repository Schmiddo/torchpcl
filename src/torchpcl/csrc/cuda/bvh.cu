#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"

#include "backend.h"
#include "common/bvh.h"
#include "common/distance.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <unordered_map>

namespace torchpcl {
namespace {

constexpr int kBlock = 256;

// Serves cuBQL's builder from torch's caching allocator. Every block is a
// byte tensor held until cuBQL frees it, so the buffers of the finished BVH
// can be taken over as tensors. cuBQL allocates on the stream it builds on,
// which is torch's current stream, as the caching allocator expects.
class TensorMemoryResource final : public cuBQL::GpuMemoryResource {
 public:
  explicit TensorMemoryResource(at::Device device)
      : options_(at::TensorOptions().dtype(at::kByte).device(device)) {}

  void malloc(void** ptr, size_t size, cudaStream_t) override {
    auto block = at::empty({static_cast<int64_t>(size)}, options_);
    *ptr = block.data_ptr();
    blocks_.emplace(*ptr, std::move(block));
  }

  void free(void* ptr, cudaStream_t) override { blocks_.erase(ptr); }

  // The block behind `ptr`, which cuBQL then no longer owns.
  at::Tensor take(void* ptr) {
    auto block = blocks_.extract(ptr);
    TORCH_CHECK(!block.empty(), "cuBQL returned memory it did not allocate");
    return std::move(block.mapped());
  }

 private:
  at::TensorOptions options_;
  std::unordered_map<void*, at::Tensor> blocks_;
};

// Also raises `any_kept` when some point's box is non-empty (not NaN).
__global__ void make_boxes(
    cuBQL::box3f* boxes, int* any_kept, const cuBQL::vec3f* points, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    boxes[index] = cuBQL::box3f().including(points[index]);
    if (!boxes[index].empty()) *any_kept = 1;
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

}  // namespace

BvhResult build_bvh_cuda(const at::Tensor& points) {
  const c10::cuda::CUDAGuard guard(points.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto count = static_cast<int>(points.size(0));
  auto boxes = at::empty({points.size(0), 6}, points.options());
  auto any_kept = at::zeros({1}, points.options().dtype(at::kInt));
  make_boxes<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      reinterpret_cast<cuBQL::box3f*>(boxes.data_ptr<float>()),
      any_kept.data_ptr<int>(),
      reinterpret_cast<const cuBQL::vec3f*>(points.data_ptr<float>()),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // cuBQL's GPU builder faults when every box is empty; as on the CPU, such
  // a cloud gets no tree and `bvh_knn_cuda` answers without one. The builder
  // synchronizes anyway, so this read costs little.
  if (any_kept.item<int>() == 0) {
    return {at::empty({0, kBvhNodeBytes}, points.options().dtype(at::kByte)),
            at::empty({0}, points.options().dtype(at::kInt))};
  }
  TensorMemoryResource memory(points.device());
  cuBQL::bvh3f bvh{};
  cuBQL::gpuBuilder(
      bvh,
      reinterpret_cast<const cuBQL::box3f*>(boxes.data_ptr<float>()),
      static_cast<uint32_t>(count),
      cuBQL::BuildConfig(kCudaBvhLeafSize),
      stream,
      memory);
  auto nodes = memory.take(bvh.nodes)
                   .narrow(0, 0, int64_t{bvh.numNodes} * kBvhNodeBytes)
                   .view({int64_t{bvh.numNodes}, kBvhNodeBytes});
  auto prim_ids = memory.take(bvh.primIDs)
                      .narrow(0, 0, int64_t{bvh.numPrims} * 4)
                      .view(at::kInt);
  return {nodes, prim_ids};
}

SearchResult bvh_knn_cuda(
    const at::Tensor& nodes,
    const at::Tensor& prim_ids,
    const at::Tensor& points,
    const at::Tensor& queries,
    int64_t k,
    double radius) {
  const c10::cuda::CUDAGuard guard(points.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto count = queries.size(0);
  auto indices = at::empty({count, k}, queries.options().dtype(at::kLong));
  auto distances = at::empty({count, k}, queries.options());
  if (count == 0) {
    return {indices, distances};
  }
  if (prim_ids.numel() == 0) {  // every reference point was NaN
    indices.fill_(-1);
    distances.fill_(std::numeric_limits<float>::infinity());
    return {indices, distances};
  }
  const cuBQL::bvh3f bvh = bvh_view(nodes, prim_ids);
  const float bound2 = inclusive_bound2(radius);
  const float cull2 = cull_bound2(bound2);
  const auto* point_data =
      reinterpret_cast<const cuBQL::vec3f*>(points.data_ptr<float>());
  const auto* query_data =
      reinterpret_cast<const cuBQL::vec3f*>(queries.data_ptr<float>());
  if (k == 1) {
    nearest_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        bvh,
        point_data,
        query_data,
        static_cast<int>(count),
        cull2,
        bound2);
  } else {
    knn_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        bvh,
        point_data,
        query_data,
        static_cast<int>(count),
        static_cast<int>(k),
        cull2,
        bound2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, distances};
}

}  // namespace torchpcl
