#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"

#include "backend.h"
#include "common/cuda.h"
#include "common/bvh.h"
#include "common/distance.h"

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
  // Blocks are allocated on the device of `like`.
  explicit TensorMemoryResource(const Tensor& like) : like_(like) {}

  void malloc(void** ptr, size_t size, cudaStream_t) override {
    auto block = torch::stable::new_empty(
        like_, {static_cast<int64_t>(size)}, kByte);
    *ptr = block.mutable_data_ptr();
    blocks_.emplace(*ptr, std::move(block));
  }

  void free(void* ptr, cudaStream_t) override { blocks_.erase(ptr); }

  // The block behind `ptr`, which cuBQL then no longer owns.
  Tensor take(void* ptr) {
    auto block = blocks_.extract(ptr);
    STD_TORCH_CHECK(
        !block.empty(), "cuBQL returned memory it did not allocate");
    return std::move(block.mapped());
  }

 private:
  const Tensor& like_;
  std::unordered_map<void*, Tensor> blocks_;
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

BvhResult build_bvh_cuda(const Tensor& points) {
  const DeviceGuard guard(points.get_device_index());
  const cudaStream_t stream = current_stream(points);
  const auto count = static_cast<int>(points.size(0));
  auto boxes = torch::stable::new_empty(points, {points.size(0), 6});
  auto any_kept = torch::stable::new_zeros(points, {1}, kInt);
  make_boxes<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      reinterpret_cast<cuBQL::box3f*>(boxes.mutable_data_ptr<float>()),
      any_kept.mutable_data_ptr<int>(),
      reinterpret_cast<const cuBQL::vec3f*>(points.const_data_ptr<float>()),
      count);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  // cuBQL's GPU builder faults when every box is empty; as on the CPU, such
  // a cloud gets no tree and `bvh_knn_cuda` answers without one. The builder
  // synchronizes anyway, so this read costs little.
  int kept = 0;
  STD_CUDA_CHECK(cudaMemcpyAsync(
      &kept, any_kept.const_data_ptr<int>(), sizeof(kept),
      cudaMemcpyDeviceToHost, stream));
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (kept == 0) {
    return {torch::stable::new_empty(points, {0, kBvhNodeBytes}, kByte),
            torch::stable::new_empty(points, {0}, kInt)};
  }
  TensorMemoryResource memory(points);
  cuBQL::bvh3f bvh{};
  cuBQL::gpuBuilder(
      bvh,
      reinterpret_cast<const cuBQL::box3f*>(boxes.mutable_data_ptr<float>()),
      static_cast<uint32_t>(count),
      cuBQL::BuildConfig(kCudaBvhLeafSize),
      stream,
      memory);
  const int64_t num_nodes = bvh.numNodes;
  auto node_block = memory.take(bvh.nodes);
  auto nodes = torch::stable::view(
      torch::stable::narrow(node_block, 0, 0, num_nodes * kBvhNodeBytes),
      {num_nodes, kBvhNodeBytes});
  // The stable ABI has no dtype view, so the ids are copied out of their
  // byte block; it is 4 bytes per point, next to a whole build.
  const int64_t num_prims = bvh.numPrims;
  auto prim_ids = torch::stable::new_empty(points, {num_prims}, kInt);
  STD_CUDA_CHECK(cudaMemcpyAsync(
      prim_ids.mutable_data_ptr<int32_t>(), bvh.primIDs,
      num_prims * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
  return {nodes, prim_ids};
}

SearchResult bvh_knn_cuda(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries,
    int64_t k,
    double radius) {
  const DeviceGuard guard(points.get_device_index());
  const cudaStream_t stream = current_stream(points);
  const auto count = queries.size(0);
  auto indices = torch::stable::new_empty(queries, {count, k}, kLong);
  auto distances = torch::stable::new_empty(queries, {count, k});
  if (count == 0) {
    return {indices, distances};
  }
  if (prim_ids.numel() == 0) {  // every reference point was NaN
    torch::stable::fill_(indices, -1);
    torch::stable::fill_(distances, std::numeric_limits<float>::infinity());
    return {indices, distances};
  }
  const cuBQL::bvh3f bvh = bvh_view(nodes, prim_ids);
  const float bound2 = inclusive_bound2(radius);
  const float cull2 = cull_bound2(bound2);
  const auto* point_data =
      reinterpret_cast<const cuBQL::vec3f*>(points.const_data_ptr<float>());
  const auto* query_data =
      reinterpret_cast<const cuBQL::vec3f*>(queries.const_data_ptr<float>());
  if (k == 1) {
    nearest_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        indices.mutable_data_ptr<int64_t>(),
        distances.mutable_data_ptr<float>(),
        bvh,
        point_data,
        query_data,
        static_cast<int>(count),
        cull2,
        bound2);
  } else {
    knn_query<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        indices.mutable_data_ptr<int64_t>(),
        distances.mutable_data_ptr<float>(),
        bvh,
        point_data,
        query_data,
        static_cast<int>(count),
        static_cast<int>(k),
        cull2,
        bound2);
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, distances};
}

}  // namespace torchpcl
