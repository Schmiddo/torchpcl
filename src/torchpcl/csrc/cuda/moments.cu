#include "backend.h"
#include "common/cuda.h"
#include "common/moments.h"


namespace torchpcl {
namespace {

constexpr int kBlock = 256;

template <typename scalar_t>
__global__ void neighbor_moments_kernel(
    const scalar_t* points,
    const int64_t* indices,
    scalar_t* means,
    scalar_t* scatter,
    int64_t* counts,
    int64_t count,
    int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  counts[row] = neighbor_moments_one(
      points, indices + row * k, k, means + row * 3, scatter + row * 9);
}

}  // namespace

MomentsResult neighbor_moments_cuda(
    const Tensor& points, const Tensor& indices) {
  const DeviceGuard guard(points.get_device_index());
  const int64_t count = indices.size(0);
  const int64_t k = indices.size(1);
  auto means = torch::stable::new_empty(points, {count, 3});
  auto scatter = torch::stable::new_empty(points, {count, 3, 3});
  auto counts = torch::stable::new_empty(indices, {count});
  if (count == 0) return {means, scatter, counts};
  const auto stream = current_stream(points);
  const auto blocks = (count + kBlock - 1) / kBlock;
  THO_DISPATCH_V2(points.scalar_type(), "neighbor_moments_cuda", ([&] {
    neighbor_moments_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        points.const_data_ptr<scalar_t>(),
        indices.const_data_ptr<int64_t>(),
        means.mutable_data_ptr<scalar_t>(),
        scatter.mutable_data_ptr<scalar_t>(),
        counts.mutable_data_ptr<int64_t>(),
        count,
        k);
  }), AT_FLOATING_TYPES);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return {means, scatter, counts};
}

}  // namespace torchpcl
