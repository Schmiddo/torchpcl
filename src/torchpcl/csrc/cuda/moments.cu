#include "backend.h"
#include "common/moments.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

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
    const at::Tensor& points, const at::Tensor& indices) {
  const c10::cuda::CUDAGuard guard(points.get_device());
  const int64_t count = indices.size(0);
  const int64_t k = indices.size(1);
  auto means = at::empty({count, 3}, points.options());
  auto scatter = at::empty({count, 3, 3}, points.options());
  auto counts = at::empty({count}, indices.options());
  if (count == 0) return {means, scatter, counts};
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto blocks = (count + kBlock - 1) / kBlock;
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "neighbor_moments_cuda", [&] {
    neighbor_moments_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        points.data_ptr<scalar_t>(),
        indices.data_ptr<int64_t>(),
        means.data_ptr<scalar_t>(),
        scatter.data_ptr<scalar_t>(),
        counts.data_ptr<int64_t>(),
        count,
        k);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means, scatter, counts};
}

}  // namespace torchpcl
