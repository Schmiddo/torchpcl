#include "backend.h"
#include "common/symmetric_eigh.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace torchpcl {
namespace {

constexpr int kBlock = 256;

template <typename scalar_t>
__global__ void symmetric_eigh_3x3_kernel(
    const scalar_t* matrices,
    scalar_t* eigenvalues,
    scalar_t* smallest_eigenvectors,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    detail::symmetric_eigh_3x3_one(
        matrices + index * 9,
        eigenvalues + index * 3,
        smallest_eigenvectors + index * 3);
  }
}

}  // namespace

EigenResult symmetric_eigh_3x3_cuda(const at::Tensor& matrices) {
  const c10::cuda::CUDAGuard guard(matrices.get_device());
  auto output_sizes = matrices.sizes().vec();
  output_sizes.resize(output_sizes.size() - 2);
  output_sizes.push_back(3);
  auto eigenvalues = at::empty(output_sizes, matrices.options());
  auto smallest_eigenvectors = at::empty(output_sizes, matrices.options());
  const int64_t count = matrices.numel() / 9;
  if (count == 0) {
    return {eigenvalues, smallest_eigenvectors};
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto blocks = (count + kBlock - 1) / kBlock;
  AT_DISPATCH_FLOATING_TYPES(
      matrices.scalar_type(), "symmetric_eigh_3x3_cuda", [&] {
        symmetric_eigh_3x3_kernel<scalar_t>
            <<<blocks, kBlock, 0, stream>>>(
                matrices.data_ptr<scalar_t>(),
                eigenvalues.data_ptr<scalar_t>(),
                smallest_eigenvectors.data_ptr<scalar_t>(),
                count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {eigenvalues, smallest_eigenvectors};
}

}  // namespace torchpcl
