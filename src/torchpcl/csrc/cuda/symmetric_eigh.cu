#include "backend.h"
#include "common/cuda.h"
#include "common/symmetric_eigh.h"


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

EigenResult symmetric_eigh_3x3_cuda(const Tensor& matrices) {
  const DeviceGuard guard(matrices.get_device_index());
  auto output_sizes = matrices.sizes().vec();
  output_sizes.resize(output_sizes.size() - 2);
  output_sizes.push_back(3);
  auto eigenvalues = torch::stable::new_empty(matrices, output_sizes);
  auto smallest_eigenvectors = torch::stable::new_empty(matrices, output_sizes);
  const int64_t count = matrices.numel() / 9;
  if (count == 0) {
    return {eigenvalues, smallest_eigenvectors};
  }

  const auto stream = current_stream(matrices);
  const auto blocks = (count + kBlock - 1) / kBlock;
  THO_DISPATCH_V2(matrices.scalar_type(), "symmetric_eigh_3x3_cuda", ([&] {
        symmetric_eigh_3x3_kernel<scalar_t>
            <<<blocks, kBlock, 0, stream>>>(
                matrices.const_data_ptr<scalar_t>(),
                eigenvalues.mutable_data_ptr<scalar_t>(),
                smallest_eigenvectors.mutable_data_ptr<scalar_t>(),
                count);
      }), AT_FLOATING_TYPES);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return {eigenvalues, smallest_eigenvectors};
}

}  // namespace torchpcl
