#pragma once

#include "common/tensor.h"

#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>

namespace torchpcl {

using torch::stable::accelerator::DeviceGuard;

// torch's current stream on `tensor`'s device. The stable Stream's id() is
// c10's stream id, not a cudaStream_t, so this asks the CUDA shim instead.
inline cudaStream_t current_stream(const Tensor& tensor) {
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream));
  return static_cast<cudaStream_t>(stream);
}

}  // namespace torchpcl
