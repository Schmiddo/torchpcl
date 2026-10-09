#pragma once

// The part of torch's stable ABI torchpcl builds on: tensors, the ops in
// torch/csrc/stable/ops.h, STD_TORCH_CHECK and THO_DISPATCH_V2. Nothing here
// links against torch's C++ symbols, so one build serves every torch it
// targets.
//
// THO_DISPATCH_V2 bodies are written `([&] { ... })`. The macro forwards its
// body unwrapped, so a top-level comma -- such as a kernel launch's
// <<<grid, block, 0, stream>>> -- would split it even inside AT_WRAP; the
// parentheses keep it whole.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

namespace torchpcl {

using torch::headeronly::ScalarType;
using torch::stable::get_num_threads;
using torch::stable::parallel_for;
using torch::stable::Tensor;

constexpr ScalarType kBool = ScalarType::Bool;
constexpr ScalarType kByte = ScalarType::Byte;
constexpr ScalarType kInt = ScalarType::Int;
constexpr ScalarType kLong = ScalarType::Long;
constexpr ScalarType kFloat = ScalarType::Float;
constexpr ScalarType kDouble = ScalarType::Double;

}  // namespace torchpcl
