#include "backend.h"
#include "common/fpfh.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace torchpcl {
namespace {

constexpr int kBlock = 128;

template <typename scalar_t>
__global__ void spfh_kernel(
    const scalar_t* points,
    const scalar_t* normals,
    const bool* normal_valid,
    const int64_t* indices,
    scalar_t* spfh,
    bool* spfh_valid,
    int64_t count,
    int64_t k) {
  const int64_t query = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= count) return;
  scalar_t histogram[kFpfhSize];
  spfh_valid[query] = fpfh_spfh_one(
      query, k, points, normals, normal_valid, indices, histogram);
  for (int bin = 0; bin < kFpfhSize; ++bin) {
    spfh[query * kFpfhSize + bin] = histogram[bin];
  }
}

template <typename scalar_t>
__global__ void fpfh_descriptor_kernel(
    const scalar_t* points,
    const bool* normal_valid,
    const int64_t* indices,
    const scalar_t* distances2,
    const scalar_t* spfh,
    const bool* spfh_valid,
    scalar_t* descriptors,
    bool* valid,
    int64_t count,
    int64_t k) {
  const int64_t query = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= count) return;
  scalar_t descriptor[kFpfhSize];
  valid[query] = fpfh_descriptor_one(
      query, k, points, normal_valid, indices, distances2, spfh, spfh_valid,
      descriptor);
  for (int bin = 0; bin < kFpfhSize; ++bin) {
    descriptors[query * kFpfhSize + bin] = descriptor[bin];
  }
}

}  // namespace

FpfhResult fpfh_cuda(
    const at::Tensor& points,
    const at::Tensor& normals,
    const at::Tensor& normal_valid,
    const at::Tensor& indices,
    const at::Tensor& distances2) {
  const c10::cuda::CUDAGuard guard(points.get_device());
  const int64_t count = points.size(0);
  const int64_t k = indices.size(1);
  auto spfh = at::empty({count, kFpfhSize}, points.options());
  auto spfh_valid = at::empty({count}, normal_valid.options());
  auto descriptors = at::empty({count, kFpfhSize}, points.options());
  auto valid = at::empty({count}, normal_valid.options());
  if (count == 0) return {descriptors, valid};

  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto blocks = (count + kBlock - 1) / kBlock;
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "fpfh_cuda", [&] {
    spfh_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        points.data_ptr<scalar_t>(),
        normals.data_ptr<scalar_t>(),
        normal_valid.data_ptr<bool>(),
        indices.data_ptr<int64_t>(),
        spfh.data_ptr<scalar_t>(),
        spfh_valid.data_ptr<bool>(),
        count,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    fpfh_descriptor_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        points.data_ptr<scalar_t>(),
        normal_valid.data_ptr<bool>(),
        indices.data_ptr<int64_t>(),
        distances2.data_ptr<scalar_t>(),
        spfh.data_ptr<scalar_t>(),
        spfh_valid.data_ptr<bool>(),
        descriptors.data_ptr<scalar_t>(),
        valid.data_ptr<bool>(),
        count,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {descriptors, valid};
}

}  // namespace torchpcl
