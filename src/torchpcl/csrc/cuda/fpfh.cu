#include "backend.h"
#include "common/cuda.h"
#include "common/fpfh.h"


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
    const Tensor& points,
    const Tensor& normals,
    const Tensor& normal_valid,
    const Tensor& indices,
    const Tensor& distances2) {
  const DeviceGuard guard(points.get_device_index());
  const int64_t count = points.size(0);
  const int64_t k = indices.size(1);
  auto spfh = torch::stable::new_empty(points, {count, kFpfhSize});
  auto spfh_valid = torch::stable::new_empty(normal_valid, {count});
  auto descriptors = torch::stable::new_empty(points, {count, kFpfhSize});
  auto valid = torch::stable::new_empty(normal_valid, {count});
  if (count == 0) return {descriptors, valid};

  const auto stream = current_stream(points);
  const auto blocks = (count + kBlock - 1) / kBlock;
  THO_DISPATCH_V2(points.scalar_type(), "fpfh_cuda", ([&] {
    spfh_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        points.const_data_ptr<scalar_t>(),
        normals.const_data_ptr<scalar_t>(),
        normal_valid.const_data_ptr<bool>(),
        indices.const_data_ptr<int64_t>(),
        spfh.mutable_data_ptr<scalar_t>(),
        spfh_valid.mutable_data_ptr<bool>(),
        count,
        k);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
    fpfh_descriptor_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        points.const_data_ptr<scalar_t>(),
        normal_valid.const_data_ptr<bool>(),
        indices.const_data_ptr<int64_t>(),
        distances2.const_data_ptr<scalar_t>(),
        spfh.mutable_data_ptr<scalar_t>(),
        spfh_valid.mutable_data_ptr<bool>(),
        descriptors.mutable_data_ptr<scalar_t>(),
        valid.mutable_data_ptr<bool>(),
        count,
        k);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }), AT_FLOATING_TYPES);
  return {descriptors, valid};
}

}  // namespace torchpcl
