#include "backend.h"
#include "common/fpfh.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>

namespace torchpcl {

FpfhResult fpfh_cpu(
    const at::Tensor& points,
    const at::Tensor& normals,
    const at::Tensor& normal_valid,
    const at::Tensor& indices,
    const at::Tensor& distances2) {
  const int64_t count = points.size(0);
  const int64_t k = indices.size(1);
  auto spfh = at::empty({count, kFpfhSize}, points.options());
  auto spfh_valid = at::empty({count}, normal_valid.options());
  auto descriptors = at::empty({count, kFpfhSize}, points.options());
  auto valid = at::empty({count}, normal_valid.options());
  constexpr int64_t kGrainSize = 256;

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "fpfh_cpu", [&] {
    const auto* point_data = points.data_ptr<scalar_t>();
    const auto* normal_data = normals.data_ptr<scalar_t>();
    const auto* normal_valid_data = normal_valid.data_ptr<bool>();
    const auto* index_data = indices.data_ptr<int64_t>();
    auto* spfh_data = spfh.data_ptr<scalar_t>();
    auto* spfh_valid_data = spfh_valid.data_ptr<bool>();
    at::parallel_for(0, count, kGrainSize, [&](int64_t begin, int64_t end) {
      for (int64_t query = begin; query < end; ++query) {
        spfh_valid_data[query] = fpfh_spfh_one(
            query, k, point_data, normal_data, normal_valid_data, index_data,
            spfh_data + query * kFpfhSize);
      }
    });
    const auto* distance_data = distances2.data_ptr<scalar_t>();
    auto* descriptor_data = descriptors.data_ptr<scalar_t>();
    auto* valid_data = valid.data_ptr<bool>();
    at::parallel_for(0, count, kGrainSize, [&](int64_t begin, int64_t end) {
      for (int64_t query = begin; query < end; ++query) {
        valid_data[query] = fpfh_descriptor_one(
            query, k, point_data, normal_valid_data, index_data, distance_data,
            spfh_data, spfh_valid_data, descriptor_data + query * kFpfhSize);
      }
    });
  });
  return {descriptors, valid};
}

}  // namespace torchpcl
