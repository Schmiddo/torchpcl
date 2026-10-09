#include "backend.h"
#include "common/fpfh.h"


namespace torchpcl {

FpfhResult fpfh_cpu(
    const Tensor& points,
    const Tensor& normals,
    const Tensor& normal_valid,
    const Tensor& indices,
    const Tensor& distances2) {
  const int64_t count = points.size(0);
  const int64_t k = indices.size(1);
  auto spfh = torch::stable::new_empty(points, {count, kFpfhSize});
  auto spfh_valid = torch::stable::new_empty(normal_valid, {count});
  auto descriptors = torch::stable::new_empty(points, {count, kFpfhSize});
  auto valid = torch::stable::new_empty(normal_valid, {count});
  constexpr int64_t kGrainSize = 256;

  THO_DISPATCH_V2(points.scalar_type(), "fpfh_cpu", ([&] {
    const auto* point_data = points.const_data_ptr<scalar_t>();
    const auto* normal_data = normals.const_data_ptr<scalar_t>();
    const auto* normal_valid_data = normal_valid.const_data_ptr<bool>();
    const auto* index_data = indices.const_data_ptr<int64_t>();
    auto* spfh_data = spfh.mutable_data_ptr<scalar_t>();
    auto* spfh_valid_data = spfh_valid.mutable_data_ptr<bool>();
    parallel_for(0, count, kGrainSize, [&](int64_t begin, int64_t end) {
      for (int64_t query = begin; query < end; ++query) {
        spfh_valid_data[query] = fpfh_spfh_one(
            query, k, point_data, normal_data, normal_valid_data, index_data,
            spfh_data + query * kFpfhSize);
      }
    });
    const auto* distance_data = distances2.const_data_ptr<scalar_t>();
    auto* descriptor_data = descriptors.mutable_data_ptr<scalar_t>();
    auto* valid_data = valid.mutable_data_ptr<bool>();
    parallel_for(0, count, kGrainSize, [&](int64_t begin, int64_t end) {
      for (int64_t query = begin; query < end; ++query) {
        valid_data[query] = fpfh_descriptor_one(
            query, k, point_data, normal_valid_data, index_data, distance_data,
            spfh_data, spfh_valid_data, descriptor_data + query * kFpfhSize);
      }
    });
  }), AT_FLOATING_TYPES);
  return {descriptors, valid};
}

}  // namespace torchpcl
