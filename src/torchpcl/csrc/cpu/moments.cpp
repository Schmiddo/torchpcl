#include "backend.h"
#include "common/moments.h"


namespace torchpcl {

MomentsResult neighbor_moments_cpu(
    const Tensor& points, const Tensor& indices) {
  const int64_t count = indices.size(0);
  const int64_t k = indices.size(1);
  auto means = torch::stable::new_empty(points, {count, 3});
  auto scatter = torch::stable::new_empty(points, {count, 3, 3});
  auto counts = torch::stable::new_empty(indices, {count});
  THO_DISPATCH_V2(points.scalar_type(), "neighbor_moments_cpu", ([&] {
    const auto* point_data = points.const_data_ptr<scalar_t>();
    const auto* index_data = indices.const_data_ptr<int64_t>();
    auto* mean_data = means.mutable_data_ptr<scalar_t>();
    auto* scatter_data = scatter.mutable_data_ptr<scalar_t>();
    auto* count_data = counts.mutable_data_ptr<int64_t>();
    parallel_for(0, count, 1024, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        count_data[row] = neighbor_moments_one(
            point_data, index_data + row * k, k, mean_data + row * 3,
            scatter_data + row * 9);
      }
    });
  }), AT_FLOATING_TYPES);
  return {means, scatter, counts};
}

}  // namespace torchpcl
