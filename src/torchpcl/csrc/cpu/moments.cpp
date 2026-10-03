#include "backend.h"
#include "common/moments.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>

namespace torchpcl {

MomentsResult neighbor_moments_cpu(
    const at::Tensor& points, const at::Tensor& indices) {
  const int64_t count = indices.size(0);
  const int64_t k = indices.size(1);
  auto means = at::empty({count, 3}, points.options());
  auto scatter = at::empty({count, 3, 3}, points.options());
  auto counts = at::empty({count}, indices.options());
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "neighbor_moments_cpu", [&] {
    const auto* point_data = points.data_ptr<scalar_t>();
    const auto* index_data = indices.data_ptr<int64_t>();
    auto* mean_data = means.data_ptr<scalar_t>();
    auto* scatter_data = scatter.data_ptr<scalar_t>();
    auto* count_data = counts.data_ptr<int64_t>();
    at::parallel_for(0, count, 1024, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        count_data[row] = neighbor_moments_one(
            point_data, index_data + row * k, k, mean_data + row * 3,
            scatter_data + row * 9);
      }
    });
  });
  return {means, scatter, counts};
}

}  // namespace torchpcl
