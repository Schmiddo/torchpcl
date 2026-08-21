#include "backend.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include <cstdint>
#include <limits>

namespace torchpcl {

SearchResult feature_knn1_cpu(
    const at::Tensor& reference, const at::Tensor& queries) {
  auto indices = at::empty({queries.size(0)}, queries.options().dtype(at::kLong));
  auto distances2 = at::empty({queries.size(0)}, queries.options());
  const int64_t reference_count = reference.size(0);
  const int64_t dimension = reference.size(1);
  auto* output_indices = indices.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(
      reference.scalar_type(), "feature_knn1_cpu", [&] {
        const auto* reference_data = reference.data_ptr<scalar_t>();
        const auto* query_data = queries.data_ptr<scalar_t>();
        auto* output_distances2 = distances2.data_ptr<scalar_t>();
        at::parallel_for(
            0, queries.size(0), 32, [&](int64_t begin, int64_t end) {
              for (int64_t query = begin; query < end; ++query) {
                scalar_t best_distance2 =
                    std::numeric_limits<scalar_t>::infinity();
                int64_t best_index = -1;
                const auto* query_row = query_data + query * dimension;
                for (int64_t candidate = 0; candidate < reference_count;
                     ++candidate) {
                  const auto* reference_row =
                      reference_data + candidate * dimension;
                  scalar_t distance2 = 0;
                  for (int64_t component = 0; component < dimension;
                       ++component) {
                    const scalar_t delta =
                        query_row[component] - reference_row[component];
                    distance2 += delta * delta;
                  }
                  if (distance2 < best_distance2 ||
                      (distance2 == best_distance2 &&
                       (best_index < 0 || candidate < best_index))) {
                    best_distance2 = distance2;
                    best_index = candidate;
                  }
                }
                output_indices[query] = best_index;
                output_distances2[query] = best_distance2;
              }
            });
      });
  return {indices, distances2};
}

}  // namespace torchpcl
