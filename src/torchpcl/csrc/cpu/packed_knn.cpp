#include "backend.h"
#include "common/candidate.h"
#include "common/distance.h"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace torchpcl {

SearchResult packed_knn_cpu(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids,
    int64_t k,
    double radius) {
  STD_TORCH_CHECK(points.is_cpu(), "points must be a CPU tensor");
  // An empty slot accepts a point exactly at the bound, so the radius is
  // inclusive (see better_candidate).
  const float bound2 = inclusive_bound2(radius);
  auto indices = torch::stable::new_empty(queries, {queries.size(0), k}, kLong);
  auto distances = torch::stable::new_empty(queries, {queries.size(0), k});
  const auto* point_data = points.const_data_ptr<float>();
  const auto* offsets = point_offsets.const_data_ptr<int64_t>();
  const auto* query_data = queries.const_data_ptr<float>();
  const auto* batches = query_batch_ids.const_data_ptr<int64_t>();
  auto* output_indices = indices.mutable_data_ptr<int64_t>();
  auto* output_distances = distances.mutable_data_ptr<float>();

  parallel_for(0, queries.size(0), 64, [&](int64_t begin, int64_t end) {
    for (int64_t query = begin; query < end; ++query) {
      float best_distances[kMaxK];
      int32_t best_indices[kMaxK];
      for (int64_t rank = 0; rank < k; ++rank) {
        best_distances[rank] = bound2;
        best_indices[rank] = -1;
      }
      const int64_t batch = batches[query];
      const int64_t point_begin = offsets[batch];
      const int64_t point_end = offsets[batch + 1];
      const float query_x = query_data[query * 3];
      const float query_y = query_data[query * 3 + 1];
      const float query_z = query_data[query * 3 + 2];
      for (int64_t point = point_begin; point < point_end; ++point) {
        const float distance2 = exact_distance2(
            point_data[point * 3], point_data[point * 3 + 1],
            point_data[point * 3 + 2], query_x, query_y, query_z);
        insert_candidate(
            best_distances,
            best_indices,
            static_cast<int>(k),
            distance2,
            static_cast<int32_t>(point));
      }
      for (int64_t rank = 0; rank < k; ++rank) {
        const bool found = best_indices[rank] >= 0;
        output_indices[query * k + rank] = best_indices[rank];
        output_distances[query * k + rank] = found
            ? best_distances[rank]
            : std::numeric_limits<float>::infinity();
      }
    }
  });
  return {indices, distances};
}

}  // namespace torchpcl
