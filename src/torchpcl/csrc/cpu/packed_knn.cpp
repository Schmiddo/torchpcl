#include "backend.h"
#include "common/candidate.h"

#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace torchpcl {

SearchResult packed_knn_cpu(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius) {
  TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
  const float max_distance2 = static_cast<float>(radius * radius);
  auto indices = at::empty(
      {queries.size(0), k}, queries.options().dtype(at::kInt));
  auto distances = at::empty({queries.size(0), k}, queries.options());
  const auto* point_data = points.data_ptr<float>();
  const auto* offsets = point_offsets.data_ptr<int64_t>();
  const auto* query_data = queries.data_ptr<float>();
  const auto* batches = query_batch_ids.data_ptr<int64_t>();
  auto* output_indices = indices.data_ptr<int32_t>();
  auto* output_distances = distances.data_ptr<float>();

  at::parallel_for(0, queries.size(0), 64, [&](int64_t begin, int64_t end) {
    for (int64_t query = begin; query < end; ++query) {
      float best_distances[kMaxK];
      int32_t best_indices[kMaxK];
      for (int64_t rank = 0; rank < k; ++rank) {
        best_distances[rank] = max_distance2;
        best_indices[rank] = -1;
      }
      const int64_t batch = batches[query];
      const int64_t point_begin = offsets[batch];
      const int64_t point_end = offsets[batch + 1];
      const float query_x = query_data[query * 3];
      const float query_y = query_data[query * 3 + 1];
      const float query_z = query_data[query * 3 + 2];
      for (int64_t point = point_begin; point < point_end; ++point) {
        const float dx = query_x - point_data[point * 3];
        const float dy = query_y - point_data[point * 3 + 1];
        const float dz = query_z - point_data[point * 3 + 2];
        const float distance2 = dx * dx + dy * dy + dz * dz;
        insert_candidate(
            best_distances,
            best_indices,
            static_cast<int>(k),
            distance2,
            static_cast<int32_t>(point));
      }
      for (int64_t rank = 0; rank < k; ++rank) {
        output_indices[query * k + rank] = best_indices[rank];
        output_distances[query * k + rank] = best_distances[rank];
      }
    }
  });
  return {indices, distances};
}

}  // namespace torchpcl
