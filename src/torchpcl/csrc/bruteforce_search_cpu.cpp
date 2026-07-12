#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int64_t kMaxK = 64;
constexpr int64_t kPointTile = 512;

void check_offsets(const at::Tensor& offsets, int64_t size, const char* name) {
  TORCH_CHECK(offsets.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(offsets.scalar_type() == at::kLong, name, " must be int64");
  TORCH_CHECK(offsets.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() >= 2,
              name, " must have shape (B + 1,)");
  const auto* values = offsets.data_ptr<int64_t>();
  TORCH_CHECK(values[0] == 0, name, " must start at zero");
  for (int64_t i = 1; i < offsets.numel(); ++i) {
    TORCH_CHECK(values[i] >= values[i - 1], name, " must be nondecreasing");
  }
  TORCH_CHECK(values[offsets.numel() - 1] == size,
              name, " must end at the tensor row count");
}

void check_points(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == 3,
              name, " must have shape (N, 3)");
}

inline bool better(float d2, int32_t index, float other_d2, int32_t other_index) {
  return d2 < other_d2 ||
      (d2 == other_d2 && (other_index < 0 || index < other_index));
}

std::tuple<at::Tensor, at::Tensor> knn(
    at::Tensor points,
    at::Tensor point_offsets,
    at::Tensor queries,
    at::Tensor query_offsets,
    int64_t k,
    double radius) {
  check_points(points, "points");
  check_points(queries, "queries");
  check_offsets(point_offsets, points.size(0), "point_offsets");
  check_offsets(query_offsets, queries.size(0), "query_offsets");
  TORCH_CHECK(point_offsets.numel() == query_offsets.numel(),
              "point_offsets and query_offsets must describe the same batch count");
  TORCH_CHECK(k >= 1 && k <= kMaxK, "k must be in [1, ", kMaxK, "]");
  TORCH_CHECK(radius > 0.0 && !std::isnan(radius), "radius must be positive");
  TORCH_CHECK(points.size(0) <= INT32_MAX, "points has too many rows");

  const float max_d2 = static_cast<float>(radius * radius);
  auto indices = at::empty({queries.size(0), k}, queries.options().dtype(at::kInt));
  auto distances = at::empty({queries.size(0), k}, queries.options());
  const auto* pts = points.data_ptr<float>();
  const auto* qs = queries.data_ptr<float>();
  const auto* po = point_offsets.data_ptr<int64_t>();
  const auto* qo = query_offsets.data_ptr<int64_t>();
  auto* out_idx = indices.data_ptr<int32_t>();
  auto* out_d2 = distances.data_ptr<float>();
  const int64_t batches = point_offsets.numel() - 1;

  for (int64_t batch = 0; batch < batches; ++batch) {
    TORCH_CHECK(po[batch] < po[batch + 1] || qo[batch] == qo[batch + 1],
                "a batch with queries must contain at least one point");
    at::parallel_for(qo[batch], qo[batch + 1], 64, [&](int64_t begin, int64_t end) {
      for (int64_t qi = begin; qi < end; ++qi) {
        float best_d2[kMaxK];
        int32_t best_idx[kMaxK];
        for (int64_t rank = 0; rank < k; ++rank) {
          best_d2[rank] = max_d2;
          best_idx[rank] = -1;
        }
        const float qx = qs[qi * 3];
        const float qy = qs[qi * 3 + 1];
        const float qz = qs[qi * 3 + 2];
        for (int64_t tile = po[batch]; tile < po[batch + 1]; tile += kPointTile) {
          const int64_t tile_end = std::min(tile + kPointTile, po[batch + 1]);
          for (int64_t pi = tile; pi < tile_end; ++pi) {
            const float dx = qx - pts[pi * 3];
            const float dy = qy - pts[pi * 3 + 1];
            const float dz = qz - pts[pi * 3 + 2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            const auto index = static_cast<int32_t>(pi);
            if (!better(d2, index, best_d2[k - 1], best_idx[k - 1])) continue;
            int64_t rank = k - 1;
            while (rank > 0 && better(d2, index, best_d2[rank - 1], best_idx[rank - 1])) {
              best_d2[rank] = best_d2[rank - 1];
              best_idx[rank] = best_idx[rank - 1];
              --rank;
            }
            best_d2[rank] = d2;
            best_idx[rank] = index;
          }
        }
        for (int64_t rank = 0; rank < k; ++rank) {
          out_idx[qi * k + rank] = best_idx[rank];
          out_d2[qi * k + rank] = best_d2[rank];
        }
      }
    });
  }
  return {indices, distances};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("knn", &knn); }
