#include "backend.h"

#include <ATen/ops/argmin.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/square.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace torchpcl {

SearchResult feature_knn1_cpu(
    const at::Tensor& reference, const at::Tensor& queries) {
  auto indices = at::empty({queries.size(0)}, queries.options().dtype(at::kLong));
  auto distances2 = at::empty({queries.size(0)}, queries.options());
  const int64_t reference_count = reference.size(0);
  if (reference_count == 0) {
    indices.fill_(-1);
    distances2.fill_(std::numeric_limits<double>::infinity());
    return {indices, distances2};
  }

  // Use bounded query blocks so matrix multiplication can reuse reference data
  // and dispatch to PyTorch's optimized BLAS backend without materializing the
  // complete Q x R distance matrix.
  constexpr int64_t kTemporaryBytes = 64 * 1024 * 1024;
  const int64_t max_entries = kTemporaryBytes / reference.element_size();
  const int64_t block_rows = std::max<int64_t>(
      1,
      std::min<int64_t>(queries.size(0), max_entries / reference_count));
  const auto reference_norms = at::sum(at::square(reference), {1});
  const auto reference_t = reference.transpose(0, 1);

  for (int64_t begin = 0; begin < queries.size(0); begin += block_rows) {
    const int64_t end = std::min(begin + block_rows, queries.size(0));
    const auto query_block = queries.slice(0, begin, end);
    auto pairwise = at::matmul(query_block, reference_t);
    pairwise.mul_(-2).add_(reference_norms);
    pairwise.add_(at::sum(at::square(query_block), {1}, true));
    const auto block_indices = at::argmin(pairwise, 1);
    indices.slice(0, begin, end).copy_(block_indices);

    // Recompute selected distances directly rather than returning cancellation
    // error from ||q||² + ||r||² - 2 q·r.
    const auto selected = at::index_select(reference, 0, block_indices);
    const auto selected_distances =
        at::sum(at::square(at::sub(query_block, selected)), {1});
    distances2.slice(0, begin, end).copy_(selected_distances);
  }
  return {indices, distances2};
}

}  // namespace torchpcl
