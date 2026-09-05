#include "backend.h"
#include "common/farthest_point.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace torchpcl {

// Runs the entire greedy loop in one call, so Python pays no per-iteration
// dispatch overhead. Each iteration merges the previously selected center's
// distances into `min_dist2` and reduces it to the next farthest point in a
// single `parallel_reduce` pass; the point just merged in is forced to
// -infinity so it (and, by the same rule, every earlier selection) can never
// be picked again, even if the cloud contains exact duplicates.
at::Tensor farthest_point_sample_cpu(
    const at::Tensor& points, int64_t num_samples) {
  const int64_t count = points.size(0);
  auto selected = at::empty({num_samples}, points.options().dtype(at::kLong));
  auto* out = selected.data_ptr<int64_t>();
  out[0] = 0;

  constexpr int64_t kGrainSize = 4096;

  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "farthest_point_sample_cpu", [&] {
        const auto* data = points.data_ptr<scalar_t>();
        const scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();
        std::vector<scalar_t> min_dist2(count, kInfinity);

        using Candidate = FarthestCandidate<scalar_t>;
        int64_t center = 0;
        for (int64_t sample = 1; sample < num_samples; ++sample) {
          const Candidate identity{-kInfinity, count};
          const Candidate best = at::parallel_reduce(
              0, count, kGrainSize, identity,
              [&](int64_t begin, int64_t end, Candidate partial) {
                for (int64_t i = begin; i < end; ++i) {
                  const scalar_t updated = i == center
                      ? -kInfinity
                      : std::min(min_dist2[i], squared_distance(data, i, center));
                  min_dist2[i] = updated;
                  partial = farther_candidate(partial, Candidate{updated, i});
                }
                return partial;
              },
              farther_candidate<scalar_t>);
          center = best.index;
          out[sample] = center;
        }
      });
  return selected;
}

}  // namespace torchpcl
