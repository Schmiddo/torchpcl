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
// single parallel pass, each thread's run reducing into its own slot;
// the point just merged in is forced to -infinity so it (and, by the same
// rule, every earlier selection) can never be picked again, even if the
// cloud contains exact duplicates.
at::Tensor farthest_point_sample_cpu(
    const at::Tensor& points, int64_t num_samples) {
  const int64_t count = points.size(0);
  auto selected = at::empty({num_samples}, points.options().dtype(at::kLong));
  auto* out = selected.data_ptr<int64_t>();
  out[0] = 0;

  constexpr int64_t kGrainSize = 4096;
  // One contiguous run of points per thread, as few as the grain allows.
  const int64_t runs = std::max<int64_t>(
      1,
      std::min<int64_t>(
          at::get_num_threads(), (count + kGrainSize - 1) / kGrainSize));

  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "farthest_point_sample_cpu", [&] {
        const auto* data = points.data_ptr<scalar_t>();
        const scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();
        std::vector<scalar_t> min_dist2(count, kInfinity);

        using Candidate = FarthestCandidate<scalar_t>;
        const Candidate identity{-kInfinity, count};
        std::vector<Candidate> partials(runs, identity);
        int64_t center = 0;
        for (int64_t sample = 1; sample < num_samples; ++sample) {
          at::parallel_for(0, runs, 1, [&](int64_t begin, int64_t end) {
            for (int64_t run = begin; run < end; ++run) {
              Candidate partial = identity;
              const int64_t last = count * (run + 1) / runs;
              for (int64_t i = count * run / runs; i < last; ++i) {
                const scalar_t updated = i == center
                    ? -kInfinity
                    : std::min(min_dist2[i], squared_distance(data, i, center));
                min_dist2[i] = updated;
                partial = farther_candidate(partial, Candidate{updated, i});
              }
              partials[run] = partial;
            }
          });
          Candidate best = identity;
          for (const Candidate& partial : partials) {
            best = farther_candidate(best, partial);
          }
          center = best.index;
          out[sample] = center;
        }
      });
  return selected;
}

}  // namespace torchpcl
