#include "backend.h"


#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

// The distance loop is compiled twice, for AVX2 and for the baseline ISA the
// extension targets, and picked at load time.
#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)
#define TORCHPCL_TARGET_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define TORCHPCL_TARGET_CLONES
#endif
#if defined(__GNUC__)
#define TORCHPCL_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define TORCHPCL_ALWAYS_INLINE inline
#endif

namespace torchpcl {
namespace {

// Queries share each loaded reference chunk; a chunk is one vector register
// pair of references, which a query's accumulators span.
constexpr int64_t kQueries = 4;

template <typename scalar_t>
constexpr int64_t chunk_width() {
  return 64 / sizeof(scalar_t);
}

// Reference rows rearranged chunk by chunk, each chunk dimension-major:
// chunk c holds `dimension` runs of chunk_width() values, so the distance
// loop reads contiguous vectors. Rows past the end are zero and are never
// selected.
template <typename scalar_t>
std::vector<scalar_t> pack_reference(
    const scalar_t* reference, int64_t count, int64_t dimension) {
  constexpr int64_t width = chunk_width<scalar_t>();
  const int64_t chunks = (count + width - 1) / width;
  std::vector<scalar_t> packed(chunks * dimension * width, scalar_t(0));
  parallel_for(0, chunks, 64, [&](int64_t begin, int64_t end) {
    for (int64_t chunk = begin; chunk < end; ++chunk) {
      scalar_t* out = packed.data() + chunk * dimension * width;
      for (int64_t lane = 0; lane < width; ++lane) {
        const int64_t row = chunk * width + lane;
        if (row >= count) break;
        for (int64_t component = 0; component < dimension; ++component) {
          out[component * width + lane] = reference[row * dimension + component];
        }
      }
    }
  });
  return packed;
}

// Nearest reference row of `query_count` (at most kQueries) queries, the
// first of which is query row `first_query`; ties keep the lower reference
// index. With `column_best` and `column_index` (one entry per reference row)
// it also lowers each reference's running nearest query, keeping the lower
// query index on ties as long as queries arrive in increasing order.
template <typename scalar_t>
TORCHPCL_ALWAYS_INLINE void nearest_rows_impl(
    const scalar_t* packed,
    int64_t reference_count,
    const scalar_t* queries,
    int64_t first_query,
    int64_t query_count,
    int64_t dimension,
    int64_t* best_indices,
    scalar_t* best_distances2,
    scalar_t* column_best,
    int64_t* column_index) {
  constexpr int64_t width = chunk_width<scalar_t>();
  const int64_t chunks = (reference_count + width - 1) / width;
  scalar_t best[kQueries];
  int64_t best_index[kQueries];
  for (int64_t q = 0; q < kQueries; ++q) {
    best[q] = std::numeric_limits<scalar_t>::infinity();
    best_index[q] = -1;
  }
  const scalar_t* rows[kQueries];
  for (int64_t q = 0; q < kQueries; ++q) {
    // Spare slots repeat the last query and are discarded.
    rows[q] = queries + std::min(q, query_count - 1) * dimension;
  }
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    scalar_t accumulators[kQueries][width] = {};
    const scalar_t* block = packed + chunk * dimension * width;
    for (int64_t component = 0; component < dimension; ++component) {
      const scalar_t* references = block + component * width;
      for (int64_t q = 0; q < kQueries; ++q) {
        const scalar_t value = rows[q][component];
        for (int64_t lane = 0; lane < width; ++lane) {
          const scalar_t delta = value - references[lane];
          accumulators[q][lane] += delta * delta;
        }
      }
    }
    const int64_t valid = std::min(width, reference_count - chunk * width);
    for (int64_t q = 0; q < kQueries; ++q) {
      for (int64_t lane = 0; lane < valid; ++lane) {
        if (accumulators[q][lane] < best[q]) {
          best[q] = accumulators[q][lane];
          best_index[q] = chunk * width + lane;
        }
      }
    }
    if (column_best != nullptr) {
      for (int64_t lane = 0; lane < valid; ++lane) {
        const int64_t reference = chunk * width + lane;
        for (int64_t q = 0; q < query_count; ++q) {
          if (accumulators[q][lane] < column_best[reference]) {
            column_best[reference] = accumulators[q][lane];
            column_index[reference] = first_query + q;
          }
        }
      }
    }
  }
  for (int64_t q = 0; q < query_count; ++q) {
    best_indices[q] = best_index[q];
    best_distances2[q] = best[q];
  }
}

// target_clones is applied to these non-template overloads, which each
// compile their inlined copy of the loop once per target.
TORCHPCL_TARGET_CLONES void nearest_rows(
    const float* packed, int64_t reference_count, const float* queries,
    int64_t first_query, int64_t query_count, int64_t dimension,
    int64_t* best_indices, float* best_distances2, float* column_best,
    int64_t* column_index) {
  nearest_rows_impl(packed, reference_count, queries, first_query, query_count,
                    dimension, best_indices, best_distances2, column_best,
                    column_index);
}

TORCHPCL_TARGET_CLONES void nearest_rows(
    const double* packed, int64_t reference_count, const double* queries,
    int64_t first_query, int64_t query_count, int64_t dimension,
    int64_t* best_indices, double* best_distances2, double* column_best,
    int64_t* column_index) {
  nearest_rows_impl(packed, reference_count, queries, first_query, query_count,
                    dimension, best_indices, best_distances2, column_best,
                    column_index);
}

// Nearest reference row of every query, and with `reverse_indices` also the
// nearest query row of every reference, from the same distances. The query
// groups are split into one contiguous run per thread; each run keeps its own
// reference minima, merged in run order afterwards. Runs cover increasing
// query ranges, so ties keep the lower index both ways.
void nearest_cpu(
    const Tensor& reference,
    const Tensor& queries,
    Tensor& indices,
    Tensor& distances2,
    Tensor* reverse_indices,
    Tensor* reverse_distances2) {
  const int64_t reference_count = reference.size(0);
  const int64_t query_count = queries.size(0);
  const int64_t dimension = reference.size(1);
  THO_DISPATCH_V2(reference.scalar_type(), "feature_knn1_cpu", ([&] {
    const auto packed = pack_reference(
        reference.const_data_ptr<scalar_t>(), reference_count, dimension);
    const auto* query_data = queries.const_data_ptr<scalar_t>();
    auto* index_data = indices.mutable_data_ptr<int64_t>();
    auto* distance_data = distances2.mutable_data_ptr<scalar_t>();
    const int64_t groups = (query_count + kQueries - 1) / kQueries;
    const bool reverse = reverse_indices != nullptr;
    const int64_t runs = std::min<int64_t>(get_num_threads(), groups);
    const int64_t buffers = reverse ? runs : 0;
    std::vector<scalar_t> column_best(
        buffers * reference_count, std::numeric_limits<scalar_t>::infinity());
    std::vector<int64_t> column_index(buffers * reference_count, -1);
    parallel_for(0, runs, 1, [&](int64_t begin, int64_t end) {
      for (int64_t run = begin; run < end; ++run) {
        const int64_t offset = run * reference_count;
        const int64_t last = groups * (run + 1) / runs;
        for (int64_t group = groups * run / runs; group < last; ++group) {
          const int64_t first = group * kQueries;
          nearest_rows(
              packed.data(),
              reference_count,
              query_data + first * dimension,
              first,
              std::min(kQueries, query_count - first),
              dimension,
              index_data + first,
              distance_data + first,
              reverse ? column_best.data() + offset : nullptr,
              reverse ? column_index.data() + offset : nullptr);
        }
      }
    });
    if (!reverse) return;
    auto* reverse_index_data = reverse_indices->mutable_data_ptr<int64_t>();
    auto* reverse_distance_data =
        reverse_distances2->mutable_data_ptr<scalar_t>();
    parallel_for(0, reference_count, 4096, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        scalar_t best = std::numeric_limits<scalar_t>::infinity();
        int64_t best_index = -1;
        for (int64_t run = 0; run < runs; ++run) {
          const scalar_t candidate = column_best[run * reference_count + row];
          const int64_t index = column_index[run * reference_count + row];
          if (index >= 0 &&
              (candidate < best || (candidate == best && index < best_index))) {
            best = candidate;
            best_index = index;
          }
        }
        reverse_index_data[row] = best_index;
        reverse_distance_data[row] = best;
      }
    });
  }), AT_FLOATING_TYPES);
}

}  // namespace

SearchResult feature_knn1_cpu(
    const Tensor& reference, const Tensor& queries) {
  auto indices = torch::stable::new_empty(queries, {queries.size(0)}, kLong);
  auto distances2 = torch::stable::new_empty(queries, {queries.size(0)});
  if (reference.size(0) == 0) {
    torch::stable::fill_(indices, -1);
    torch::stable::fill_(distances2, std::numeric_limits<double>::infinity());
    return {indices, distances2};
  }
  nearest_cpu(reference, queries, indices, distances2, nullptr, nullptr);
  return {indices, distances2};
}

MutualSearchResult feature_mutual_knn1_cpu(
    const Tensor& source, const Tensor& target) {
  auto indices = torch::stable::new_empty(source, {source.size(0)}, kLong);
  auto distances2 = torch::stable::new_empty(source, {source.size(0)});
  auto reverse_indices =
      torch::stable::new_empty(target, {target.size(0)}, kLong);
  auto reverse_distances2 = torch::stable::new_empty(target, {target.size(0)});
  if (source.size(0) == 0 || target.size(0) == 0) {
    for (auto* tensor : {&indices, &reverse_indices}) {
      torch::stable::fill_(*tensor, -1);
    }
    for (auto* tensor : {&distances2, &reverse_distances2}) {
      torch::stable::fill_(*tensor, std::numeric_limits<double>::infinity());
    }
    return {indices, distances2, reverse_indices, reverse_distances2};
  }
  nearest_cpu(
      target, source, indices, distances2, &reverse_indices, &reverse_distances2);
  return {indices, distances2, reverse_indices, reverse_distances2};
}

}  // namespace torchpcl
