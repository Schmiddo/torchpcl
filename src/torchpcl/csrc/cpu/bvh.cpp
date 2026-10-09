#include "backend.h"
#include "common/bvh.h"
#include "common/distance.h"

#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"


#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torchpcl {
namespace {

// Whether this thread already runs inside an OpenMP parallel region, where a
// nested one would oversubscribe. torch's libgomp is the one loaded (same
// soname), so this sees torch's parallel_for regions too.
bool in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

// Subtrees larger than this are built as separate OpenMP tasks.
constexpr uint32_t kTaskPoints = 32768;

// A parallel version of cuBQL's CPU spatial-median builder, which is serial
// and was most of the index build. Same splits: each node is cut through the
// middle of its points' bounds along the widest axis, or through the middle
// of its point list when every point lands on one side. It writes cuBQL's
// BinaryBVH layout directly, so the queries are unchanged. The two children
// of a node are adjacent, allocated from an atomic counter so subtrees can be
// built concurrently, and a node's bounds are known once its children
// return, so there is no separate refit pass. This might be replaced if
// cuBQL gains a parallel CPU builder.
class Builder {
 public:
  Builder(const cuBQL::vec3f* points, uint32_t* ids, cuBQL::bvh3f::Node* nodes,
          uint32_t leaf_size)
      : points_(points), ids_(ids), nodes_(nodes), leaf_size_(leaf_size) {}

  uint64_t node_count() const { return next_.load(); }

  cuBQL::box3f build(uint64_t node, uint32_t begin, uint32_t end) {
    cuBQL::box3f bounds;
    for (uint32_t i = begin; i < end; ++i) bounds.extend(points_[ids_[i]]);
    if (end - begin <= leaf_size_) return make_leaf(node, begin, end, bounds);

    const int axis = cuBQL::arg_max(bounds.size());
    const float split = bounds.center()[axis];
    uint32_t mid = static_cast<uint32_t>(
        std::partition(ids_ + begin, ids_ + end, [&](uint32_t id) {
          return points_[id][axis] < split;
        }) - ids_);
    if (mid == begin || mid == end) mid = begin + (end - begin) / 2;

    const uint64_t child = next_.fetch_add(2);
    nodes_[node].admin.offset = child;
    nodes_[node].admin.count = 0;
    cuBQL::box3f left;
    cuBQL::box3f right;
    if (end - begin > kTaskPoints) {
#pragma omp task shared(left)
      left = build(child, begin, mid);
      right = build(child + 1, mid, end);
#pragma omp taskwait
    } else {
      left = build(child, begin, mid);
      right = build(child + 1, mid, end);
    }
    nodes_[node].bounds = left.including(right);
    return nodes_[node].bounds;
  }

 private:
  cuBQL::box3f make_leaf(
      uint64_t node, uint32_t begin, uint32_t end, const cuBQL::box3f& bounds) {
    nodes_[node].admin.offset = begin;
    nodes_[node].admin.count = end - begin;
    nodes_[node].bounds = bounds;
    return bounds;
  }

  const cuBQL::vec3f* points_;
  uint32_t* ids_;
  cuBQL::bvh3f::Node* nodes_;
  uint32_t leaf_size_;
  std::atomic<uint64_t> next_{1};
};

}  // namespace

BvhResult build_bvh_cpu(const Tensor& points) {
  const auto* point_data =
      reinterpret_cast<const cuBQL::vec3f*>(points.const_data_ptr<float>());
  const auto count = static_cast<uint32_t>(points.size(0));
  // Like cuBQL's builder, leave out points whose box is empty (NaN).
  auto prim_ids = torch::stable::new_empty(points, {count}, kInt);
  auto* ids = reinterpret_cast<uint32_t*>(prim_ids.mutable_data_ptr<int32_t>());
  uint32_t kept = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (!cuBQL::box3f().including(point_data[i]).empty()) ids[kept++] = i;
  }
  // cuBQL reads a node with no points as an inner node, so a tree over no
  // points cannot be traversed; `bvh_knn_cpu` answers that case without one.
  if (kept == 0) {
    return {torch::stable::new_empty(points, {0, kBvhNodeBytes}, kByte),
            torch::stable::narrow(prim_ids, 0, 0, 0)};
  }

  // A binary tree over n points has at most 2n - 1 nodes.
  auto nodes = torch::stable::new_empty(
      points, {2 * int64_t{kept} - 1, kBvhNodeBytes}, kByte);
  Builder builder(
      point_data, ids,
      reinterpret_cast<cuBQL::bvh3f::Node*>(nodes.mutable_data_ptr<uint8_t>()),
      kCpuBvhLeafSize);
  if (kept <= kTaskPoints || in_parallel_region()) {
    builder.build(0, 0, kept);
  } else {
#pragma omp parallel num_threads(get_num_threads())
#pragma omp single
    builder.build(0, 0, kept);
  }

  // Shrink to the nodes actually used.
  const auto used = static_cast<int64_t>(builder.node_count());
  return {torch::stable::clone(torch::stable::narrow(nodes, 0, 0, used)),
          torch::stable::narrow(prim_ids, 0, 0, kept)};
}

SearchResult bvh_knn_cpu(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries,
    int64_t k,
    double radius) {
  const auto count = queries.size(0);
  auto indices = torch::stable::new_empty(queries, {count, k}, kLong);
  auto distances = torch::stable::new_empty(queries, {count, k});
  if (prim_ids.numel() == 0) {  // every reference point was NaN
    torch::stable::fill_(indices, -1);
    torch::stable::fill_(distances, std::numeric_limits<float>::infinity());
    return {indices, distances};
  }
  const cuBQL::bvh3f bvh = bvh_view(nodes, prim_ids);
  const float bound2 = inclusive_bound2(radius);
  const float cull2 = cull_bound2(bound2);
  const auto* point_data =
      reinterpret_cast<const cuBQL::vec3f*>(points.const_data_ptr<float>());
  const auto* query_data =
      reinterpret_cast<const cuBQL::vec3f*>(queries.const_data_ptr<float>());
  auto* output_indices = indices.mutable_data_ptr<int64_t>();
  auto* output_distances = distances.mutable_data_ptr<float>();
  // One output slot: the candidate and its exact distance, or empty.
  const auto write = [&](int64_t slot, int index, cuBQL::vec3f query) {
    const float distance2 = index >= 0
        ? exact_distance2(point_data[index], query)
        : std::numeric_limits<float>::infinity();
    const bool kept = index >= 0 && distance2 <= bound2;
    output_indices[slot] = kept ? index : -1;
    output_distances[slot] =
        kept ? distance2 : std::numeric_limits<float>::infinity();
  };

  if (k == 1) {
    parallel_for(0, count, 512, [&](int64_t begin, int64_t end) {
      for (int64_t query = begin; query < end; ++query) {
        write(query,
              cuBQL::points::findClosest(
                  bvh, point_data, query_data[query], cull2),
              query_data[query]);
      }
    });
    return {indices, distances};
  }

  parallel_for(0, count, 128, [&](int64_t begin, int64_t end) {
    cuBQL::knn::Candidate candidates[kMaxK];
    for (int64_t query = begin; query < end; ++query) {
      const auto result = cuBQL::points::findKNN(
          candidates,
          static_cast<int>(k),
          bvh,
          point_data,
          query_data[query],
          cull2);
      for (int64_t rank = 0; rank < k; ++rank) {
        write(query * k + rank,
              rank < result.numFound ? candidates[rank].primID : -1,
              query_data[query]);
      }
    }
  });
  return {indices, distances};
}

}  // namespace torchpcl
