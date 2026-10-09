#pragma once

#include "backend.h"


#include <cmath>
#include <cstdint>

namespace torchpcl {

inline void check_points(const Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.scalar_type() == kFloat, name, " must be float32");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(1) == 3,
      name,
      " must have shape (N, 3)");
}

inline void check_feature_pair(
    const Tensor& reference, const Tensor& queries) {
  STD_TORCH_CHECK(
      reference.scalar_type() == kFloat ||
          reference.scalar_type() == kDouble,
      "features must have dtype float32 or float64");
  STD_TORCH_CHECK(
      reference.scalar_type() == queries.scalar_type(),
      "reference and query features must have the same dtype");
  STD_TORCH_CHECK(
      reference.device() == queries.device(),
      "reference and query features must be on the same device");
  STD_TORCH_CHECK(
      reference.is_contiguous() && queries.is_contiguous(),
      "feature tensors must be contiguous");
  STD_TORCH_CHECK(
      reference.dim() == 2 && queries.dim() == 2 &&
          reference.size(1) == queries.size(1),
      "feature tensors must have shape (N, D) with matching D");
}

inline void check_knn_args(int64_t k, double radius) {
  STD_TORCH_CHECK(k >= 1 && k <= kMaxK, "k must be in [1, ", kMaxK, "]");
  STD_TORCH_CHECK(
      radius > 0.0 && !std::isnan(radius), "radius must be positive");
}

// The tensors `build_bvh` made of `points` (see BvhResult), and queries on
// their device.
inline void check_bvh(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries) {
  STD_TORCH_CHECK(
      nodes.scalar_type() == kByte && nodes.dim() == 2 &&
          nodes.size(1) == kBvhNodeBytes && nodes.is_contiguous(),
      "nodes must be a contiguous uint8 tensor of shape (M, ", kBvhNodeBytes, ")");
  STD_TORCH_CHECK(
      prim_ids.scalar_type() == kInt && prim_ids.dim() == 1 &&
          prim_ids.is_contiguous(),
      "prim_ids must be a contiguous int32 tensor of shape (P,)");
  STD_TORCH_CHECK(
      nodes.device() == points.device() &&
          prim_ids.device() == points.device() &&
          queries.device() == points.device(),
      "queries and the BVH must be on the device of the indexed points");
}

inline void check_packed_metadata(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids) {
  STD_TORCH_CHECK(
      point_offsets.scalar_type() == kLong,
      "point_offsets must be int64");
  STD_TORCH_CHECK(
      point_offsets.is_contiguous(), "point_offsets must be contiguous");
  STD_TORCH_CHECK(
      point_offsets.dim() == 1 && point_offsets.numel() >= 2,
      "point_offsets must have shape (B + 1,)");
  STD_TORCH_CHECK(
      query_batch_ids.scalar_type() == kLong,
      "query_batch_ids must be int64");
  STD_TORCH_CHECK(
      query_batch_ids.is_contiguous(), "query_batch_ids must be contiguous");
  STD_TORCH_CHECK(
      query_batch_ids.dim() == 1 &&
          query_batch_ids.numel() == queries.size(0),
      "query_batch_ids must have shape (M,)");
  STD_TORCH_CHECK(
      points.device() == queries.device() &&
          points.device() == point_offsets.device() &&
          points.device() == query_batch_ids.device(),
      "packed search tensors must be on the same device");
  STD_TORCH_CHECK(points.size(0) <= INT32_MAX, "points has too many rows");
}

}  // namespace torchpcl
