#pragma once

#include "backend.h"

#include <c10/util/Exception.h>

#include <cmath>
#include <cstdint>

namespace torchpcl {

inline void check_points(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(1) == 3,
      name,
      " must have shape (N, 3)");
}

inline void check_feature_pair(
    const at::Tensor& reference, const at::Tensor& queries) {
  TORCH_CHECK(
      reference.scalar_type() == at::kFloat ||
          reference.scalar_type() == at::kDouble,
      "features must have dtype float32 or float64");
  TORCH_CHECK(
      reference.scalar_type() == queries.scalar_type(),
      "reference and query features must have the same dtype");
  TORCH_CHECK(
      reference.device() == queries.device(),
      "reference and query features must be on the same device");
  TORCH_CHECK(
      reference.is_contiguous() && queries.is_contiguous(),
      "feature tensors must be contiguous");
  TORCH_CHECK(
      reference.dim() == 2 && queries.dim() == 2 &&
          reference.size(1) == queries.size(1),
      "feature tensors must have shape (N, D) with matching D");
}

inline void check_knn_args(int64_t k, double radius) {
  TORCH_CHECK(k >= 1 && k <= kMaxK, "k must be in [1, ", kMaxK, "]");
  TORCH_CHECK(radius > 0.0 && !std::isnan(radius), "radius must be positive");
}

inline void check_packed_metadata(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids) {
  TORCH_CHECK(
      point_offsets.scalar_type() == at::kLong,
      "point_offsets must be int64");
  TORCH_CHECK(point_offsets.is_contiguous(), "point_offsets must be contiguous");
  TORCH_CHECK(
      point_offsets.dim() == 1 && point_offsets.numel() >= 2,
      "point_offsets must have shape (B + 1,)");
  TORCH_CHECK(
      query_batch_ids.scalar_type() == at::kLong,
      "query_batch_ids must be int64");
  TORCH_CHECK(
      query_batch_ids.is_contiguous(), "query_batch_ids must be contiguous");
  TORCH_CHECK(
      query_batch_ids.dim() == 1 &&
          query_batch_ids.numel() == queries.size(0),
      "query_batch_ids must have shape (M,)");
  TORCH_CHECK(
      points.device() == queries.device() &&
          points.device() == point_offsets.device() &&
          points.device() == query_batch_ids.device(),
      "packed search tensors must be on the same device");
  TORCH_CHECK(points.size(0) <= INT32_MAX, "points has too many rows");
}

}  // namespace torchpcl
