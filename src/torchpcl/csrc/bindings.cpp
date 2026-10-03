#include "backend.h"
#include "common/checks.h"

#include <torch/csrc/utils/pybind.h>

namespace torchpcl {

BvhIndex::BvhIndex(const at::Tensor& points) {
  check_points(points, "points");
  TORCH_CHECK(points.size(0) > 0, "points must be non-empty");
  TORCH_CHECK(points.size(0) <= INT32_MAX, "points has too many rows");
  if (points.device().is_cpu()) {
    impl_ = make_cpu_bvh(points);
    return;
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    impl_ = make_cuda_bvh(points);
    return;
#else
    TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  TORCH_CHECK(false, "unsupported points device: ", points.device());
}

SearchResult BvhIndex::knn(
    const at::Tensor& queries, int64_t k, double radius) {
  check_knn_args(k, radius);
  return impl_->knn(queries, k, radius);
}

SearchResult packed_knn(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius) {
  check_points(points, "points");
  check_points(queries, "queries");
  check_knn_args(k, radius);
  check_packed_metadata(points, point_offsets, queries, query_batch_ids);
  if (points.device().is_cpu()) {
    return packed_knn_cpu(
        points, point_offsets, queries, query_batch_ids, k, radius);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return packed_knn_cuda(
        points, point_offsets, queries, query_batch_ids, k, radius);
#else
    TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  TORCH_CHECK(false, "unsupported points device: ", points.device());
}

SearchResult feature_knn1(
    const at::Tensor& reference, const at::Tensor& queries) {
  check_feature_pair(reference, queries);
  if (reference.device().is_cpu()) {
    return feature_knn1_cpu(reference, queries);
  }
  if (reference.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return feature_knn1_cuda(reference, queries);
#else
    TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  TORCH_CHECK(false, "unsupported feature device: ", reference.device());
}

MutualSearchResult feature_mutual_knn1(
    const at::Tensor& source, const at::Tensor& target) {
  check_feature_pair(target, source);
  if (source.device().is_cpu()) {
    return feature_mutual_knn1_cpu(source, target);
  }
  if (source.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    // Each direction is one kernel launch; sharing distances would need a
    // cross-thread column reduction that costs more than it saves.
    auto [indices, distances2] = feature_knn1_cuda(target, source);
    auto [reverse_indices, reverse_distances2] = feature_knn1_cuda(source, target);
    return {indices, distances2, reverse_indices, reverse_distances2};
#else
    TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  TORCH_CHECK(false, "unsupported feature device: ", source.device());
}

EigenResult symmetric_eigh_3x3(const at::Tensor& matrices) {
  TORCH_CHECK(
      matrices.scalar_type() == at::kFloat ||
          matrices.scalar_type() == at::kDouble,
      "matrices must be float32 or float64");
  TORCH_CHECK(matrices.is_contiguous(), "matrices must be contiguous");
  TORCH_CHECK(
      matrices.dim() >= 2 && matrices.size(-2) == 3 &&
          matrices.size(-1) == 3,
      "matrices must have shape (..., 3, 3)");
  if (matrices.device().is_cpu()) {
    return symmetric_eigh_3x3_cpu(matrices);
  }
  if (matrices.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return symmetric_eigh_3x3_cuda(matrices);
#else
    TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  TORCH_CHECK(false, "unsupported matrices device: ", matrices.device());
}

at::Tensor farthest_point_sample(
    const at::Tensor& points, int64_t num_samples) {
  TORCH_CHECK(
      points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble,
      "points must be float32 or float64");
  TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(
      points.dim() == 2 && points.size(1) == 3,
      "points must have shape (N, 3)");
  TORCH_CHECK(
      num_samples >= 1 && num_samples < points.size(0),
      "num_samples must be in [1, N)");
  if (points.device().is_cpu()) {
    return farthest_point_sample_cpu(points, num_samples);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return farthest_point_sample_cuda(points, num_samples);
#else
    TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  TORCH_CHECK(false, "unsupported points device: ", points.device());
}

bool has_cuda() {
#ifdef TORCHPCL_WITH_CUDA
  return true;
#else
  return false;
#endif
}

}  // namespace torchpcl

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<torchpcl::BvhIndex>(module, "BvhIndex")
      .def(py::init<const at::Tensor&>())
      .def("knn", &torchpcl::BvhIndex::knn);
  module.def("packed_knn", &torchpcl::packed_knn);
  module.def("feature_knn1", &torchpcl::feature_knn1);
  module.def("feature_mutual_knn1", &torchpcl::feature_mutual_knn1);
  module.def("symmetric_eigh_3x3", &torchpcl::symmetric_eigh_3x3);
  module.def("farthest_point_sample", &torchpcl::farthest_point_sample);
  module.def("has_cuda", &torchpcl::has_cuda);
}
