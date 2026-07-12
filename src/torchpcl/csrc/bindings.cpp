#include "backend.h"
#include "common/checks.h"

#include <torch/extension.h>

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
  module.def("has_cuda", &torchpcl::has_cuda);
}
