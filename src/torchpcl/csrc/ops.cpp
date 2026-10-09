// The torchpcl op library: schemas, argument checks and device dispatch.
// The ops reach Python through the dispatcher (torch.ops.torchpcl); the
// module itself is empty and only exists so that importing it loads them.
#include <Python.h>

#include "backend.h"
#include "common/checks.h"


namespace torchpcl {

BvhResult build_bvh(const Tensor& points) {
  check_points(points, "points");
  STD_TORCH_CHECK(points.size(0) > 0, "points must be non-empty");
  STD_TORCH_CHECK(points.size(0) <= INT32_MAX, "points has too many rows");
  if (points.is_cpu()) {
    return build_bvh_cpu(points);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return build_bvh_cuda(points);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "points must be on the CPU or a CUDA device");
}

SearchResult bvh_knn(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries,
    int64_t k,
    double radius) {
  check_points(points, "points");
  check_points(queries, "queries");
  check_knn_args(k, radius);
  check_bvh(nodes, prim_ids, points, queries);
  if (points.is_cpu()) {
    return bvh_knn_cpu(nodes, prim_ids, points, queries, k, radius);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return bvh_knn_cuda(nodes, prim_ids, points, queries, k, radius);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "points must be on the CPU or a CUDA device");
}

SearchResult packed_knn(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids,
    int64_t k,
    double radius) {
  check_points(points, "points");
  check_points(queries, "queries");
  check_knn_args(k, radius);
  check_packed_metadata(points, point_offsets, queries, query_batch_ids);
  if (points.is_cpu()) {
    return packed_knn_cpu(
        points, point_offsets, queries, query_batch_ids, k, radius);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return packed_knn_cuda(
        points, point_offsets, queries, query_batch_ids, k, radius);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "points must be on the CPU or a CUDA device");
}

SearchResult feature_knn1(
    const Tensor& reference, const Tensor& queries) {
  check_feature_pair(reference, queries);
  if (reference.is_cpu()) {
    return feature_knn1_cpu(reference, queries);
  }
  if (reference.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return feature_knn1_cuda(reference, queries);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "feature must be on the CPU or a CUDA device");
}

MutualSearchResult feature_mutual_knn1(
    const Tensor& source, const Tensor& target) {
  check_feature_pair(target, source);
  if (source.is_cpu()) {
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
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "feature must be on the CPU or a CUDA device");
}

EigenResult symmetric_eigh_3x3(const Tensor& matrices) {
  STD_TORCH_CHECK(
      matrices.scalar_type() == kFloat ||
          matrices.scalar_type() == kDouble,
      "matrices must be float32 or float64");
  STD_TORCH_CHECK(matrices.is_contiguous(), "matrices must be contiguous");
  STD_TORCH_CHECK(
      matrices.dim() >= 2 && matrices.size(matrices.dim() - 2) == 3 &&
          matrices.size(matrices.dim() - 1) == 3,
      "matrices must have shape (..., 3, 3)");
  if (matrices.is_cpu()) {
    return symmetric_eigh_3x3_cpu(matrices);
  }
  if (matrices.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return symmetric_eigh_3x3_cuda(matrices);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "matrices must be on the CPU or a CUDA device");
}

Tensor farthest_point_sample(
    const Tensor& points, int64_t num_samples) {
  STD_TORCH_CHECK(
      points.scalar_type() == kFloat || points.scalar_type() == kDouble,
      "points must be float32 or float64");
  STD_TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  STD_TORCH_CHECK(
      points.dim() == 2 && points.size(1) == 3,
      "points must have shape (N, 3)");
  STD_TORCH_CHECK(
      num_samples >= 1 && num_samples < points.size(0),
      "num_samples must be in [1, N)");
  if (points.is_cpu()) {
    return farthest_point_sample_cpu(points, num_samples);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return farthest_point_sample_cuda(points, num_samples);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "points must be on the CPU or a CUDA device");
}

MomentsResult neighbor_moments(
    const Tensor& points, const Tensor& indices) {
  STD_TORCH_CHECK(
      points.scalar_type() == kFloat || points.scalar_type() == kDouble,
      "points must be float32 or float64");
  STD_TORCH_CHECK(
      points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
  STD_TORCH_CHECK(
      indices.scalar_type() == kLong && indices.dim() == 2,
      "indices must be an int64 tensor of shape (M, K)");
  STD_TORCH_CHECK(
      points.is_contiguous() && indices.is_contiguous(),
      "neighbor_moments inputs must be contiguous");
  STD_TORCH_CHECK(
      points.device() == indices.device(),
      "neighbor_moments inputs must be on the same device");
  if (points.is_cpu()) {
    return neighbor_moments_cpu(points, indices);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return neighbor_moments_cuda(points, indices);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "points must be on the CPU or a CUDA device");
}

FpfhResult fpfh(
    const Tensor& points,
    const Tensor& normals,
    const Tensor& normal_valid,
    const Tensor& indices,
    const Tensor& distances2) {
  STD_TORCH_CHECK(
      points.scalar_type() == kFloat || points.scalar_type() == kDouble,
      "points must be float32 or float64");
  STD_TORCH_CHECK(
      points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
  const int64_t count = points.size(0);
  STD_TORCH_CHECK(
      normals.sizes().equals(points.sizes()) &&
          normals.scalar_type() == points.scalar_type(),
      "normals must match points");
  STD_TORCH_CHECK(
      normal_valid.scalar_type() == kBool && normal_valid.dim() == 1 &&
          normal_valid.size(0) == count,
      "normal_valid must be a bool tensor of shape (N,)");
  STD_TORCH_CHECK(
      indices.scalar_type() == kLong && indices.dim() == 2 &&
          indices.size(0) == count,
      "indices must be an int64 tensor of shape (N, K)");
  STD_TORCH_CHECK(
      distances2.sizes().equals(indices.sizes()) &&
          distances2.scalar_type() == points.scalar_type(),
      "distances2 must match indices in shape and points in dtype");
  for (const auto* tensor : {&points, &normals, &normal_valid, &indices, &distances2}) {
    STD_TORCH_CHECK(tensor->is_contiguous(), "fpfh inputs must be contiguous");
    STD_TORCH_CHECK(
        tensor->device() == points.device(),
        "fpfh inputs must be on the same device");
  }
  if (points.is_cpu()) {
    return fpfh_cpu(points, normals, normal_valid, indices, distances2);
  }
  if (points.is_cuda()) {
#ifdef TORCHPCL_WITH_CUDA
    return fpfh_cuda(points, normals, normal_valid, indices, distances2);
#else
    STD_TORCH_CHECK(
        false,
        "torchpcl was built without CUDA support; reinstall with "
        "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit");
#endif
  }
  STD_TORCH_CHECK(false, "points must be on the CPU or a CUDA device");
}

bool has_cuda() {
#ifdef TORCHPCL_WITH_CUDA
  return true;
#else
  return false;
#endif
}

}  // namespace torchpcl

extern "C" PyObject* PyInit__C(void) {
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}

// Every kernel checks its arguments and picks the CPU or CUDA implementation
// itself, so a CPU-only build still explains a CUDA tensor.
STABLE_TORCH_LIBRARY(torchpcl, m) {
  m.def("build_bvh(Tensor points) -> (Tensor, Tensor)");
  m.def(
      "bvh_knn(Tensor nodes, Tensor prim_ids, Tensor points, Tensor queries, "
      "int k, float radius) -> (Tensor, Tensor)");
  m.def(
      "packed_knn(Tensor points, Tensor point_offsets, Tensor queries, "
      "Tensor query_batch_ids, int k, float radius) -> (Tensor, Tensor)");
  m.def("feature_knn1(Tensor reference, Tensor queries) -> (Tensor, Tensor)");
  m.def(
      "feature_mutual_knn1(Tensor source, Tensor target) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def("symmetric_eigh_3x3(Tensor matrices) -> (Tensor, Tensor)");
  m.def("farthest_point_sample(Tensor points, int num_samples) -> Tensor");
  m.def(
      "fpfh(Tensor points, Tensor normals, Tensor normal_valid, "
      "Tensor indices, Tensor distances2) -> (Tensor, Tensor)");
  m.def(
      "neighbor_moments(Tensor points, Tensor indices) "
      "-> (Tensor, Tensor, Tensor)");
  m.def("has_cuda() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(torchpcl, CompositeExplicitAutograd, m) {
  m.impl("build_bvh", TORCH_BOX(&torchpcl::build_bvh));
  m.impl("bvh_knn", TORCH_BOX(&torchpcl::bvh_knn));
  m.impl("packed_knn", TORCH_BOX(&torchpcl::packed_knn));
  m.impl("feature_knn1", TORCH_BOX(&torchpcl::feature_knn1));
  m.impl("feature_mutual_knn1", TORCH_BOX(&torchpcl::feature_mutual_knn1));
  m.impl("symmetric_eigh_3x3", TORCH_BOX(&torchpcl::symmetric_eigh_3x3));
  m.impl("farthest_point_sample", TORCH_BOX(&torchpcl::farthest_point_sample));
  m.impl("fpfh", TORCH_BOX(&torchpcl::fpfh));
  m.impl("neighbor_moments", TORCH_BOX(&torchpcl::neighbor_moments));
  m.impl("has_cuda", TORCH_BOX(&torchpcl::has_cuda));
}
