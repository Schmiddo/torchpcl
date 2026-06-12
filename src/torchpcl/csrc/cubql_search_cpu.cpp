// cuBQL-backed 1-NN / k-NN search on the CPU.
//
// Plain C++ translation unit (no nvcc): cuBQL's queries are
// host/device-agnostic and builder/cpu.h provides a header-only host
// builder, so this compiles with the system C++ compiler and runs the
// exact same traversal code as the CUDA extension. Queries are
// parallelized over the intra-op thread pool.

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cpu.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <vector>

namespace {

// Matches the CUDA extension's per-thread candidate cap.
constexpr int kMaxKnn = 64;

class PointBVH {
 public:
  explicit PointBVH(at::Tensor points) {
    TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
    TORCH_CHECK(points.scalar_type() == at::kFloat, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
    TORCH_CHECK(points.size(0) > 0, "points must be non-empty");
    points_ = points.contiguous();
    const auto n = static_cast<uint32_t>(points_.size(0));

    const auto* pts = reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>());
    std::vector<cuBQL::box3f> boxes(n);
    at::parallel_for(0, n, 4096, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        boxes[i] = cuBQL::box3f().including(pts[i]);
      }
    });
    // Note: cuBQL's only host builder is single-threaded; its OpenMP
    // builder targets device offload and needs OpenMP 5.1 (gcc >= 13),
    // so queries are parallel (at::parallel_for) but the build is not.
    cuBQL::cpuBuilder(bvh_, boxes.data(), n, cuBQL::BuildConfig{});
  }

  ~PointBVH() { cuBQL::cpu::freeBVH(bvh_); }

  PointBVH(const PointBVH&) = delete;
  PointBVH& operator=(const PointBVH&) = delete;

  std::tuple<at::Tensor, at::Tensor> query(at::Tensor queries, double radius) {
    auto queries_c = check_queries(queries);
    const auto m = queries_c.size(0);
    at::Tensor out_idx = at::empty({m}, queries_c.options().dtype(at::kInt));
    at::Tensor out_dist2 = at::empty({m}, queries_c.options().dtype(at::kFloat));
    const float max_d2 = static_cast<float>(radius * radius);
    const auto* pts = reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>());
    const auto* qs = reinterpret_cast<const cuBQL::vec3f*>(queries_c.data_ptr<float>());
    auto* idx = out_idx.data_ptr<int32_t>();
    auto* d2 = out_dist2.data_ptr<float>();

    at::parallel_for(0, m, 512, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int j = cuBQL::points::findClosest(bvh_, pts, qs[i], max_d2);
        idx[i] = j;
        d2[i] = j >= 0 ? cuBQL::sqrDistance(pts[j], qs[i]) : max_d2;
      }
    });
    return {out_idx, out_dist2};
  }

  std::tuple<at::Tensor, at::Tensor> knn(at::Tensor queries, int64_t k, double radius) {
    auto queries_c = check_queries(queries);
    TORCH_CHECK(k >= 1 && k <= kMaxKnn, "k must be in [1, ", kMaxKnn, "]");
    const auto m = queries_c.size(0);
    at::Tensor out_idx = at::empty({m, k}, queries_c.options().dtype(at::kInt));
    at::Tensor out_dist2 = at::empty({m, k}, queries_c.options().dtype(at::kFloat));
    const float max_d2 = static_cast<float>(radius * radius);
    const auto* pts = reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>());
    const auto* qs = reinterpret_cast<const cuBQL::vec3f*>(queries_c.data_ptr<float>());
    auto* idx = out_idx.data_ptr<int32_t>();
    auto* d2 = out_dist2.data_ptr<float>();

    at::parallel_for(0, m, 128, [&](int64_t begin, int64_t end) {
      cuBQL::knn::Candidate cand[kMaxKnn];
      for (int64_t i = begin; i < end; ++i) {
        const cuBQL::knn::Result res = cuBQL::points::findKNN(
            cand, static_cast<int>(k), bvh_, pts, qs[i], max_d2);
        for (int64_t j = 0; j < k; ++j) {
          const bool found = j < res.numFound;
          idx[i * k + j] = found ? cand[j].primID : -1;
          d2[i * k + j] = found ? cand[j].sqrDist : max_d2;
        }
      }
    });
    return {out_idx, out_dist2};
  }

 private:
  at::Tensor check_queries(const at::Tensor& queries) const {
    TORCH_CHECK(queries.device().is_cpu(), "queries must be a CPU tensor");
    TORCH_CHECK(queries.scalar_type() == at::kFloat, "queries must be float32");
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 3, "queries must have shape (M, 3)");
    return queries.contiguous();
  }

  at::Tensor points_;  // keeps the indexed memory alive
  cuBQL::bvh3f bvh_{};
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PointBVH>(m, "PointBVH")
      .def(py::init<at::Tensor>())
      .def("query", &PointBVH::query)
      .def("knn", &PointBVH::knn);
}
