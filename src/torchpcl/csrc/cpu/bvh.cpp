#include "backend.h"
#include "common/checks.h"

#include "cuBQL/builder/cpu.h"
#include "cuBQL/bvh.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include "cuBQL/queries/pointData/knn.h"

#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace torchpcl {
namespace {

class CpuBvh final : public BvhImpl {
 public:
  explicit CpuBvh(const at::Tensor& points) : points_(points) {
    TORCH_CHECK(points_.device().is_cpu(), "points must be a CPU tensor");
    const auto count = static_cast<uint32_t>(points_.size(0));
    const auto* data = reinterpret_cast<const cuBQL::vec3f*>(
        points_.data_ptr<float>());
    std::vector<cuBQL::box3f> boxes(count);
    at::parallel_for(0, count, 4096, [&](int64_t begin, int64_t end) {
      for (int64_t index = begin; index < end; ++index) {
        boxes[index] = cuBQL::box3f().including(data[index]);
      }
    });
    cuBQL::cpuBuilder(bvh_, boxes.data(), count, cuBQL::BuildConfig{});
  }

  ~CpuBvh() override { cuBQL::cpu::freeBVH(bvh_); }

  SearchResult knn(
      const at::Tensor& queries, int64_t k, double radius) override {
    check_points(queries, "queries");
    TORCH_CHECK(queries.device().is_cpu(), "queries must be a CPU tensor");
    const auto count = queries.size(0);
    auto indices = at::empty({count, k}, queries.options().dtype(at::kInt));
    auto distances = at::empty({count, k}, queries.options());
    const float max_distance2 = static_cast<float>(radius * radius);
    const auto* points = reinterpret_cast<const cuBQL::vec3f*>(
        points_.data_ptr<float>());
    const auto* query_data = reinterpret_cast<const cuBQL::vec3f*>(
        queries.data_ptr<float>());
    auto* output_indices = indices.data_ptr<int32_t>();
    auto* output_distances = distances.data_ptr<float>();

    if (k == 1) {
      at::parallel_for(0, count, 512, [&](int64_t begin, int64_t end) {
        for (int64_t query = begin; query < end; ++query) {
          const int index = cuBQL::points::findClosest(
              bvh_, points, query_data[query], max_distance2);
          output_indices[query] = index;
          output_distances[query] = index >= 0
              ? cuBQL::sqrDistance(points[index], query_data[query])
              : max_distance2;
        }
      });
      return {indices, distances};
    }

    at::parallel_for(0, count, 128, [&](int64_t begin, int64_t end) {
      cuBQL::knn::Candidate candidates[kMaxK];
      for (int64_t query = begin; query < end; ++query) {
        const auto result = cuBQL::points::findKNN(
            candidates,
            static_cast<int>(k),
            bvh_,
            points,
            query_data[query],
            max_distance2);
        for (int64_t rank = 0; rank < k; ++rank) {
          const bool found = rank < result.numFound;
          output_indices[query * k + rank] =
              found ? candidates[rank].primID : -1;
          output_distances[query * k + rank] =
              found ? candidates[rank].sqrDist : max_distance2;
        }
      }
    });
    return {indices, distances};
  }

 private:
  at::Tensor points_;
  cuBQL::bvh3f bvh_{};
};

}  // namespace

std::shared_ptr<BvhImpl> make_cpu_bvh(const at::Tensor& points) {
  return std::make_shared<CpuBvh>(points);
}

}  // namespace torchpcl
