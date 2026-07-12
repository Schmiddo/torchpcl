#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>
#include <tuple>

namespace torchpcl {

constexpr int64_t kMaxK = 64;

using SearchResult = std::tuple<at::Tensor, at::Tensor>;

class BvhImpl {
 public:
  virtual ~BvhImpl() = default;
  virtual SearchResult knn(
      const at::Tensor& queries, int64_t k, double radius) = 0;
};

std::shared_ptr<BvhImpl> make_cpu_bvh(const at::Tensor& points);
SearchResult packed_knn_cpu(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius);

#ifdef TORCHPCL_WITH_CUDA
std::shared_ptr<BvhImpl> make_cuda_bvh(const at::Tensor& points);
SearchResult packed_knn_cuda(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius);
#endif

class BvhIndex {
 public:
  explicit BvhIndex(const at::Tensor& points);
  SearchResult knn(const at::Tensor& queries, int64_t k, double radius);

 private:
  std::shared_ptr<BvhImpl> impl_;
};

SearchResult packed_knn(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius);

bool has_cuda();

}  // namespace torchpcl
