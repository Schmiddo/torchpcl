#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <tuple>

namespace torchpcl {

constexpr int64_t kMaxK = 64;

using SearchResult = std::tuple<at::Tensor, at::Tensor>;
using EigenResult = std::tuple<at::Tensor, at::Tensor>;

class BvhImpl {
 public:
  virtual ~BvhImpl() = default;
  virtual SearchResult knn(
      const at::Tensor& queries, int64_t k, double radius) = 0;
};

std::shared_ptr<BvhImpl> make_cpu_bvh(const at::Tensor& points);
SearchResult feature_knn1_cpu(
    const at::Tensor& reference, const at::Tensor& queries);
SearchResult packed_knn_cpu(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius);
EigenResult symmetric_eigh_3x3_cpu(const at::Tensor& matrices);

#ifdef TORCHPCL_WITH_CUDA
std::shared_ptr<BvhImpl> make_cuda_bvh(const at::Tensor& points);
SearchResult feature_knn1_cuda(
    const at::Tensor& reference, const at::Tensor& queries);
SearchResult packed_knn_cuda(
    const at::Tensor& points,
    const at::Tensor& point_offsets,
    const at::Tensor& queries,
    const at::Tensor& query_batch_ids,
    int64_t k,
    double radius);
EigenResult symmetric_eigh_3x3_cuda(const at::Tensor& matrices);
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
SearchResult feature_knn1(
    const at::Tensor& reference, const at::Tensor& queries);
EigenResult symmetric_eigh_3x3(const at::Tensor& matrices);

bool has_cuda();

}  // namespace torchpcl
