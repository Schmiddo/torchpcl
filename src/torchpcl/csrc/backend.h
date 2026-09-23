#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <tuple>

namespace torchpcl {

constexpr int64_t kMaxK = 64;
// Points per BVH leaf. cuBQL's default gives every point a leaf of its own,
// 2N-1 nodes for N points. Larger leaves mean fewer nodes to build and walk,
// but a fixed-radius query then tests more points outside its radius.
// Measured on 200k-point clouds (operations-random/-redwood): on CPU, 8 cuts
// the build by 29-40% and knn-based operations by 9-26%, while radius search
// on uniform points slows by ~11%; 16 gains a little more but slows it by
// ~26%. The GPU pays that radius cost sooner -- from 3 up, radius and hybrid
// queries (radius, FPFH) slow down faster than knn speeds up -- and 2 is the
// largest that costs neither.
constexpr int kCpuBvhLeafSize = 8;
constexpr int kCudaBvhLeafSize = 2;

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
at::Tensor farthest_point_sample_cpu(
    const at::Tensor& points, int64_t num_samples);

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
at::Tensor farthest_point_sample_cuda(
    const at::Tensor& points, int64_t num_samples);
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
at::Tensor farthest_point_sample(
    const at::Tensor& points, int64_t num_samples);

bool has_cuda();

}  // namespace torchpcl
