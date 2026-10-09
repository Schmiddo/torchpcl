#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <tuple>

namespace torchpcl {

constexpr int64_t kMaxK = 64;
// sizeof(cuBQL::bvh3f::Node), asserted in common/bvh.h.
constexpr int64_t kBvhNodeBytes = 32;
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

using SearchResult = std::tuple<Tensor, Tensor>;
// A BVH over N points as two tensors on the points' device: `nodes`
// (num_nodes, kBvhNodeBytes) uint8 in cuBQL's BinaryBVH node layout, and
// `prim_ids` (num_prims,) int32, the point rows the leaves name. Points with
// a NaN coordinate are left out, so num_prims may be less than N.
using BvhResult = std::tuple<Tensor, Tensor>;
using EigenResult = std::tuple<Tensor, Tensor>;
// FPFH descriptors (N, 33) and their validity (N,).
using FpfhResult = std::tuple<Tensor, Tensor>;
// Nearest target row of every source row and nearest source row of every
// target row, each with its squared distance.
using MutualSearchResult =
    std::tuple<Tensor, Tensor, Tensor, Tensor>;
// Per-row neighborhood means (N, 3), centered scatter matrices (N, 3, 3) and
// neighbor counts (N,).
using MomentsResult = std::tuple<Tensor, Tensor, Tensor>;

// Every native search (bvh_knn, packed_knn) returns up to k nearest points
// per query within an inclusive `radius`, as final results: int64 indices
// and float32 squared distances from exact_distance2 (common/distance.h),
// with -1 and +inf in empty slots.

BvhResult build_bvh_cpu(const Tensor& points);
SearchResult bvh_knn_cpu(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries,
    int64_t k,
    double radius);
SearchResult feature_knn1_cpu(
    const Tensor& reference, const Tensor& queries);
MutualSearchResult feature_mutual_knn1_cpu(
    const Tensor& source, const Tensor& target);
SearchResult packed_knn_cpu(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids,
    int64_t k,
    double radius);
EigenResult symmetric_eigh_3x3_cpu(const Tensor& matrices);
Tensor farthest_point_sample_cpu(
    const Tensor& points, int64_t num_samples);
MomentsResult neighbor_moments_cpu(
    const Tensor& points, const Tensor& indices);
FpfhResult fpfh_cpu(
    const Tensor& points,
    const Tensor& normals,
    const Tensor& normal_valid,
    const Tensor& indices,
    const Tensor& distances2);

#ifdef TORCHPCL_WITH_CUDA
BvhResult build_bvh_cuda(const Tensor& points);
SearchResult bvh_knn_cuda(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries,
    int64_t k,
    double radius);
SearchResult feature_knn1_cuda(
    const Tensor& reference, const Tensor& queries);
SearchResult packed_knn_cuda(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids,
    int64_t k,
    double radius);
EigenResult symmetric_eigh_3x3_cuda(const Tensor& matrices);
Tensor farthest_point_sample_cuda(
    const Tensor& points, int64_t num_samples);
MomentsResult neighbor_moments_cuda(
    const Tensor& points, const Tensor& indices);
FpfhResult fpfh_cuda(
    const Tensor& points,
    const Tensor& normals,
    const Tensor& normal_valid,
    const Tensor& indices,
    const Tensor& distances2);
#endif

BvhResult build_bvh(const Tensor& points);
// Search the BVH `build_bvh` made of `points` (see BvhResult).
SearchResult bvh_knn(
    const Tensor& nodes,
    const Tensor& prim_ids,
    const Tensor& points,
    const Tensor& queries,
    int64_t k,
    double radius);

SearchResult packed_knn(
    const Tensor& points,
    const Tensor& point_offsets,
    const Tensor& queries,
    const Tensor& query_batch_ids,
    int64_t k,
    double radius);
SearchResult feature_knn1(
    const Tensor& reference, const Tensor& queries);
MutualSearchResult feature_mutual_knn1(
    const Tensor& source, const Tensor& target);
EigenResult symmetric_eigh_3x3(const Tensor& matrices);
Tensor farthest_point_sample(
    const Tensor& points, int64_t num_samples);
// Moments of the points named by fixed-width neighborhood rows `indices`
// (M, K), -1 in empty slots.
MomentsResult neighbor_moments(
    const Tensor& points, const Tensor& indices);
// Descriptors from a fixed-width self-neighborhood: `indices` (N, K) with -1
// in empty slots and their squared `distances2`, unit `normals` (zero where
// `normal_valid` is false). Everything but the masks shares one float dtype.
FpfhResult fpfh(
    const Tensor& points,
    const Tensor& normals,
    const Tensor& normal_valid,
    const Tensor& indices,
    const Tensor& distances2);

bool has_cuda();

}  // namespace torchpcl
