#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace torchpcl {

#ifdef __CUDACC__
#define TORCHPCL_HOST_DEVICE __host__ __device__
#else
#define TORCHPCL_HOST_DEVICE
#endif

constexpr int kFpfhBins = 11;
constexpr int kFpfhSize = 3 * kFpfhBins;

namespace fpfh_detail {

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline scalar_t tiny() {
  return std::numeric_limits<scalar_t>::min();
}

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline scalar_t clamp_unit(scalar_t value) {
  return value < scalar_t(-1) ? scalar_t(-1)
                              : (value > scalar_t(1) ? scalar_t(1) : value);
}

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline scalar_t dot(const scalar_t* a, const scalar_t* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline void cross(
    const scalar_t* a, const scalar_t* b, scalar_t* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline int bin(scalar_t scaled) {
  const int value = static_cast<int>(::floor(scaled));
  return value < 0 ? 0 : (value > kFpfhBins - 1 ? kFpfhBins - 1 : value);
}

}  // namespace fpfh_detail

// The Open3D/PCL (theta, alpha, phi) pair feature of query point `p` with
// normal `n` and neighbor `q` with normal `m`, as histogram bins offset into
// their 11-bin blocks. Returns whether the pair distance is nonzero, which
// the aggregation pass needs. A pair without a valid Darboux frame lands in
// the bins of (0, 0, 0), like the tensor implementation it replaces.
template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline bool fpfh_pair_bins(
    const scalar_t* p,
    const scalar_t* n,
    const scalar_t* q,
    const scalar_t* m,
    int bins[3]) {
  using namespace fpfh_detail;
  const scalar_t displacement[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
  const scalar_t distance = ::sqrt(dot(displacement, displacement));
  const bool nonzero = distance > tiny<scalar_t>();
  const scalar_t safe_distance = nonzero ? distance : scalar_t(1);

  const scalar_t angle1 = clamp_unit(dot(n, displacement) / safe_distance);
  const scalar_t angle2 = clamp_unit(dot(m, displacement) / safe_distance);
  const bool swap = ::fabs(angle1) < ::fabs(angle2);
  const scalar_t* source_normal = swap ? m : n;
  const scalar_t* target_normal = swap ? n : m;
  const scalar_t sign = swap ? scalar_t(-1) : scalar_t(1);
  const scalar_t oriented[3] = {
      sign * displacement[0], sign * displacement[1], sign * displacement[2]};
  scalar_t phi = swap ? -angle2 : angle1;

  scalar_t v[3];
  cross(oriented, source_normal, v);
  const scalar_t v_norm = ::sqrt(dot(v, v));
  const bool frame_valid = nonzero && v_norm > tiny<scalar_t>();
  const scalar_t safe_v_norm = frame_valid ? v_norm : scalar_t(1);
  v[0] /= safe_v_norm;
  v[1] /= safe_v_norm;
  v[2] /= safe_v_norm;
  scalar_t w[3];
  cross(source_normal, v, w);

  scalar_t alpha = clamp_unit(dot(v, target_normal));
  scalar_t theta = ::atan2(dot(w, target_normal), dot(source_normal, target_normal));
  if (!frame_valid) {
    theta = alpha = phi = scalar_t(0);
  }
  const scalar_t pi = static_cast<scalar_t>(3.141592653589793238462643383279502884);
  const scalar_t two_pi = static_cast<scalar_t>(2 * 3.141592653589793238462643383279502884);
  bins[0] = bin(scalar_t(kFpfhBins) * (theta + pi) / two_pi);
  bins[1] = kFpfhBins + bin(scalar_t(kFpfhBins) * (alpha + scalar_t(1)) * scalar_t(0.5));
  bins[2] = 2 * kFpfhBins + bin(scalar_t(kFpfhBins) * (phi + scalar_t(1)) * scalar_t(0.5));
  return nonzero;
}

// A neighbor slot contributes an edge when it holds a point other than the
// query and both normals are valid.
TORCHPCL_HOST_DEVICE inline bool fpfh_edge(
    int64_t query, int64_t neighbor, const bool* normal_valid) {
  return neighbor >= 0 && neighbor != query && normal_valid[query] &&
      normal_valid[neighbor];
}

// Pass one: the simplified point feature histogram of `query`, each of its
// three blocks summing to 100 over its edges. Returns whether it has any.
template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline bool fpfh_spfh_one(
    int64_t query,
    int64_t k,
    const scalar_t* points,
    const scalar_t* normals,
    const bool* normal_valid,
    const int64_t* indices,
    scalar_t* spfh) {
  for (int bin = 0; bin < kFpfhSize; ++bin) spfh[bin] = scalar_t(0);
  const int64_t* row = indices + query * k;
  int64_t edges = 0;
  for (int64_t slot = 0; slot < k; ++slot) {
    edges += fpfh_edge(query, row[slot], normal_valid);
  }
  if (edges == 0) return false;
  const scalar_t increment = scalar_t(100) / static_cast<scalar_t>(edges);
  for (int64_t slot = 0; slot < k; ++slot) {
    const int64_t neighbor = row[slot];
    if (!fpfh_edge(query, neighbor, normal_valid)) continue;
    int bins[3];
    fpfh_pair_bins(
        points + query * 3, normals + query * 3,
        points + neighbor * 3, normals + neighbor * 3, bins);
    spfh[bins[0]] += increment;
    spfh[bins[1]] += increment;
    spfh[bins[2]] += increment;
  }
  return true;
}

// Pass two: the descriptor of `query`, its own SPFH plus the inverse squared
// distance weighted SPFHs of its neighbors, each block of that sum rescaled
// to 100. Zero, and invalid, without an own SPFH or a neighbor to aggregate.
template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline bool fpfh_descriptor_one(
    int64_t query,
    int64_t k,
    const scalar_t* points,
    const bool* normal_valid,
    const int64_t* indices,
    const scalar_t* distances2,
    const scalar_t* spfh,
    const bool* spfh_valid,
    scalar_t* descriptor) {
  using namespace fpfh_detail;
  scalar_t aggregated[kFpfhSize];
  for (int bin = 0; bin < kFpfhSize; ++bin) aggregated[bin] = scalar_t(0);
  int64_t count = 0;
  const int64_t* row = indices + query * k;
  if (spfh_valid[query]) {
    const scalar_t* p = points + query * 3;
    for (int64_t slot = 0; slot < k; ++slot) {
      const int64_t neighbor = row[slot];
      if (!fpfh_edge(query, neighbor, normal_valid) || !spfh_valid[neighbor]) {
        continue;
      }
      const scalar_t* q = points + neighbor * 3;
      const scalar_t displacement[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
      if (!(::sqrt(dot(displacement, displacement)) > tiny<scalar_t>())) continue;
      const scalar_t distance2 = distances2[query * k + slot];
      const scalar_t weight =
          scalar_t(1) / (distance2 > tiny<scalar_t>() ? distance2 : tiny<scalar_t>());
      const scalar_t* neighbor_spfh = spfh + neighbor * kFpfhSize;
      for (int bin = 0; bin < kFpfhSize; ++bin) {
        aggregated[bin] += neighbor_spfh[bin] * weight;
      }
      ++count;
    }
  }
  if (count == 0) {
    for (int bin = 0; bin < kFpfhSize; ++bin) descriptor[bin] = scalar_t(0);
    return false;
  }
  const scalar_t* own = spfh + query * kFpfhSize;
  for (int block = 0; block < 3; ++block) {
    scalar_t sum = scalar_t(0);
    for (int bin = 0; bin < kFpfhBins; ++bin) sum += aggregated[block * kFpfhBins + bin];
    const scalar_t scale = sum > scalar_t(0)
        ? scalar_t(100) / (sum > tiny<scalar_t>() ? sum : tiny<scalar_t>())
        : scalar_t(0);
    for (int bin = 0; bin < kFpfhBins; ++bin) {
      const int index = block * kFpfhBins + bin;
      descriptor[index] = own[index] + aggregated[index] * scale;
    }
  }
  return true;
}

#undef TORCHPCL_HOST_DEVICE

}  // namespace torchpcl
