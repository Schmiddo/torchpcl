#pragma once

#include <cstdint>

namespace torchpcl {

#ifdef __CUDACC__
#define TORCHPCL_HOST_DEVICE __host__ __device__
#else
#define TORCHPCL_HOST_DEVICE
#endif

TORCHPCL_HOST_DEVICE inline bool better_candidate(
    float distance2,
    int32_t index,
    float other_distance2,
    int32_t other_index) {
  return distance2 < other_distance2 ||
      (distance2 == other_distance2 &&
       (other_index < 0 || index < other_index));
}

TORCHPCL_HOST_DEVICE inline void insert_candidate(
    float* distances2,
    int32_t* indices,
    int k,
    float distance2,
    int32_t index) {
  if (!better_candidate(
          distance2, index, distances2[k - 1], indices[k - 1])) {
    return;
  }
  int rank = k - 1;
  while (rank > 0 &&
         better_candidate(
             distance2, index, distances2[rank - 1], indices[rank - 1])) {
    distances2[rank] = distances2[rank - 1];
    indices[rank] = indices[rank - 1];
    --rank;
  }
  distances2[rank] = distance2;
  indices[rank] = index;
}

#undef TORCHPCL_HOST_DEVICE

}  // namespace torchpcl
