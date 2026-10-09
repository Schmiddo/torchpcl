#include "backend.h"
#include "common/farthest_point.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace torchpcl {
namespace {

constexpr int kBlockThreads = 256;
// Points per thread targeted when sizing the grid. Low enough that a cloud of
// a few hundred thousand points still spreads across the whole device, high
// enough that the second pass stays small.
constexpr int64_t kItemsPerThread = 4;
// Resident blocks per SM to aim for. The step kernel is memory-bound, so
// oversubscribing the SMs keeps loads in flight across the grid-stride loop.
constexpr int64_t kBlocksPerSm = 8;

// Tree-reduce one candidate per thread down to `shared[0]`, which every
// thread may read once this returns. `farther_candidate` is associative and
// breaks ties toward the lower index, so the result does not depend on how
// points were split across threads or blocks.
template <typename scalar_t>
__device__ FarthestCandidate<scalar_t> reduce_candidates(
    FarthestCandidate<scalar_t>* shared, FarthestCandidate<scalar_t> best) {
  shared[threadIdx.x] = best;
  __syncthreads();
  for (int offset = kBlockThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] =
          farther_candidate(shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  return shared[0];
}

// Pass one: the grid splits the points into disjoint strides, so every block
// updates its own slice of `min_dist2` without racing, then reduces it to one
// candidate. A single block can finish the job on its own and writes the
// answer straight to `selected`; a larger grid leaves one candidate per block
// for `reduce_block_candidates_kernel`.
//
// Reading `center` from `selected[sample - 1]` and writing `selected[sample]`
// keeps the whole loop device-resident: launches are issued back-to-back on
// one stream with no host readback between them.
template <typename scalar_t>
__global__ void farthest_point_sample_step_kernel(
    const scalar_t* points,
    scalar_t* min_dist2,
    int64_t* selected,
    scalar_t* block_values,
    int64_t* block_indices,
    int64_t count,
    int64_t sample) {
  __shared__ FarthestCandidate<scalar_t> shared[kBlockThreads];
  const scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();
  const int64_t center = selected[sample - 1];
  // Hoisted out of the loop; otherwise every point re-reads the center.
  const scalar_t center_x = points[center * 3 + 0];
  const scalar_t center_y = points[center * 3 + 1];
  const scalar_t center_z = points[center * 3 + 2];

  FarthestCandidate<scalar_t> best{-kInfinity, count};
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kBlockThreads;
  for (int64_t i = blockIdx.x * kBlockThreads + threadIdx.x; i < count;
       i += stride) {
    scalar_t updated = -kInfinity;
    if (i != center) {
      const scalar_t dx = points[i * 3 + 0] - center_x;
      const scalar_t dy = points[i * 3 + 1] - center_y;
      const scalar_t dz = points[i * 3 + 2] - center_z;
      updated = min(min_dist2[i], dx * dx + dy * dy + dz * dz);
    }
    min_dist2[i] = updated;
    best = farther_candidate(best, FarthestCandidate<scalar_t>{updated, i});
  }

  best = reduce_candidates(shared, best);
  if (threadIdx.x == 0) {
    if (gridDim.x == 1) {
      selected[sample] = best.index;
    } else {
      block_values[blockIdx.x] = best.value;
      block_indices[blockIdx.x] = best.index;
    }
  }
}

// Pass two: one block reduces the per-block candidates to the next center.
template <typename scalar_t>
__global__ void reduce_block_candidates_kernel(
    const scalar_t* block_values,
    const int64_t* block_indices,
    int64_t* selected,
    int64_t blocks,
    int64_t count,
    int64_t sample) {
  __shared__ FarthestCandidate<scalar_t> shared[kBlockThreads];
  const scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();

  FarthestCandidate<scalar_t> best{-kInfinity, count};
  for (int64_t i = threadIdx.x; i < blocks; i += kBlockThreads) {
    best = farther_candidate(
        best, FarthestCandidate<scalar_t>{block_values[i], block_indices[i]});
  }

  best = reduce_candidates(shared, best);
  if (threadIdx.x == 0) {
    selected[sample] = best.index;
  }
}

// Spread the points across the device without making pass two's reduction
// wider than it needs to be.
int grid_blocks(int64_t count) {
  const int64_t per_block = kBlockThreads * kItemsPerThread;
  const int64_t needed = (count + per_block - 1) / per_block;
  int device = 0;
  int multiprocessors = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, device));
  const int64_t saturating = int64_t{multiprocessors} * kBlocksPerSm;
  return static_cast<int>(std::max<int64_t>(1, std::min(needed, saturating)));
}

}  // namespace

at::Tensor farthest_point_sample_cuda(
    const at::Tensor& points, int64_t num_samples) {
  const c10::cuda::CUDAGuard guard(points.get_device());
  const int64_t count = points.size(0);
  auto selected = at::empty({num_samples}, points.options().dtype(at::kLong));
  selected.narrow(0, 0, 1).zero_();
  auto min_dist2 = at::full(
      {count}, std::numeric_limits<double>::infinity(), points.options());

  const int blocks = grid_blocks(count);
  // Only a multi-block grid needs somewhere to stage its partial results.
  const int64_t staging = blocks > 1 ? blocks : 0;
  auto block_values = at::empty({staging}, points.options());
  auto block_indices =
      at::empty({staging}, points.options().dtype(at::kLong));

  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "farthest_point_sample_cuda", [&] {
        auto* values = blocks > 1 ? block_values.data_ptr<scalar_t>() : nullptr;
        auto* indices = blocks > 1 ? block_indices.data_ptr<int64_t>() : nullptr;
        for (int64_t sample = 1; sample < num_samples; ++sample) {
          farthest_point_sample_step_kernel<scalar_t>
              <<<blocks, kBlockThreads, 0, stream>>>(
                  points.data_ptr<scalar_t>(),
                  min_dist2.data_ptr<scalar_t>(),
                  selected.data_ptr<int64_t>(),
                  values,
                  indices,
                  count,
                  sample);
          if (blocks > 1) {
            reduce_block_candidates_kernel<scalar_t>
                <<<1, kBlockThreads, 0, stream>>>(
                    values,
                    indices,
                    selected.data_ptr<int64_t>(),
                    blocks,
                    count,
                    sample);
          }
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return selected;
}

}  // namespace torchpcl
