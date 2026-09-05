#include "backend.h"
#include "common/farthest_point.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

namespace torchpcl {
namespace {

constexpr int kBlockThreads = 256;

// One block, one iteration: every thread grid-strides over all points,
// merging in the previous center's distance and tracking its own local best,
// then a shared-memory tree reduction picks the block-wide (i.e. global)
// next farthest point. Reading `center` from `selected[sample - 1]` and
// writing `selected[sample]` keeps the whole loop device-resident: launches
// are issued back-to-back on one stream with no host readback between them.
template <typename scalar_t>
__global__ void farthest_point_sample_step_kernel(
    const scalar_t* points,
    scalar_t* min_dist2,
    int64_t* selected,
    int64_t count,
    int64_t sample) {
  __shared__ FarthestCandidate<scalar_t> shared[kBlockThreads];
  const scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();
  const int64_t center = selected[sample - 1];

  FarthestCandidate<scalar_t> best{-kInfinity, count};
  for (int64_t i = threadIdx.x; i < count; i += kBlockThreads) {
    const scalar_t updated = i == center
        ? -kInfinity
        : min(min_dist2[i], squared_distance(points, i, center));
    min_dist2[i] = updated;
    best = farther_candidate(best, FarthestCandidate<scalar_t>{updated, i});
  }

  shared[threadIdx.x] = best;
  __syncthreads();
  for (int offset = kBlockThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] =
          farther_candidate(shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    selected[sample] = shared[0].index;
  }
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

  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "farthest_point_sample_cuda", [&] {
        for (int64_t sample = 1; sample < num_samples; ++sample) {
          farthest_point_sample_step_kernel<scalar_t>
              <<<1, kBlockThreads, 0, stream>>>(
                  points.data_ptr<scalar_t>(),
                  min_dist2.data_ptr<scalar_t>(),
                  selected.data_ptr<int64_t>(),
                  count,
                  sample);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return selected;
}

}  // namespace torchpcl
