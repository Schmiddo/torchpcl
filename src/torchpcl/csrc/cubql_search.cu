// cuBQL-backed 1-NN-within-radius search for torchpcl.
//
// Single translation unit: pulls in the cuBQL GPU builder implementation
// (header-only library, see third_party/cuBQL) and exposes a PointBVH
// class to Python via the torch extension machinery.

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/findClosest.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__global__ void make_boxes(cuBQL::box3f* boxes, const cuBQL::vec3f* pts, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  boxes[i] = cuBQL::box3f().including(pts[i]);
}

__global__ void nn_query(
    int32_t* out_idx,
    float* out_dist2,
    cuBQL::bvh3f bvh,
    const cuBQL::vec3f* pts,
    const cuBQL::vec3f* queries,
    int n,
    float max_d2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const cuBQL::vec3f q = queries[i];
  const int j = cuBQL::points::findClosest(bvh, pts, q, max_d2);
  out_idx[i] = j;
  // dist2 is only meaningful where j >= 0 (mirrors the warp backend,
  // which leaves radius^2 in unmatched slots).
  out_dist2[i] = j >= 0 ? cuBQL::sqrDistance(pts[j], q) : max_d2;
}

constexpr int kBlock = 256;

class PointBVH {
 public:
  explicit PointBVH(at::Tensor points) {
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.scalar_type() == at::kFloat, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
    TORCH_CHECK(points.size(0) > 0, "points must be non-empty");
    points_ = points.contiguous();
    device_index_ = points_.get_device();

    const c10::cuda::CUDAGuard guard(device_index_);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto n = static_cast<uint32_t>(points_.size(0));

    // Temporary box array; the caching allocator keeps it stream-ordered,
    // so it can be dropped as soon as the build is enqueued.
    at::Tensor boxes = at::empty(
        {points_.size(0), 6}, points_.options().dtype(at::kFloat));
    make_boxes<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        reinterpret_cast<cuBQL::box3f*>(boxes.data_ptr<float>()),
        reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
        static_cast<int>(n));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cuBQL::gpuBuilder(
        bvh_,
        reinterpret_cast<const cuBQL::box3f*>(boxes.data_ptr<float>()),
        n,
        cuBQL::BuildConfig{},
        stream);
  }

  ~PointBVH() {
    try {
      const c10::cuda::CUDAGuard guard(device_index_);
      // Freeing on the query stream orders the free after in-flight queries.
      cuBQL::cuda::free(bvh_, at::cuda::getCurrentCUDAStream());
    } catch (...) {
      // Swallow errors during interpreter shutdown / lost CUDA context.
    }
  }

  PointBVH(const PointBVH&) = delete;
  PointBVH& operator=(const PointBVH&) = delete;

  std::tuple<at::Tensor, at::Tensor> query(at::Tensor queries, double radius) {
    TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
    TORCH_CHECK(queries.get_device() == device_index_,
                "queries must be on the same device as the indexed points");
    TORCH_CHECK(queries.scalar_type() == at::kFloat, "queries must be float32");
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 3, "queries must have shape (M, 3)");
    auto queries_c = queries.contiguous();
    const auto m = queries_c.size(0);

    const c10::cuda::CUDAGuard guard(device_index_);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::Tensor out_idx = at::empty({m}, queries_c.options().dtype(at::kInt));
    at::Tensor out_dist2 = at::empty({m}, queries_c.options().dtype(at::kFloat));
    if (m > 0) {
      const float max_d2 = static_cast<float>(radius * radius);
      nn_query<<<(m + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
          out_idx.data_ptr<int32_t>(),
          out_dist2.data_ptr<float>(),
          bvh_,
          reinterpret_cast<const cuBQL::vec3f*>(points_.data_ptr<float>()),
          reinterpret_cast<const cuBQL::vec3f*>(queries_c.data_ptr<float>()),
          static_cast<int>(m),
          max_d2);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_idx, out_dist2};
  }

 private:
  at::Tensor points_;  // keeps the indexed device memory alive
  cuBQL::bvh3f bvh_{};
  c10::DeviceIndex device_index_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PointBVH>(m, "PointBVH")
      .def(py::init<at::Tensor>())
      .def("query", &PointBVH::query);
}
