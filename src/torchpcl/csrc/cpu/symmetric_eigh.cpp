#include "backend.h"
#include "common/symmetric_eigh.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>

namespace torchpcl {

EigenResult symmetric_eigh_3x3_cpu(const at::Tensor& matrices) {
  auto output_sizes = matrices.sizes().vec();
  output_sizes.resize(output_sizes.size() - 2);
  output_sizes.push_back(3);
  auto eigenvalues = at::empty(output_sizes, matrices.options());
  auto smallest_eigenvectors = at::empty(output_sizes, matrices.options());
  const int64_t count = matrices.numel() / 9;

  AT_DISPATCH_FLOATING_TYPES(
      matrices.scalar_type(), "symmetric_eigh_3x3_cpu", [&] {
        const auto* input = matrices.data_ptr<scalar_t>();
        auto* values = eigenvalues.data_ptr<scalar_t>();
        auto* vectors = smallest_eigenvectors.data_ptr<scalar_t>();
        at::parallel_for(0, count, 256, [&](int64_t begin, int64_t end) {
          for (int64_t index = begin; index < end; ++index) {
            detail::symmetric_eigh_3x3_one(
                input + index * 9, values + index * 3, vectors + index * 3);
          }
        });
      });
  return {eigenvalues, smallest_eigenvectors};
}

}  // namespace torchpcl
