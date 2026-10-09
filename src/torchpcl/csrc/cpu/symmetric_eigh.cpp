#include "backend.h"
#include "common/symmetric_eigh.h"


namespace torchpcl {

EigenResult symmetric_eigh_3x3_cpu(const Tensor& matrices) {
  auto output_sizes = matrices.sizes().vec();
  output_sizes.resize(output_sizes.size() - 2);
  output_sizes.push_back(3);
  auto eigenvalues = torch::stable::new_empty(matrices, output_sizes);
  auto smallest_eigenvectors = torch::stable::new_empty(matrices, output_sizes);
  const int64_t count = matrices.numel() / 9;

  THO_DISPATCH_V2(matrices.scalar_type(), "symmetric_eigh_3x3_cpu", ([&] {
        const auto* input = matrices.const_data_ptr<scalar_t>();
        auto* values = eigenvalues.mutable_data_ptr<scalar_t>();
        auto* vectors = smallest_eigenvectors.mutable_data_ptr<scalar_t>();
        parallel_for(0, count, 256, [&](int64_t begin, int64_t end) {
          for (int64_t index = begin; index < end; ++index) {
            detail::symmetric_eigh_3x3_one(
                input + index * 9, values + index * 3, vectors + index * 3);
          }
        });
      }), AT_FLOATING_TYPES);
  return {eigenvalues, smallest_eigenvectors};
}

}  // namespace torchpcl
