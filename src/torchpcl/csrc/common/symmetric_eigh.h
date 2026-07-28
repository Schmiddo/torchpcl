#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define TORCHPCL_HOST_DEVICE __host__ __device__
#else
#define TORCHPCL_HOST_DEVICE
#endif

namespace torchpcl {
namespace detail {

constexpr double kDoubleEpsilon = 2.2204460492503131e-16;
constexpr double kPi = 3.141592653589793238462643383279502884;

TORCHPCL_HOST_DEVICE inline double square(double value) {
  return value * value;
}

TORCHPCL_HOST_DEVICE inline double absolute(double value) {
  return value < 0.0 ? -value : value;
}

TORCHPCL_HOST_DEVICE inline double maximum(double first, double second) {
  return first > second ? first : second;
}

TORCHPCL_HOST_DEVICE inline void cross(
    const double first[3], const double second[3], double output[3]) {
  output[0] = first[1] * second[2] - first[2] * second[1];
  output[1] = first[2] * second[0] - first[0] * second[2];
  output[2] = first[0] * second[1] - first[1] * second[0];
}

TORCHPCL_HOST_DEVICE inline double norm_squared(const double vector[3]) {
  return square(vector[0]) + square(vector[1]) + square(vector[2]);
}

template <typename scalar_t>
TORCHPCL_HOST_DEVICE inline void symmetric_eigh_3x3_one(
    const scalar_t* matrix,
    scalar_t* eigenvalues,
    scalar_t* smallest_eigenvector) {
  // Use double intermediates for both input types. This matches the previous
  // tensor implementation's promotion of float32 and keeps close eigenvalue
  // gaps from collapsing at the acos boundary.
  const double diagonal[3] = {
      static_cast<double>(matrix[0]),
      static_cast<double>(matrix[4]),
      static_cast<double>(matrix[8]),
  };
  const double off_diagonal[3] = {
      static_cast<double>(matrix[1]),
      static_cast<double>(matrix[2]),
      static_cast<double>(matrix[5]),
  };
  const double mean = (diagonal[0] + diagonal[1] + diagonal[2]) / 3.0;
  const double centered_diagonal[3] = {
      diagonal[0] - mean,
      diagonal[1] - mean,
      diagonal[2] - mean,
  };
  const double p = ::sqrt(
      (square(centered_diagonal[0]) + square(centered_diagonal[1]) +
       square(centered_diagonal[2]) + 2.0 * square(off_diagonal[0]) +
       2.0 * square(off_diagonal[1]) + 2.0 * square(off_diagonal[2])) /
      6.0);
  const double safe_p = p > 0.0 ? p : 1.0;
  const double b00 = centered_diagonal[0] / safe_p;
  const double b11 = centered_diagonal[1] / safe_p;
  const double b22 = centered_diagonal[2] / safe_p;
  const double b01 = off_diagonal[0] / safe_p;
  const double b02 = off_diagonal[1] / safe_p;
  const double b12 = off_diagonal[2] / safe_p;
  const double half_determinant =
      (b00 * b11 * b22 + 2.0 * b01 * b02 * b12 -
       b00 * square(b12) - b11 * square(b02) - b22 * square(b01)) /
      2.0;

  double determinant_ratio;
  if (absolute(1.0 - absolute(half_determinant)) <=
      4.0 * kDoubleEpsilon) {
    determinant_ratio =
        half_determinant > 0.0 ? 1.0 : (half_determinant < 0.0 ? -1.0 : 0.0);
  } else {
    determinant_ratio =
        half_determinant < -1.0
        ? -1.0
        : (half_determinant > 1.0 ? 1.0 : half_determinant);
  }
  const double phi = ::acos(determinant_ratio) / 3.0;
  const double smallest =
      mean + 2.0 * p * ::cos(phi + 2.0 * kPi / 3.0);
  const double largest = mean + 2.0 * p * ::cos(phi);
  const double middle = 3.0 * mean - smallest - largest;
  eigenvalues[0] = static_cast<scalar_t>(smallest);
  eigenvalues[1] = static_cast<scalar_t>(middle);
  eigenvalues[2] = static_cast<scalar_t>(largest);

  // The input contract is symmetric, so only consume its upper triangle.
  // Mirroring those entries also prevents insignificant lower/upper storage
  // differences from affecting the eigenvector construction.
  double shifted[3][3] = {
      {diagonal[0] - smallest, off_diagonal[0], off_diagonal[1]},
      {off_diagonal[0], diagonal[1] - smallest, off_diagonal[2]},
      {off_diagonal[1], off_diagonal[2], diagonal[2] - smallest},
  };
  double scale = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      scale = maximum(scale, absolute(shifted[row][column]));
    }
  }

  double cross_products[3][3];
  cross(shifted[0], shifted[1], cross_products[0]);
  cross(shifted[0], shifted[2], cross_products[1]);
  cross(shifted[1], shifted[2], cross_products[2]);
  double best_cross_norm2 = norm_squared(cross_products[0]);
  int best_cross = 0;
  for (int index = 1; index < 3; ++index) {
    const double candidate_norm2 = norm_squared(cross_products[index]);
    if (candidate_norm2 > best_cross_norm2) {
      best_cross_norm2 = candidate_norm2;
      best_cross = index;
    }
  }

  const double eigenvalue_scale = maximum(
      absolute(smallest), maximum(absolute(middle), absolute(largest)));
  const bool smallest_is_unique =
      middle - smallest > 16.0 * kDoubleEpsilon * eigenvalue_scale;
  const double cross_threshold = kDoubleEpsilon * scale * scale;
  const bool stable_cross =
      best_cross_norm2 > cross_threshold * cross_threshold &&
      smallest_is_unique;

  double vector[3];
  if (stable_cross) {
    for (int component = 0; component < 3; ++component) {
      vector[component] = cross_products[best_cross][component];
    }
  } else {
    int best_row = 0;
    double best_row_norm2 = norm_squared(shifted[0]);
    for (int row = 1; row < 3; ++row) {
      const double candidate_norm2 = norm_squared(shifted[row]);
      if (candidate_norm2 > best_row_norm2) {
        best_row_norm2 = candidate_norm2;
        best_row = row;
      }
    }
    int axis_index = 0;
    if (absolute(shifted[best_row][1]) <
        absolute(shifted[best_row][axis_index])) {
      axis_index = 1;
    }
    if (absolute(shifted[best_row][2]) <
        absolute(shifted[best_row][axis_index])) {
      axis_index = 2;
    }
    const double axis[3] = {
        axis_index == 0 ? 1.0 : 0.0,
        axis_index == 1 ? 1.0 : 0.0,
        axis_index == 2 ? 1.0 : 0.0,
    };
    cross(shifted[best_row], axis, vector);
    if (norm_squared(vector) == 0.0) {
      for (int component = 0; component < 3; ++component) {
        vector[component] = axis[component];
      }
    }
  }

  const double inverse_norm = 1.0 / ::sqrt(norm_squared(vector));
  for (int component = 0; component < 3; ++component) {
    smallest_eigenvector[component] =
        static_cast<scalar_t>(vector[component] * inverse_norm);
  }
}

}  // namespace detail
}  // namespace torchpcl

#undef TORCHPCL_HOST_DEVICE
