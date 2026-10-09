#pragma once

#include "backend.h"

#include "cuBQL/bvh.h"

#include <cstdint>

namespace torchpcl {

static_assert(sizeof(cuBQL::bvh3f::Node) == kBvhNodeBytes);

// cuBQL's view of the tensors of a BvhResult; it borrows their memory.
inline cuBQL::bvh3f bvh_view(const at::Tensor& nodes, const at::Tensor& prim_ids) {
  cuBQL::bvh3f bvh{};
  bvh.nodes = reinterpret_cast<cuBQL::bvh3f::Node*>(nodes.data_ptr<uint8_t>());
  bvh.numNodes = static_cast<uint32_t>(nodes.size(0));
  bvh.primIDs = reinterpret_cast<uint32_t*>(prim_ids.data_ptr<int32_t>());
  bvh.numPrims = static_cast<uint32_t>(prim_ids.size(0));
  return bvh;
}

}  // namespace torchpcl
