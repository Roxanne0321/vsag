#include <cstdint>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>

#include "vsag/dataset.h"
namespace vsag {
    float 
    DenseComputeIP(const std::vector<float> &query, const SparseVector& base);

    float 
    SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2);

    std::vector<uint32_t>
    mass_prune(const SparseVector& vec, float alpha);
}