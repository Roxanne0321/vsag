#include <cstdint>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "vsag/dataset.h"
namespace vsag {
    float 
    DenseComputeIP(const std::vector<float> &query, const SparseVector& base);

    float 
    SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2);
}