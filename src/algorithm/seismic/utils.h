#include <cstdint>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "vsag/dataset.h"

namespace vsag {

std::vector<uint32_t>
get_top_n_indices(const SparseVector& vec, uint32_t n);

void
fixed_pruning(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
              int n_postings, uint32_t dim);

void
global_pruning(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
               int n_postings, uint32_t dim);
}  // namespace vsag