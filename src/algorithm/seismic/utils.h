#include <cstdint>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>
#include <mutex>
#include <iostream>

#include "vsag/dataset.h"
#include "typing.h"

namespace vsag {
void
vector_prune(VectorPruneStrategy vector_prune_strategy,
             const SparseVector* data,
             uint32_t num,
             std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map);

void
list_prune(ListPruneStrategy list_prune_strategy, 
           uint32_t dim,
           std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map);

std::vector<uint32_t>
get_top_n_indices(const SparseVector& vec, float n_cut);

void
fixed_pruning(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
              int n_postings, uint32_t dim);

void
global_pruning(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
               int n_postings, uint32_t dim);

float
SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2);

float 
DenseComputeIP(const std::vector<float> &query, const SparseVector& base);

void
do_kmeans_on_doc_id(const SparseVector* data,
                    const std::vector<std::pair<uint32_t, float>> &postings_ids_vals,
                    std::vector<std::vector<std::pair<uint32_t, float>>>& clusters,
                    uint32_t n_centroids,
                    uint32_t min_cluster_size);

void
energy_preserving_summary(const SparseVector* data,
                          std::vector<uint32_t>& ids,
                          std::vector<float>& vals,
                          std::vector<uint32_t> cluster,
                          float fraction);
}  // namespace vsag