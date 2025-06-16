#include <cstdint>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>
#include <mutex>
#include <iostream>

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

float
SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2);

void
do_kmeans_on_doc_id(const SparseVector* data,
                    std::vector<uint32_t>& doc_ids,
                    std::vector<std::vector<uint32_t>>& clusters,
                    uint32_t n_centroids,
                    uint32_t min_cluster_size,
                    uint32_t k);

void
initialize_kmeans(const SparseVector* data,
                  std::vector<uint32_t>& doc_ids,
                  std::vector<std::vector<uint32_t>>& clusters,
                  uint32_t n_centroids,
                  uint32_t min_cluster_size);

void
update_cluster_centers(const SparseVector* data,
                        std::vector<std::vector<uint32_t>>& clusters,
                        std::vector<uint32_t>& centroids_ids,
                        uint32_t n_centroids);

void
random_kmeans(const SparseVector* data,
              std::vector<uint32_t> doc_ids,
              std::vector<std::vector<uint32_t>>& clusters,
              uint32_t n_centroids,
              uint32_t min_cluster_size);

void
energy_preserving_summary(const SparseVector* data,
                          std::vector<uint32_t>& ids,
                          std::vector<float>& vals,
                          std::vector<uint32_t> cluster,
                          float fraction);
}  // namespace vsag