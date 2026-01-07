#include <algorithm/sindi/utils.h>

namespace vsag {

float 
DenseComputeIP(const std::vector<float> &query, const SparseVector& base) {
    const size_t N_LANES = 4;
    float result[N_LANES] = {0.0, 0.0, 0.0, 0.0};
    
    // 处理完整的N_LANES块
    size_t full_blocks = base.dim_ / N_LANES;
    
    for (size_t i = 0; i < full_blocks * N_LANES; i += N_LANES) {
        result[0] += query[base.ids_[i]] * base.vals_[i];
        result[1] += query[base.ids_[i + 1]] * base.vals_[i + 1];
        result[2] += query[base.ids_[i + 2]] * base.vals_[i + 2];
        result[3] += query[base.ids_[i + 3]] * base.vals_[i + 3];
    }

    // 处理剩余部分
    for (size_t i = full_blocks * N_LANES; i < base.dim_; ++i) {
        result[0] += query[base.ids_[i]] * base.vals_[i];
    }

    // 汇总结果
    return - (result[0] + result[1] + result[2] + result[3]);
}

float
SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2) {
    float sum = 0.0f;
    int i = 0, j = 0;

    while (i < sv1.dim_ && j < sv2.dim_) {
        if (sv1.ids_[i] == sv2.ids_[j]) {
            sum += sv1.vals_[i] * sv2.vals_[j];
            i++;
            j++;
        } else if (sv1.ids_[i] < sv2.ids_[j]) {
            i++;
        } else {
            j++;
        }
    }
    return -sum;
}

std::vector<uint32_t>
mass_prune(const SparseVector& vec, float alpha) {
    float total_mass = 0.0f;
    std::vector<uint32_t> indices(vec.dim_);
    for (uint32_t i = 0; i < vec.dim_; ++i) {
        indices[i] = i;
        total_mass += vec.vals_[i];
    }

    if (alpha == 1) {
        return indices;
    }
    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
        return vec.vals_[a] > vec.vals_[b];
    });

    float part_mass = total_mass * alpha;
    float temp_mass = 0.0f;
    int max_index = 0;
    while (temp_mass < part_mass) {
        temp_mass += vec.vals_[indices[max_index]];
        max_index++;
    }

    indices.resize(max_index);

    return indices;
}
}