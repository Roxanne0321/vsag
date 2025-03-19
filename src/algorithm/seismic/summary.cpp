#include "algorithm/seismic/summary.h"

#include <iostream>

namespace vsag {
std::tuple<float, float, std::vector<uint8_t>>
quantize(float* values, uint32_t dim, size_t n_classes) {
    assert(values != nullptr);

    // 初始化为第一个元素
    float min_value = values[0];
    float max_value = values[0];

    // 计算 min 和 max
    for (uint32_t i = 1; i < dim; ++i) {
        min_value = std::min(min_value, values[i]);
        max_value = std::max(max_value, values[i]);
    }

    // 量化处理，将[min, max]区间分成 n_classes 块
    float quant = (max_value - min_value) / static_cast<float>(n_classes);
    std::vector<uint8_t> query_values;
    query_values.reserve(dim);

    for (uint32_t i = 0; i < dim; ++i) {
        float normalized_value = std::min(n_classes - 1.0f, (values[i] - min_value) / quant);
        uint8_t q = static_cast<uint8_t>(normalized_value);
        query_values.push_back(q);
    }

    return std::make_tuple(min_value, quant, query_values);
}

QuantizedSummary::QuantizedSummary(
    std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> dataset,
    uint32_t original_dim) {
    uint32_t num = dataset.size();
    int nnz = 0;

    for (auto vector : dataset) {
        nnz += vector.first.size();
    }

    std::vector<std::vector<std::pair<uint8_t, uint32_t>>> inverted_pairs(original_dim);

    uint32_t n_classes = 256;

    for (size_t doc_id = 0; doc_id < num; ++doc_id) {
        // 获取组件和值，这里需要您自行定义获得组件和值的方法
        auto sv = dataset[doc_id];

        auto [minimum, quant, current_codes] =
            quantize(sv.second.data(), sv.second.size(), n_classes);
        minimums_.emplace_back(minimum);
        quants_.emplace_back(quant);

        for (size_t i = 0; i < sv.first.size(); ++i) {
            uint32_t c = sv.first[i];
            uint8_t score = current_codes[i];
            inverted_pairs[c].emplace_back(score, doc_id);
        }
    }
    offsets_.emplace_back(0);

    for (const auto& ip : inverted_pairs) {
        for (const auto& [s, id] : ip) {
            values_.emplace_back(s);
            summaries_ids_.emplace_back(id);
        }
        offsets_.emplace_back(ip.size());
    }

    for (size_t id = 1; id < offsets_.size(); ++id) {
        offsets_[id] += offsets_[id - 1];
    }
    //assert(offsets_.size() == (original_dim + 1) && "bad offsets summary");

    this->n_summaries_ = num;
    this->d_ = original_dim;
}

std::vector<float>
QuantizedSummary::matmul_with_query(uint32_t* query_ids, float* query_vals, uint32_t dim) {
    std::vector<float> accumulator(n_summaries_, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        uint32_t qc = query_ids[i];
        float qv = query_vals[i];

        if(qc > offsets_.size()) {
            std::cout << "qc: " << qc << " offsets size: " << offsets_.size() << " dim: " << d_ <<std::endl;
        }
        assert(qc < offsets_.size() && "wrong query term id");

        auto current_offset = offsets_[qc];
        auto next_offset = offsets_[qc + 1];
        if (next_offset - current_offset == 0)
            continue;

        std::vector<uint32_t> current_summaries_ids(summaries_ids_.begin() + current_offset,
                                                    summaries_ids_.begin() + next_offset);
        std::vector<uint8_t> current_values(values_.begin() + current_offset,
                                            values_.begin() + next_offset);

        for (size_t j = 0; j < current_summaries_ids.size(); ++j) {
            uint32_t s_id = current_summaries_ids[j];
            uint8_t v = current_values[j];

            float val = static_cast<float>(v) * quants_[s_id] + minimums_[s_id];
            accumulator[s_id] += val * qv;
        }
    }

    return accumulator;
}
}  // namespace vsag