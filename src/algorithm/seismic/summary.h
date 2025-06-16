#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>
#include <fstream>

#include "vsag/dataset.h"

namespace vsag {
class QuantizedSummary {
public:
    QuantizedSummary() {
        n_summaries_ = 0;
        d_ = 0;
    }
    QuantizedSummary(std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> dataset, uint32_t original_dim);
    ~QuantizedSummary() {}
    std::vector<float>
    matmul_with_query(uint32_t* query_ids, float* query_vals, uint32_t dim) const;

    void
    serialize(std::ostream& out_stream);

    void
    deserialize(std::istream& in_stream);

//private:
    uint32_t n_summaries_;
    uint32_t d_;
    uint32_t nnz_;
    std::vector<uint32_t> offsets_;
    std::vector<uint32_t> summaries_ids_;
    std::vector<uint8_t> values_;
    std::vector<float> minimums_;
    std::vector<float> quants_;
};
}  // namespace vsag