#include "algorithm/seismic/summary.h"
#include "vsag/dataset.h"
#include <iostream>

int
main() {
    std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> summary_data;
    summary_data.emplace_back(std::vector<uint32_t>{0, 1, 2}, std::vector<float>{0.1, 0.2, 0.3});
    summary_data.emplace_back(std::vector<uint32_t>{1, 2, 3}, std::vector<float>{0.4, 0.5, 0.6});

    vsag::QuantizedSummary summaries = vsag::QuantizedSummary(summary_data, 4);

    uint32_t* query_ids;
    float* query_vals;

    query_ids = new uint32_t[3];;
    query_vals = new float[3];
    query_ids[0] = 0;
    query_ids[1] = 1;
    query_ids[2] = 2;

    query_vals[0] = 1;
    query_vals[1] = 0.5;
    query_vals[2] = 0.3;

    auto results = summaries.matmul_with_query(query_ids, query_vals, 3);
 
    for (auto result : results) {
        std::cout << result << std::endl;
    }
    delete []query_ids;
    delete []query_vals;
}