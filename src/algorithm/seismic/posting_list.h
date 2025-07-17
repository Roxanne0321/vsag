#pragma once
#include "algorithm/seismic/summary.h"
#include "algorithm/seismic/utils.h"

#include <vector>

namespace vsag {
class PostingList {
public:
    PostingList();

    PostingList(std::vector<std::vector<std::pair<uint32_t, float>>>& clusters);

    ~PostingList() {}

    void
    search_posting_list();

    void
    serialize(std::ostream& out_stream, uint32_t n_centroids);

    void
    deserialize(std::istream& in_stream, uint32_t n_centroids);

private:
    std::vector<uint32_t> block_offsets;
    std::vector<uint32_t> ids_;
    std::vector<float> vals_;
};
}// namespace vsag