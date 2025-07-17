#include "algorithm/seismic/posting_list.h"
namespace vsag {
PostingList::PostingList() {
}

PostingList::PostingList(std::vector<std::vector<std::pair<uint32_t, float>>>& clusters) {
    block_offsets.emplace_back(0);
    for (auto cluster : clusters) {
        if (cluster.size() != 0) {
            for (auto id_val : cluster) {
                ids_.emplace_back(id_val.first);
                vals_.emplace_back(id_val.second);
            }
        }
        block_offsets.emplace_back(ids_.size());
    }
}

void
PostingList::serialize(std::ostream& out_stream, uint32_t n_centroids) {
    if (n_centroids != 0) {
        out_stream.write(reinterpret_cast<const char*>(block_offsets.data()),
                         block_offsets.size() * sizeof(uint32_t));
        out_stream.write(reinterpret_cast<const char*>(ids_.data()),
                         ids_.size() * sizeof(uint32_t));
        out_stream.write(reinterpret_cast<const char*>(vals_.data()), vals_.size() * sizeof(float));
    }
}

void
PostingList::deserialize(std::istream& in_stream, uint32_t n_centroids) {
    std::cout << "n_centroids: " << n_centroids << "\n";  // Print n_centroids_

    if (n_centroids != 0) {
        block_offsets.resize(n_centroids);
        in_stream.read(reinterpret_cast<char*>(block_offsets.data()),
                       block_offsets.size() * sizeof(uint32_t));
        std::cout << "block_offsets: ";
        for (const auto& offset : block_offsets) {
            std::cout << offset << " ";
        }
        std::cout << "\n";

        auto num = block_offsets[n_centroids - 1];
        ids_.resize(num);
        vals_.resize(num);

        in_stream.read(reinterpret_cast<char*>(ids_.data()), num * sizeof(uint32_t));
        in_stream.read(reinterpret_cast<char*>(vals_.data()), num * sizeof(float));

        std::cout << "ids_: ";
        for (const auto& id : ids_) {
            std::cout << id << " ";
        }
        std::cout << "\n";

        std::cout << "vals_: ";
        for (const auto& val : vals_) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}
}  // namespace vsag