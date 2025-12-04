
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sparse_term_datacell.h"

namespace vsag {

void
SparseTermDataCell::Query(float* global_dists, const SparseTermComputerPtr& computer) const {
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto term_it = active_term_sizes_.find(term);
        if (term_it == active_term_sizes_.end()) {
            continue;
        }
        computer->ScanForAccumulate(it,
                                    term_ids_.at(term).data(),
                                    term_datas_.at(term).data(),
                                    static_cast<uint32_t>(static_cast<float>(term_it->second) *
                                                          computer->term_retain_ratio_),
                                    global_dists);
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::InsertHeap(float* dists,
                               const SparseTermComputerPtr& computer,
                               MaxHeap& heap,
                               const InnerSearchParam& param,
                               uint32_t offset_id) const {
    uint32_t id = 0;
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        // note that radius = 1 - ip -> radius - 1 = 0 - ip
        // the dist in heap is equal to 0 - ip
        // thus, we need to compare dist with radius - 1
        cur_heap_top = radius - 1;
    }

    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto term_it = active_term_sizes_.find(term);
        if (term_it == active_term_sizes_.end()) {
            continue;
        }

        uint32_t i = 0;
        auto term_size = static_cast<uint32_t>(static_cast<float>(term_it->second) *
                                               computer->term_retain_ratio_);
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (heap.size() < n_candidate) {
                for (; i < term_size; i++) {
                    id = term_ids_.at(term)[i];

                    if constexpr (type == InnerSearchType::WITH_FILTER) {
                        if (not filter->CheckValid(id + offset_id)) {
                            dists[id] = 0;
                            continue;
                        }
                    }

                    if (dists[id] != 0) {
                        heap.emplace(dists[id], id + offset_id);
                        cur_heap_top = heap.top().first;
                        dists[id] = 0;
                    }

                    if (heap.size() == n_candidate) {
                        break;
                    }
                }
            }
        }

        for (; i < term_size; i++) {
            id = term_ids_.at(term)[i];

            if constexpr (type == InnerSearchType::WITH_FILTER) {
#if __cplusplus >= 202002L
                if (dists[id] > cur_heap_top or not filter->CheckValid(id + offset_id)) [[likely]] {
#else
                if (__builtin_expect(
                        dists[id] > cur_heap_top or not filter->CheckValid(id + offset_id), 1)) {
#endif
                    dists[id] = 0;
                    continue;
                }
            } else {
#if __cplusplus >= 202002L
                if (dists[id] > cur_heap_top) [[likely]] {
#else
                if (__builtin_expect(dists[id] > cur_heap_top, 1)) {
#endif
                    dists[id] = 0;
                    continue;
                }
            }
            heap.emplace(dists[id], id + offset_id);
            if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
                heap.pop();
                cur_heap_top = heap.top().first;
            }
            if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
                cur_heap_top = radius - 1;
            }
            dists[id] = 0;
        }
    }
    computer->ResetTerm();
}

void
SparseTermDataCell::DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const {
    // use this function when inserting
    if (sorted_base.size() <= 1 || doc_retain_ratio_ == 1) {
        return;
    }
    float total_mass = 0.0F;
    for (const auto& pair : sorted_base) {
        total_mass += pair.second;
    }

    float part_mass = total_mass * doc_retain_ratio_;
    float temp_mass = 0.0F;
    int pruned_doc_len = 0;

    while (temp_mass < part_mass) {
        temp_mass += sorted_base[pruned_doc_len++].second;
    }

    sorted_base.resize(pruned_doc_len);
}

void
SparseTermDataCell::InsertVector(const SparseVector& sparse_base, uint32_t base_id) {
    // check term
    uint32_t max_term_id = 0;
    for (auto i = 0; i < sparse_base.len_; i++) {
        auto term_id = sparse_base.ids_[i];
        max_term_id = std::max(max_term_id, term_id);
    }
    if (max_term_id > term_id_limit_) {
        throw std::runtime_error(
            fmt::format("max term id of sparse vector {} is greater than term id limit {}",
                        max_term_id,
                        term_id_limit_));
    }
    
    Vector<std::pair<uint32_t, float>> sorted_base(allocator_);
    sort_sparse_vector(sparse_base, sorted_base);

    // doc prune
    DocPrune(sorted_base);

    // insert vector
    for (auto& item : sorted_base) {
        auto term = item.first;
        auto val = item.second;

        active_term_sizes_[term] += 1;

        term_ids_.try_emplace(term, allocator_);
        term_ids_.at(term).push_back(base_id);

        term_datas_.try_emplace(term, allocator_);
        term_datas_.at(term).push_back(val);
    }
}

float
SparseTermDataCell::CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint32_t base_id) {
    float ip = 0;
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto term_it = active_term_sizes_.find(term);
        if (term_it == active_term_sizes_.end()) {
            continue;
        }
        computer->ScanForCalculateDist(
            it, term_ids_.at(term).data(), term_datas_.at(term).data(), term_it->second, base_id, &ip);
    }
    computer->ResetTerm();
    return 1 + ip;
}

void
SparseTermDataCell::Serialize(StreamWriter& writer) const {
    uint32_t num_active_terms = active_term_sizes_.size();
    StreamWriter::WriteObj(writer, num_active_terms);
    for (auto &term_size : active_term_sizes_) {
        uint32_t term = static_cast<uint32_t>(term_size.first);
        uint32_t size = static_cast<uint32_t>(term_size.second);
        StreamWriter::WriteObj(writer, term);
        StreamWriter::WriteObj(writer, size);
        StreamWriter::WriteVector(writer, term_ids_.at(term));
        StreamWriter::WriteVector(writer, term_datas_.at(term));
    }
}

void
SparseTermDataCell::Deserialize(StreamReader& reader) {
    uint32_t num_active_terms;
    StreamReader::ReadObj(reader, num_active_terms);
    for (auto i = 0; i < num_active_terms; i++) {
        uint32_t term, size;
        StreamReader::ReadObj(reader, term);
        StreamReader::ReadObj(reader, size);
        active_term_sizes_[term] = size;
        term_ids_.try_emplace(term, size, allocator_); 
        term_datas_.try_emplace(term, size, allocator_); 
        StreamReader::ReadVector(reader, term_ids_.at(term));
        StreamReader::ReadVector(reader, term_datas_.at(term));
    }
}

template void
SparseTermDataCell::InsertHeap<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeap<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeap<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeap<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

}  // namespace vsag
