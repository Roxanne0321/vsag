#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_set>

#include "vsag/vsag.h"

using namespace std::chrono;

using IntMatrix = std::vector<std::vector<int>>;
using FloatMatrix = std::vector<std::vector<float>>;

std::pair<IntMatrix, FloatMatrix>
knn_result_read(const std::string& fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Can not open file " + fname);
    }

    uint32_t n, d;
    file.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    if (file_size != 8 + n * d * (sizeof(int32_t) + sizeof(float))) {
        throw std::runtime_error("File size wrong");
    }
    file.seekg(8, std::ios::beg);

    IntMatrix I(n, std::vector<int32_t>(d));
    for (size_t i = 0; i < n; ++i) {
        file.read(reinterpret_cast<char*>(I[i].data()), d * sizeof(int32_t));
    }

    FloatMatrix D(n, std::vector<float>(d));
    for (size_t i = 0; i < n; ++i) {
        file.read(reinterpret_cast<char*>(D[i].data()), d * sizeof(float));
    }

    file.close();
    return {I, D};
}

float
cal_recall(const IntMatrix& ids,
           const IntMatrix& gt_ids,
           int64_t nq,
           int64_t result_topk,
           int64_t gt_topk) {
    int hit_count = 0;
    int64_t min_k = std::min(result_topk, gt_topk);

    for (int i = 0; i < nq; ++i) {
        std::unordered_set<int> gt_set(gt_ids[i].begin(), gt_ids[i].begin() + min_k);
        for (int j = 0; j < min_k; ++j) {
            if (gt_set.count(ids[i][j])) {
                hit_count++;
            }
        }
    }
    return static_cast<float>(hit_count) / (nq * min_k);
}

std::pair<vsag::SparseVector*, int64_t>
read_sparse_vectors_from_csr_file(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file " + filename);
    }

    int64_t sizes[3];
    infile.read(reinterpret_cast<char*>(sizes), 3 * sizeof(int64_t));
    int64_t num_rows = sizes[0];
    int64_t num_cols = sizes[1];
    int64_t nnz = sizes[2];

    std::vector<int64_t> indptr(num_rows + 1);
    infile.read(reinterpret_cast<char*>(indptr.data()), (num_rows + 1) * sizeof(int64_t));

    std::vector<int32_t> indices(nnz);
    infile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));

    std::vector<float> data(nnz);
    infile.read(reinterpret_cast<char*>(data.data()), nnz * sizeof(float));

    infile.close();

    vsag::SparseVector* sparse_vectors = new vsag::SparseVector[num_rows];

    for (int64_t i = 0; i < num_rows; ++i) {
        int64_t row_start = indptr[i];
        int64_t row_end = indptr[i + 1];
        int64_t row_size = row_end - row_start;

        sparse_vectors[i].dim_ = static_cast<uint32_t>(row_size);
        sparse_vectors[i].ids_ = new uint32_t[row_size];
        sparse_vectors[i].vals_ = new float[row_size];

        std::memcpy(
            sparse_vectors[i].ids_, indices.data() + row_start, row_size * sizeof(uint32_t));
        std::memcpy(sparse_vectors[i].vals_, data.data() + row_start, row_size * sizeof(float));
    }
    return std::make_pair(sparse_vectors, num_rows);
}

int
main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <index_path> <queryfile> <gtfile> <beta> <gamma> <topk> <num_threads>\n";
        return 1;
    }

    std::string index_path = argv[1];
    std::string queryfile = argv[2];
    std::string gtfile = argv[3];
    float beta = std::stof(argv[4]);
    int gamma = std::stoi(argv[5]);
    int topk = std::stoi(argv[6]);
    int num_threads = std::stoi(argv[7]);

    if (beta < 0.0f || beta > 1.0f) {
        std::cerr << "beta must be in [0, 1], got: " << beta << "\n";
        return 1;
    }
    if (gamma < topk) {
        std::cout << "[warn] gamma < topk, SINDI will internally use gamma = topk." << std::endl;
    }

    std::cout << "index_path: " << index_path << "\n";
    std::cout << "queryfile: " << queryfile << "\n";
    std::cout << "gtfile: " << gtfile << "\n";
    std::cout << "beta: " << beta << "\n";
    std::cout << "gamma: " << gamma << "\n";
    std::cout << "topk: " << topk << "\n";
    std::cout << "num_threads: " << num_threads << "\n";

    vsag::init();
    nlohmann::json sindi_build_parameters = {
        {"dtype", "float32"}, {"metric_type", "ip"}, {"dim", 30000}, {"sindi", {}}};

    auto index = vsag::Factory::CreateIndex("sindi", sindi_build_parameters.dump()).value();

    std::ifstream index_file(index_path, std::ios::binary);

    if (!index_file) {
        std::cerr << "Error opening file for deserializing." << std::endl;
        return 1;
    }
    index->Deserialize(index_file);

    /***************** Search SparseIVF Index ***************/
    std::pair<vsag::SparseVector*, int64_t> query_results =
        read_sparse_vectors_from_csr_file(queryfile);

    auto query = vsag::Dataset::Make();
    query->SparseVectors(query_results.first)->NumElements(query_results.second)->Owner(true);

    nlohmann::json sindi_search_parameters = {
        {"sindi", {{"num_threads", num_threads}, {"beta", beta}, {"gamma", gamma}}}};

    // warmup (run twice with the same beta/gamma)
    index->KnnSearch(query, topk, sindi_search_parameters.dump()).value();
    index->KnnSearch(query, topk, sindi_search_parameters.dump()).value();

    auto start_time = high_resolution_clock::now();

    // third run is used for qps statistics
    auto result = index->KnnSearch(query, topk, sindi_search_parameters.dump()).value();

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end_time - start_time).count();

    float qps = static_cast<float>(query_results.second) / (duration * 1.0e-9);
    std::cout << "qps: " << qps << std::endl;
    
    auto [gt_ids, gt_dists] = knn_result_read(gtfile);
    int64_t gt_topk = gt_ids.empty() ? 0 : gt_ids[0].size();

    IntMatrix ids(query_results.second, std::vector<int>(topk));

    for (size_t i = 0; i < query_results.second; ++i) {
        for (size_t j = 0; j < topk; ++j) {
            ids[i][j] = result->GetIds()[i * topk + j];
        }
    }

    float recall = cal_recall(ids, gt_ids, query_results.second, topk, gt_topk);
    std::cout << "recall: " << recall << std::endl;

    return 0;
}