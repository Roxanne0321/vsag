#include <omp.h>
#include <sys/stat.h>
#include <vsag/vsag.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <getopt.h>
using namespace std::chrono;

std::pair<vsag::SparseVector*, int64_t>
read_sparse_vectors_from_csr_file(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file");
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

int main(int argc, char** argv) {
    vsag::init();

    std::string dataset = "base_1M"; // Provide default values
    int window_size = 100000;
    int n_cut = 40;

    struct option long_options[] = {
        {"dataset", required_argument, 0, 'd'},
        {"window_size", required_argument, 0, 'w'},
        {"n_cut", required_argument, 0, 'n'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:w:n:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                dataset = optarg;
                break;
            case 'w':
                window_size = std::stoi(optarg);
                break;
            case 'n':
                n_cut = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [--dataset <dataset>] [--window_size <window_size>] [--n_cut <n_cut>]" << std::endl;
                return 1;
        }
    }

    std::cout << "dataset: " << dataset << std::endl;
    std::cout << "window_size: " << window_size << std::endl;
    std::cout << "n_cut: " << n_cut << std::endl;

    std::string basefile = "sparse/data/" + dataset + ".csr";
    std::pair<vsag::SparseVector*, int64_t> base_results = read_sparse_vectors_from_csr_file(basefile);

    auto base = vsag::Dataset::Make();
    base->SparseVectors(base_results.first)->NumElements(base_results.second)->Owner(true);

    nlohmann::json sparse_ipivf_build_parameters = {
        {"dtype", "float32"},
        {"metric_type", "ip"},
        {"dim", 30000},
        {"sparse_ipivf",
         {{"window_size", window_size},
          {"doc_prune_strategy", {{"prune_type", "NotPrune"}}},
          {"vector_prune_strategy",
           {{"prune_type", "VectorPrune"}, {"n_cut", n_cut}}}}}};

    auto index =
        vsag::Factory::CreateIndex("sparse_ipivf", sparse_ipivf_build_parameters.dump()).value();

    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index SparseIPIVF contains: "
                  << index->GetNumElements() << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    std::string index_path = "sparse/index/" + dataset + "_ws_" + std::to_string(window_size) +
                             "_nc_" + std::to_string(n_cut) + ".idx";
    std::ofstream index_file(index_path, std::ios::binary);

    if (!index_file) {
        std::cerr << "Error opening file for serialization." << std::endl;
        return 1;
    }
    index->Serialize(index_file);

    return 0;
}