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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <fstream>
#include <map>

#include "iostream"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

namespace py = pybind11;

void
SetLoggerOff() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kOFF);
}

void
SetLoggerInfo() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
}

void
SetLoggerDebug() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
}

template <typename T>
static void
writeBinaryPOD(std::ostream& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
}

template <typename T>
static void
readBinaryPOD(std::istream& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}

class Index {
public:
    Index(std::string name, const std::string& parameters) {
        if (auto index = vsag::Factory::CreateIndex(name, parameters)) {
            index_ = index.value();
        } else {
            vsag::Error error_code = index.error();
            if (error_code.type == vsag::ErrorType::UNSUPPORTED_INDEX) {
                throw std::runtime_error("error type: UNSUPPORTED_INDEX");
            } else if (error_code.type == vsag::ErrorType::INVALID_ARGUMENT) {
                throw std::runtime_error("error type: invalid_parameter");
            } else {
                throw std::runtime_error("error type: unexpectedError");
            }
        }
    }

public:
    void
    Build(py::array_t<float> vectors, py::array_t<int64_t> ids, size_t num_elements, size_t dim) {
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(dim)
            ->NumElements(num_elements)
            ->Ids(ids.mutable_data())
            ->Float32Vectors(vectors.mutable_data());
        index_->Build(dataset);
    }

    void
    SindiBuild(const std::string& filename) {
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
        auto base = vsag::Dataset::Make();
        base->SparseVectors(sparse_vectors)->NumElements(num_rows)->Owner(true);

        index_->Build(base);
    }

    py::object
    BatchSearch(int32_t nq,
                py::array_t<uint32_t> indptr,
                py::array_t<uint32_t> indices,
                py::array_t<float> data,
                int32_t topk, std::string& parameters) {
        vsag::SparseVector* query_vectors = new vsag::SparseVector[nq];

        auto indptr_unchecked = indptr.unchecked<1>();

        for (int32_t i = 0; i < nq; ++i) {
            uint32_t row_start = indptr_unchecked(i);
            uint32_t row_end = indptr_unchecked(i + 1);
            uint32_t row_size = row_end - row_start;

            query_vectors[i].dim_ = row_size;
            query_vectors[i].ids_ = new uint32_t[row_size];
            query_vectors[i].vals_ = new float[row_size];

            std::memcpy(query_vectors[i].ids_,
                        indices.mutable_data() + row_start,
                        row_size * sizeof(uint32_t));
            std::memcpy(
                query_vectors[i].vals_, data.mutable_data() + row_start, row_size * sizeof(float));
        }
        auto query = vsag::Dataset::Make();
        query->SparseVectors(query_vectors)->NumElements(nq)->Owner(true);

        std::vector<size_t> ids_shape = {static_cast<size_t>(nq), static_cast<size_t>(topk)};
        std::vector<size_t> ids_strides = {sizeof(int64_t) * topk, sizeof(int64_t)};

        std::vector<size_t> dists_shape = {static_cast<size_t>(nq), static_cast<size_t>(topk)};
        std::vector<size_t> dists_strides = {sizeof(float) * topk, sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);

        if (auto result = index_->KnnSearch(query, topk, parameters);
            result.has_value()) {
            auto ids_view = ids.mutable_unchecked<2>();  // 注意这里指定为二维
            auto vsag_ids = result.value()->GetIds();

            auto dists_view = dists.mutable_unchecked<2>();  // 注意这里指定为二维
            auto vsag_dists = result.value()->GetDistances();

            for (int i = 0; i < nq; ++i) {
                for (int j = 0; j < topk; ++j) {
                    ids_view(i, j) = vsag_ids[i * topk + j];
                    dists_view(i, j) = vsag_dists[i * topk + j];
                }
            }
        }

        return py::make_tuple(ids, dists);
    }

    py::object
    KnnSearch(py::array_t<float> vector, size_t k, std::string& parameters) {
        auto query = vsag::Dataset::Make();
        size_t data_num = 1;
        query->NumElements(data_num)
            ->Dim(vector.size())
            ->Float32Vectors(vector.mutable_data())
            ->Owner(false);

        size_t ids_shape[1]{k};
        size_t ids_strides[1]{sizeof(int64_t)};
        size_t dists_shape[1]{k};
        size_t dists_strides[1]{sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);
        if (auto result = index_->KnnSearch(query, k, parameters); result.has_value()) {
            auto ids_view = ids.mutable_unchecked<1>();
            auto dists_view = dists.mutable_unchecked<1>();

            auto vsag_ids = result.value()->GetIds();
            auto vsag_distances = result.value()->GetDistances();
            for (int i = 0; i < data_num * k; ++i) {
                ids_view(i) = vsag_ids[i];
                dists_view(i) = vsag_distances[i];
            }
        }

        return py::make_tuple(ids, dists);
    }

    py::object
    RangeSearch(py::array_t<float> point, float threshold, std::string& parameters) {
        auto query = vsag::Dataset::Make();
        size_t data_num = 1;
        query->NumElements(data_num)
            ->Dim(point.size())
            ->Float32Vectors(point.mutable_data())
            ->Owner(false);

        py::array_t<int64_t> labels;
        py::array_t<float> dists;
        if (auto result = index_->RangeSearch(query, threshold, parameters); result.has_value()) {
            auto ids = result.value()->GetIds();
            auto distances = result.value()->GetDistances();
            auto k = result.value()->GetDim();
            labels.resize({k});
            dists.resize({k});
            auto labels_data = labels.mutable_data();
            auto dists_data = dists.mutable_data();
            for (int i = 0; i < data_num * k; ++i) {
                labels_data[i] = ids[i];
                dists_data[i] = distances[i];
            }
        }

        return py::make_tuple(labels, dists);
    }

    void
    Save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        index_->Serialize(file);
        file.close();
    }

    void
    Load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        index_->Deserialize(file);
        file.close();
    }

private:
    std::shared_ptr<vsag::Index> index_;
};

PYBIND11_MODULE(_pyvsag, m) {
    m.def("set_logger_off", &SetLoggerOff, "SetLoggerOff");
    m.def("set_logger_info", &SetLoggerInfo, "SetLoggerInfo");
    m.def("set_logger_debug", &SetLoggerDebug, "SetLoggerDebug");
    py::class_<Index>(m, "Index")
        .def(py::init<std::string, std::string&>(), py::arg("name"), py::arg("parameters"))
        .def("build",
             &Index::Build,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"))
        .def("sindi_build", &Index::SindiBuild, py::arg("filename"))
        .def(
            "knn_search", &Index::KnnSearch, py::arg("vector"), py::arg("k"), py::arg("parameters"))
        .def("batch_search",
             &Index::BatchSearch,
             py::arg("nq"),
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("data"),
             py::arg("topk"),
             py::arg("parameters"))
        .def("range_search",
             &Index::RangeSearch,
             py::arg("vector"),
             py::arg("threshold"),
             py::arg("parameters"))
        .def("save", &Index::Save, py::arg("filename"))
        .def("load", &Index::Load, py::arg("filename"));
}
