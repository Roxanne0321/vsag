#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pyvsag
import numpy as np
import pickle
import sys
import json
import os
import struct
import time

def save_list_to_pickle(my_list, filename):
    with open(filename, 'wb') as file:
        pickle.dump(my_list, file)

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D

def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return sizes, data, indices, indptr

def cal_recall(ids, dists, gt_ids, gt_dists, nq, topk):
    right_num = 0
    for i in range(nq):  # 遍历每个查询
        for j in range(topk):  # 遍历 ids 中的每个 ID
            if ids[i][j] == gt_ids[i][j]: 
                right_num += 1
            elif dists[i][j] == gt_dists[i][j]:
                right_num += 1
    return right_num / (nq * topk)  # 计算召回率

def gt_check(bf_ids, bf_dists, gt_ids, gt_dists, nq, topk):
    for i in range(0, nq):
        for j in range(0, topk):
            if(bf_ids[i][j] != gt_ids[i][j]):
                print(f"{i} query in top{j} dismatch, bf doc id is: {bf_ids[i][j]}, gt doc id is: {gt_ids[i][j]}, bf doc dist is: {bf_dists[i][j]}, gt doc dist is: {gt_dists[i][j]}")

def is_rows_descending(arr):
    # 计算每行相邻元素的差值
    diffs = np.diff(arr, axis=1)
    
    # 检查差值是否都小于等于 0
    is_descending = np.all(diffs <= 0, axis=1)
    
    return is_descending


def write_indices_to_gtfile(I, D, fname, d):
    n = len(I)

    # 打开文件以写入二进制数据
    with open(fname, 'wb') as f:
        # 写 n 和 d
        f.write(struct.pack('II', n, d))   
        # 写 I，展平为一维数组并转换为 int32
        I_flat = np.array(I, dtype=np.int32).flatten()
        f.write(I_flat.tobytes())
        # 写 D，展平为一维数组并转换为 float32
        D_flat = np.array(D, dtype=np.float32).flatten()
        f.write(D_flat.tobytes())


def bf_gt():
    basefile = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    queryfile = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    gtfile = "/root/support-sparse-dataset/data/safe/safe_bf.dev.gt"
    gt_ids, gt_dists = knn_result_read(gtfile)

    topk = 10
    sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
    bf_index_params = json.dumps({
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 25000,
        "sparse_brute_force": {
        }
    })
    bf_search_params = json.dumps({"sparse_brute_force": {"num_threads": 104}})
    print("sparse brute force begin: ")
    brute_index = pyvsag.Index("sparse_brute_force", bf_index_params)
    brute_index.sparse_ivf_build(basefile)
    brute_ids, brute_dists = brute_index.batch_search(sizes[0], indptr, indices, data, topk, bf_search_params)

    bf_recall = cal_recall(brute_ids, gt_ids, sizes[0], topk)
    print(f"brute force recall ", bf_recall)


#ivf test
def ivf_test():
    # Data file
    b1 = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    q1 = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    g1 = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"
    b2 = ("/root/support-sparse-dataset/data/splade/base_small.csr")
    q2 = ("/root/support-sparse-dataset/data/splade/queries.dev.csr")
    g2 = ("/root/support-sparse-dataset/data/splade/base_small.dev.gt")

    basefiles = [b1, b2]
    queryfiles = [q1, q2]
    gtfiles = [g1, g2]

    index_params = json.dumps({
            "dtype": "float32",
            "metric_type": "ip",
            "dim": 30000,
            "sparse_ivf": {
                "doc_prune_strategy": {
                    "prune_type": "NotPrune"
                },
                "build_strategy": {
                    "build_type": "NotKmeans"
                }
            }
        })
    search_params = json.dumps(
    {
        "sparse_ivf": {
            "query_cut": 0,
            "num_threads": 104,
            "heap_factor": 0
        }
    }
    )

    topk = 10

    for (index, basefile) in enumerate(basefiles):
        queryfile = queryfiles[index]
        gtfile = gtfiles[index]

        gt_ids, gt_dists = knn_result_read(gtfile)

        print(f"Building sparse IVF index for {basefile}")
        index = pyvsag.Index("sparse_ivf", index_params)
        index.sparse_ivf_build(basefile)

        print(f"  Testing beigin...")
        # Perform search
        sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)

        begin_time = time.time()
        ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
        end_time = time.time()
        qps = sizes[0] / (end_time - begin_time)
        recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
        print(f"qps is {qps}, recall is {recall}")


#GLobal doc cut test
def doc_cut_test():
    # Data file
    b1 = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    q1 = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    g1 = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"
    b2 = "/root/support-sparse-dataset/data/splade/base_small.csr"
    q2 = "/root/support-sparse-dataset/data/splade/queries.dev.csr"
    g2 = "/root/support-sparse-dataset/data/splade/base_small.dev.gt"
    recall_file1 = "/root/support-sparse-dataset/results/doc_cut/safe_recall.pkl"
    recall_file2 = "/root/support-sparse-dataset/results/doc_cut/base_small_recall.pkl"
    qps_file1 = "/root/support-sparse-dataset/results/doc_cut/safe_qps.pkl"
    qps_file2 = "/root/support-sparse-dataset/results/doc_cut/base_small_qsp.pkl"

    basefiles = [b1, b2]
    queryfiles = [q1, q2]
    gtfiles = [g1, g2]
    recall_files = [recall_file1, recall_file2]
    qps_files = [qps_file1, qps_file2]

    topk = 10
    search_params = json.dumps(
    {
        "sparse_ivf": {
            "query_cut": 0,
            "num_threads": 104,
            "heap_factor": 0
        }
    }
    )
    
    #safe
    num_postings_1 = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
    #small
    num_postings_2 = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    all_num_postings = [num_postings_1, num_postings_2]

    for (i, basefile) in enumerate(basefiles):
        all_recall = []
        all_qps = []
        queryfile = queryfiles[i]
        sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
        gtfile = gtfiles[i]
        gt_ids, gt_dists = knn_result_read(gtfile)
        num_posting_values = all_num_postings[i]

        for num_postings in num_posting_values:
            index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "GlobalPrune",
                        "num_postings": num_postings,
                        "max_fraction": 1.5
                    },
                    "build_strategy": {
                        "build_type": "NotKmeans"
                    }
                }
            })
            print(f"Building sparse IVF index for {basefile} with num postings {num_postings}")

            index = pyvsag.Index("sparse_ivf", index_params)
            index.sparse_ivf_build(basefile)

            print(f"  Testing beigin...")
            # Perform search
            begin_time = time.time()
            ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
            end_time = time.time()
            qps = sizes[0] / (end_time - begin_time)
            all_qps.append(qps)
            recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
            all_recall.append(recall)

        save_list_to_pickle(all_qps, qps_files[i])
        save_list_to_pickle(all_recall, recall_files[i])


#query cut test
def query_cut_test():
    # Data file
    b1 = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    q1 = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    g1 = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"
    b2 = "/root/support-sparse-dataset/data/splade/base_small.csr"
    q2 = "/root/support-sparse-dataset/data/splade/queries.dev.csr"
    g2 = "/root/support-sparse-dataset/data/splade/base_small.dev.gt"
    recall_file1 = "/root/support-sparse-dataset/results/query_cut/safe_recall.pkl"
    recall_file2 = "/root/support-sparse-dataset/results/query_cut/base_small_recall.pkl"
    qps_file1 = "/root/support-sparse-dataset/results/query_cut/safe_qps.pkl"
    qps_file2 = "/root/support-sparse-dataset/results/query_cut/base_small_qsp.pkl"

    basefiles = [b1, b2]
    queryfiles = [q1, q2]
    gtfiles = [g1, g2]
    recall_files = [recall_file1, recall_file2]
    qps_files = [qps_file1, qps_file2]

    topk = 10
    index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "NotPrune"
                    },
                    "build_strategy": {
                        "build_type": "NotKmeans"
                    }
                }
            })

    #safe
    query_cuts_1 = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    #small
    query_cuts_2 = [3, 4, 5, 6, 7, 8, 9]
    all_query_cuts = [query_cuts_1, query_cuts_2]

    for (i, basefile) in enumerate(basefiles):
        all_recall = []
        all_qps = []
        queryfile = queryfiles[i]
        sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
        gtfile = gtfiles[i]
        gt_ids, gt_dists = knn_result_read(gtfile)
        query_cuts = all_query_cuts[i]

        print(f"Building sparse IVF index for {basefile}")

        index = pyvsag.Index("sparse_ivf", index_params)
        index.sparse_ivf_build(basefile)

        for query_cut in query_cuts:
            search_params = json.dumps({
                "sparse_ivf": {
                    "query_cut": query_cut,
                    "num_threads": 104,
                    "heap_factor": 0
                 }
            })

            print(f"  Testing with query cut = {query_cut}")
            # Perform search
            begin_time = time.time()
            ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
            end_time = time.time()
            qps = sizes[0] / (end_time - begin_time)
            all_qps.append(qps)
            recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
            all_recall.append(recall)

        save_list_to_pickle(all_qps, qps_files[i])
        save_list_to_pickle(all_recall, recall_files[i])


#k-means + summary energy test
def kmeans_test():
    # Data file
    b1 = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    q1 = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    g1 = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"
    b2 = "/root/support-sparse-dataset/data/splade/base_small.csr"
    q2 = "/root/support-sparse-dataset/data/splade/queries.dev.csr"
    g2 = "/root/support-sparse-dataset/data/splade/base_small.dev.gt"
    recall_file1 = "/root/support-sparse-dataset/results/kmeans/safe_recall.pkl"
    recall_file2 = "/root/support-sparse-dataset/results/kmeans/base_small_recall.pkl"
    qps_file1 = "/root/support-sparse-dataset/results/kmeans/safe_qps.pkl"
    qps_file2 = "/root/support-sparse-dataset/results/kmeans/base_small_qsp.pkl"

    basefiles = [b1, b2]
    queryfiles = [q1, q2]
    gtfiles = [g1, g2]
    recall_files = [recall_file1, recall_file2]
    qps_files = [qps_file1, qps_file2]

    topk = 10
    index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "NotPrune"
                    },
                    "build_strategy": {
                        "build_type": "Kmeans",
                        "centroid_fraction": 0.1,
                        "min_cluster_size": 2,
                        "summary_energy": 0.6
                    }
                }
            })

    heap_factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for (i, basefile) in enumerate(basefiles):
        all_recall = []
        all_qps = []
        queryfile = queryfiles[i]
        sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
        gtfile = gtfiles[i]
        gt_ids, gt_dists = knn_result_read(gtfile)

        print(f"Building sparse IVF index for {basefile}")

        index = pyvsag.Index("sparse_ivf", index_params)
        index.sparse_ivf_build(basefile)

        for heap_factor in heap_factors:
            search_params = json.dumps({
                "sparse_ivf": {
                    "query_cut": 0,
                    "num_threads": 104,
                    "heap_factor": heap_factor
                 }
            })

            print(f"  Testing with heap factor = {heap_factor}")
            # Perform search
            begin_time = time.time()
            ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
            end_time = time.time()
            qps = sizes[0] / (end_time - begin_time)
            all_qps.append(qps)
            recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
            all_recall.append(recall)

        save_list_to_pickle(all_qps, qps_files[i])
        save_list_to_pickle(all_recall, recall_files[i])

def safe_test():
    b1 = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    q1 = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    g1 = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"

    sizes, data, indices, indptr = read_sparse_matrix_fields(q1)
    gt_ids, gt_dists = knn_result_read(g1)

    topk = 10
    index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "NotPrune"
                    },
                    "build_strategy": {
                        "build_type": "Kmeans",
                        "centroid_fraction": 0.1,
                        "min_cluster_size": 2,
                        "summary_energy": 0.6
                    }
                }
            })

    print(f"Building sparse IVF index for {b1}")

    index = pyvsag.Index("sparse_ivf", index_params)
    index.sparse_ivf_build(b1)

    search_params = json.dumps({
                "sparse_ivf": {
                    "query_cut": 0,
                    "num_threads": 104,
                    "heap_factor": 0
                 }
    })

    print(f"  Testing...")
    # Perform search
    begin_time = time.time()
    ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
    end_time = time.time()
    qps = sizes[0] / (end_time - begin_time)
    recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
    print(f"recall is {recall}")

def doc_query_cut_safe():
    # Data file
    basefile = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    queryfile = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    gtfile = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"
    recall_file = "/root/support-sparse-dataset/results/doc_query/safe_recall.pkl"
    qps_file = "/root/support-sparse-dataset/results/doc_query/safe_qps.pkl"

    topk = 10

    num_postings = [100, 150, 200, 250, 300]
    query_cuts = [4, 9, 13]

    all_recall = []
    all_qps = []

    sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
    gt_ids, gt_dists = knn_result_read(gtfile)

    for num_posting in num_postings:
        for query_cut in query_cuts:
            index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "GlobalPrune",
                        "num_postings": num_posting,
                        "max_fraction": 1.5
                    },
                    "build_strategy": {
                        "build_type": "NotKmeans"
                    }
                }
            })

            search_params = json.dumps({
                "sparse_ivf": {
                    "query_cut": query_cut,
                    "num_threads": 104,
                    "heap_factor": 0
                 }
            })

            print(f"Building sparse IVF index for {basefile} with num_postings {num_posting}")

            index = pyvsag.Index("sparse_ivf", index_params)
            index.sparse_ivf_build(basefile)

            print(f"  Testing with query_cut = {query_cut}")
            # Perform search
            begin_time = time.time()
            ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
            end_time = time.time()
            qps = sizes[0] / (end_time - begin_time)
            all_qps.append(qps)
            recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
            all_recall.append(recall)

    save_list_to_pickle(all_qps, qps_file)
    save_list_to_pickle(all_recall, recall_file)


def doc_query_cut_base_small():
    # Data file
    basefile = "/root/support-sparse-dataset/data/splade/base_small.csr"
    queryfile = "/root/support-sparse-dataset/data/splade/queries.dev.csr"
    gtfile = "/root/support-sparse-dataset/data/splade/base_small.dev.gt"
    recall_file = "/root/support-sparse-dataset/results/doc_query/base_small_recall.pkl"
    qps_file = "/root/support-sparse-dataset/results/doc_query/base_small_qsp.pkl"

    topk = 10

    num_postings = [200, 400, 700]
    query_cuts = [3, 6, 9]

    all_recall = []
    all_qps = []

    sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
    gt_ids, gt_dists = knn_result_read(gtfile)

    for num_posting in num_postings:
        for query_cut in query_cuts:
            index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "GlobalPrune",
                        "num_postings": num_posting,
                        "max_fraction": 1.5
                    },
                    "build_strategy": {
                        "build_type": "NotKmeans"
                    }
                }
            })

            search_params = json.dumps({
                "sparse_ivf": {
                    "query_cut": query_cut,
                    "num_threads": 104,
                    "heap_factor": 0
                 }
            })

            print(f"Building sparse IVF index for {basefile} with num_postings {num_posting}")

            index = pyvsag.Index("sparse_ivf", index_params)
            index.sparse_ivf_build(basefile)

            print(f"  Testing with query_cut = {query_cut}")
            # Perform search
            begin_time = time.time()
            ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
            end_time = time.time()
            qps = sizes[0] / (end_time - begin_time)
            all_qps.append(qps)
            recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
            all_recall.append(recall)

    save_list_to_pickle(all_qps, qps_file)
    save_list_to_pickle(all_recall, recall_file)


def vsag_test_safe():
    # Data file
    basefile = "/root/support-sparse-dataset/data/safe/bge_safe_doc.csr"
    queryfile = "/root/support-sparse-dataset/data/safe/bge_safe_query.csr"
    gtfile = "/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt"
    recall_file = "/root/support-sparse-dataset/results/vsag/safe_recall.pkl"
    qps_file = "/root/support-sparse-dataset/results/vsag/safe_qps.pkl"

    sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
    gt_ids, gt_dists = knn_result_read(gtfile)

    num_postings = [150, 210, 270]
    heap_factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    query_cuts = [8, 10, 12]

    topk = 10

    all_recall = []
    all_qps = []

    for num_posting in num_postings:
        index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "GlobalPrune",
                        "num_postings": num_posting,
                        "max_fraction": 1.5
                    },
                    "build_strategy": {
                        "build_type": "Kmeans",
                        "centroid_fraction": 0.1,
                        "min_cluster_size": 2,
                        "summary_energy": 0.6
                    }
                }
        })

        print(f"Building sparse IVF index for {basefile} with num postings = {num_posting}")

        index = pyvsag.Index("sparse_ivf", index_params)
        index.sparse_ivf_build(basefile)

        for heap_factor in heap_factors:
            for query_cut in query_cuts:
                search_params = json.dumps({
                    "sparse_ivf": {
                        "query_cut": query_cut,
                        "num_threads": 104,
                        "heap_factor": heap_factor
                    }
                })

                print(f"  Testing with heap factor = {heap_factor} and query cut = {query_cut}")
                # Perform search
                begin_time = time.time()
                ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
                end_time = time.time()
                qps = sizes[0] / (end_time - begin_time)
                all_qps.append(qps)
                recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
                all_recall.append(recall)

    #save_list_to_pickle(all_qps, qps_file)
    #save_list_to_pickle(all_recall, recall_file)
    

def vsag_test_base_small():
    # Data file
    basefile = "/root/support-sparse-dataset/data/splade/base_small.csr"
    queryfile = "/root/support-sparse-dataset/data/splade/queries.dev.csr"
    gtfile = "/root/support-sparse-dataset/data/splade/base_small.dev.gt"
    recall_file = "/root/support-sparse-dataset/results/vsag/base_small_recall.pkl"
    qps_file = "/root/support-sparse-dataset/results/vsag/base_small_qsp.pkl"

    sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
    gt_ids, gt_dists = knn_result_read(gtfile)

    num_postings = [700]
    heap_factors = [0.1, 0.4, 0.7]
    query_cuts = [3, 6, 9]

    topk = 10

    all_recall = []
    all_qps = []

    for num_posting in num_postings:
        index_params = json.dumps({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": 30000,
                "sparse_ivf": {
                    "doc_prune_strategy": {
                        "prune_type": "GlobalPrune",
                        "num_postings": num_posting,
                        "max_fraction": 1.5
                    },
                    "build_strategy": {
                        "build_type": "Kmeans",
                        "centroid_fraction": 0.1,
                        "min_cluster_size": 2,
                        "summary_energy": 0.6
                    }
                }
        })

        print(f"Building sparse IVF index for {basefile} with num postings = {num_posting}")

        index = pyvsag.Index("sparse_ivf", index_params)
        index.sparse_ivf_build(basefile)

        for heap_factor in heap_factors:
            for query_cut in query_cuts:
                search_params = json.dumps({
                    "sparse_ivf": {
                        "query_cut": query_cut,
                        "num_threads": 1,
                        "heap_factor": heap_factor
                    }
                })

                print(f"  Testing with heap factor = {heap_factor} and query cut = {query_cut}")
                # Perform search
                begin_time = time.time()
                ids, dists = index.batch_search(sizes[0], indptr, indices, data, topk, search_params)
                end_time = time.time()
                qps = sizes[0] / (end_time - begin_time)
                all_qps.append(qps)
                recall = cal_recall(ids, dists, gt_ids, gt_dists, sizes[0], topk)
                all_recall.append(recall)

    save_list_to_pickle(all_qps, qps_file)
    save_list_to_pickle(all_recall, recall_file)

if __name__ == '__main__':
    vsag_test_base_small()

"""
using non_recursive_mutex = std::mutex;
using LockGuard = std::lock_guard<non_recursive_mutex>;
"""