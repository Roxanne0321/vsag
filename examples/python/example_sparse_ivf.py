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

def cal_recall(ids, gt_ids, nq, topk):
    right_num = 0
    for i in range(0, nq):
        for j in range(0, topk):
            if(ids[i][j] == gt_ids[i][j]):
                right_num = right_num + 1
    return right_num / nq * topk

def safe_test():
    #Data file
    basefile = ("/root/support-sparse-dataset/data/safe/bge_safe_doc.csr")   
    queryfile = ("/root/support-sparse-dataset/data/safe/bge_safe_query.csr")
    gtfile = ("/root/support-sparse-dataset/data/safe/bge_safe_recall.dev.gt")

    # Declaring index
    index_params = json.dumps({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 25000
    })

    #Build index
    index = pyvsag.Index("sparse_ivf", index_params)
    index.sparse_iv_build(basefile)

    #search
    topk = 10
    sizes, data, indices, indptr = read_sparse_matrix_fields(queryfile)
    ids = index.batch_search(sizes.shape[0], indptr, indices, data, 10)

    #cal recall
    gt_ids, _ = knn_result_read(gtfile)
    recall = cal_recall(ids, gt_ids, sizes.shape[0], topk)
    print("topk is ", recall)


if __name__ == '__main__':
    safe_test()

