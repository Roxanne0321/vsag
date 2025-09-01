import os
import argparse
import numpy as np

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

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D

def read_pkl(file_name):
    with open(file_name, "rb") as f:
        sizes = np.fromfile(f, dtype=np.int64, count=2)
        num = sizes[0]
        topk = sizes[1]
        int_array = np.fromfile(f, dtype=np.int64, count=topk*num).reshape((num, topk))
        float_array = np.fromfile(f, dtype=np.float32, count=topk*num).reshape((num, topk))

    return sizes, int_array, float_array

def cal_recall(ids, gt_ids, nq, gt_topk):
    hit_count = 0
    
    for i in range(nq):
        # 将 ground truth 转换为集合
        gt_set = set(gt_ids[i])
        
        # 转换 predicted ids 为集合，然后计算交集
        predicted_set = set(ids[i])
        hits = predicted_set.intersection(gt_set)
        
        # 统计命中次数
        hit_count += len(hits)

    return hit_count / (nq * gt_topk)  # 计算召回率

def recall_print(dataset, topk, n_cut, query_cut, reorder_k):
    print(f"topk: {topk}")
    gtfile = f"sparse/data/{dataset}_top{topk}.dev.gt"
    gt_ids, gt_dists = knn_result_read(gtfile)

    print(f"n_cut: {n_cut}")
    print(f"query_cut: {query_cut}")
    print(f"reorder_k: {reorder_k}")

    filename = f"sparse/results/{dataset}_nc_{n_cut}_qc_{query_cut}_rk_{reorder_k}_top{topk}.pkl"
    sizes, ids, dists = read_pkl(filename)
    recall = cal_recall(ids, gt_ids, sizes[0], topk)
    print(f"recall: {recall}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate recall from K-NN results.")
    parser.add_argument("--dataset", type=str, required=True, help="dataset to be tested.")
    parser.add_argument("--topk", type=int, required=True, help="Top K value for nearest neighbors.")
    parser.add_argument("--n_cut", type=int, required=True, help="N cut value.")
    parser.add_argument("--query_cut", type=int, required=True, help="Query cut value.")
    parser.add_argument("--reorder_k", type=int, required=True, help="Reorder K value.")

    args = parser.parse_args()

    recall_print(args.dataset, args.topk, args.n_cut, args.query_cut, args.reorder_k)