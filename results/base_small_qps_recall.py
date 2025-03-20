import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_list_from_pickle(filename):
    with open(filename, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

def filter_recall_less(qps, recall):
    # 转换为 numpy 数组
    qps = np.array(qps)
    recall = np.array(recall)

    # 过滤掉 recall 小于 0.9 的值
    mask = recall >= 0.9
    filtered_qps = qps[mask]
    filtered_recall = recall[mask]

    return filtered_qps.tolist(), filtered_recall.tolist()

def filter_qps_by_recall(qps, recall):
    # 确保输入都是numpy数组
    qps = np.array(qps)
    recall = np.array(recall)

    # 定义区间边界
    min_recall = np.min(recall)
    max_recall = np.max(recall)
    intervals = np.linspace(min_recall, max_recall, num=11)

    # 初始化结果列表
    filtered_qps = []
    filtered_recall = []

    # 在每个区间内选择qps最高的点
    for i in range(len(intervals) - 1):
        # 找到属于当前区间的点
        mask = (recall >= intervals[i]) & (recall < intervals[i+1])
        if np.any(mask):
            max_qps_index = np.argmax(qps[mask])
            filtered_qps.append(qps[mask][max_qps_index])
            filtered_recall.append(recall[mask][max_qps_index])

    return filtered_qps, filtered_recall


def qps_recall():
    recall_file1 = "/root/support-sparse-dataset/results/doc_cut/base_small_recall.pkl"
    qps_file1 = "/root/support-sparse-dataset/results/doc_cut/base_small_qsp.pkl"
    recall_file2 = "/root/support-sparse-dataset/results/query_cut/base_small_recall.pkl"
    qps_file2 = "/root/support-sparse-dataset/results/query_cut/base_small_qsp.pkl"
    recall_file3 = "/root/support-sparse-dataset/results/doc_query/base_small_recall.pkl"
    qps_file3 = "/root/support-sparse-dataset/results/doc_query/base_small_qsp.pkl"
    recall_file4 = "/root/support-sparse-dataset/results/seismic/1/base_small_recall.pkl"
    qps_file4 = "/root/support-sparse-dataset/results/seismic/1/base_small_qps.pkl"
    recall_file5 = "/root/support-sparse-dataset/results/pyanns/base_small_recall.pkl"
    qps_file5 = "/root/support-sparse-dataset/results/pyanns/base_small_qps.pkl"
    recall_file6 = "/root/support-sparse-dataset/results/vsag/base_small_recall.pkl"
    qps_file6 = "/root/support-sparse-dataset/results/vsag/base_small_qsp.pkl"

    doc_recall = load_list_from_pickle(recall_file1)
    doc_qps = load_list_from_pickle(qps_file1)
    query_recall = load_list_from_pickle(recall_file2)
    query_qps = load_list_from_pickle(qps_file2)
    doc_query_recall = load_list_from_pickle(recall_file3)
    doc_query_qps = load_list_from_pickle(qps_file3)
    seismic_recall = load_list_from_pickle(recall_file4)
    seismic_qps = load_list_from_pickle(qps_file4)
    pyanns_recall = load_list_from_pickle(recall_file5)
    pyanns_qps = load_list_from_pickle(qps_file5)
    vsag_recall = load_list_from_pickle(recall_file6)
    vsag_qps = load_list_from_pickle(qps_file6)

    f_dq_qps, f_dq_recall = filter_qps_by_recall(doc_query_qps, doc_query_recall)
    f_seismic_qps, f_seismic_recall = filter_qps_by_recall(seismic_qps, seismic_recall)
    f_vsag_qps, f_vsag_recall = filter_qps_by_recall(vsag_qps, vsag_recall)

    f_doc_qps, f_doc_recall = filter_recall_less(doc_qps, doc_recall)
    f_query_qps, f_query_recall = filter_recall_less(query_qps, query_recall)
    f_dq_qps, f_dq_recall = filter_recall_less(f_dq_qps, f_dq_recall)
    f_seismic_qps, f_seismic_recall = filter_recall_less(f_seismic_qps, f_seismic_recall)
    f_vsag_qps, f_vsag_recall = filter_recall_less(f_vsag_qps, f_vsag_recall)

    plt.figure(figsize=(10, 6))
    #plt.plot(f_doc_recall, f_doc_qps, label='Doc Cut', marker='o')
    #plt.plot(f_query_recall, f_query_qps, label='Query Cut', marker='s')
    #plt.plot(f_dq_recall, f_dq_qps, label='Doc Query Cut', marker='^')
    #plt.plot(f_seismic_recall, f_seismic_qps, label='Seismic', marker='^')
    plt.plot(pyanns_recall, pyanns_qps, label='Pyanns', marker='.')
    plt.plot(f_vsag_recall, f_vsag_qps, label='vsag', marker='.')

    # Labels and title
    plt.xlabel('Recall')
    plt.ylabel('qps')
    plt.title('Comparison of qps vs Recall')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('results/base_small_qps_recall.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def print_pkl():
    recall_file6 = "/root/support-sparse-dataset/results/vsag/base_small_recall.pkl"
    qps_file6 = "/root/support-sparse-dataset/results/vsag/base_small_qsp.pkl"
    vsag_recall = load_list_from_pickle(recall_file6)
    vsag_qps = load_list_from_pickle(qps_file6)
    print(vsag_recall)
    print(vsag_qps)


if __name__ == '__main__':
    qps_recall()