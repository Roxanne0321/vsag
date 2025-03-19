import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_list_from_pickle(filename):
    with open(filename, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

def qps_recall():
    recall_file1 = "/root/support-sparse-dataset/results/doc_cut/safe_recall.pkl"
    qps_file1 = "/root/support-sparse-dataset/results/doc_cut/safe_qps.pkl"
    recall_file2 = "/root/support-sparse-dataset/results/query_cut/safe_recall.pkl"
    qps_file2 = "/root/support-sparse-dataset/results/query_cut/safe_qps.pkl"
    recall_file3 = "/root/support-sparse-dataset/results/kmeans/safe_recall.pkl"
    qps_file3 = "/root/support-sparse-dataset/results/kmeans/safe_qps.pkl"
    recall_file4 = "/root/support-sparse-dataset/results/vsag/safe_recall.pkl"
    qps_file4 = "/root/support-sparse-dataset/results/vsag/safe_qps.pkl"

    doc_recall = load_list_from_pickle(recall_file1)
    doc_qps = load_list_from_pickle(qps_file1)
    query_recall = load_list_from_pickle(recall_file2)
    query_qps = load_list_from_pickle(qps_file2)
    kmeans_recall = load_list_from_pickle(recall_file3)
    kmeans_qps = load_list_from_pickle(qps_file3)
    vsag_recall = load_list_from_pickle(recall_file4)
    vsag_qps = load_list_from_pickle(qps_file4)

    plt.figure(figsize=(10, 6))
    plt.plot(doc_recall, doc_qps, label='Doc Cut', marker='o')
    plt.plot(query_recall, query_qps, label='Query Cut', marker='s')
    plt.plot(kmeans_recall, kmeans_qps, label='K-Means Heap Factor', marker='^')
    plt.plot(vsag_recall, vsag_qps, label='vsag', marker='.')

    # Labels and title
    plt.xlabel('Recall')
    plt.ylabel('qps')
    plt.title('Comparison of qps vs Recall')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('results/safe_qps_recall.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

if __name__ == '__main__':
    qps_recall()