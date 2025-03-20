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
    recall_file3 = "/root/support-sparse-dataset/results/doc_query/safe_recall.pkl"
    qps_file3 = "/root/support-sparse-dataset/results/doc_query/safe_qps.pkl"
    recall_file4 = "/root/support-sparse-dataset/results/seismic/safe_recall.pkl"
    qps_file4 = "/root/support-sparse-dataset/results/seismic/safe_qps.pkl"
    recall_file5 = "/root/support-sparse-dataset/results/pyanns/safe_recall.pkl"
    qps_file5 = "/root/support-sparse-dataset/results/pyanns/safe_qps.pkl"
    recall_file6 = "/root/support-sparse-dataset/results/vsag/safe_recall.pkl"
    qps_file6 = "/root/support-sparse-dataset/results/vsag/safe_qps.pkl"
    

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

    combined = sorted(zip(doc_query_recall, doc_query_qps), key=lambda x: x[0])
    doc_query_recall_sorted, doc_query_qps_sorted = zip(*combined)
    doc_query_recall_sorted = list(doc_query_recall_sorted)
    doc_query_qps_sorted = list(doc_query_qps_sorted)

    combined = sorted(zip(vsag_recall, vsag_qps), key=lambda x: x[0])
    vsag_recall_sorted, vsag_qps_sorted = zip(*combined)
    vsag_recall_sorted = list(vsag_recall_sorted)
    vsag_qps_sorted = list(vsag_qps_sorted)

    plt.figure(figsize=(10, 6))
    plt.plot(doc_recall, doc_qps, label='Doc Cut', marker='o')
    plt.plot(query_recall, query_qps, label='Query Cut', marker='s')
    plt.plot(doc_query_recall_sorted, doc_query_qps_sorted, label='Doc Query Cut', marker='^')
    plt.plot(seismic_recall, seismic_qps, label='Seismic', marker='.')
    plt.plot(pyanns_recall, pyanns_qps, label='Pyanns', marker='.')
    plt.plot(vsag_recall_sorted, vsag_qps_sorted, label='vsag', marker='.')

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