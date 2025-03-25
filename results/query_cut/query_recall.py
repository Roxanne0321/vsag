import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_list_from_pickle(filename):
    with open(filename, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

def safe_plot():
    recall_file1 = "/root/support-sparse-dataset/results/query_cut/safe_recall.pkl"
    doc_recall = load_list_from_pickle(recall_file1)
    doc_cut = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    plt.plot(doc_cut, doc_recall, marker='o')

    # Labels and title
    plt.xlabel('query cut')
    plt.ylabel('recall')
    plt.title('impact query cut to recall')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('results/query_cut/safe_query_cut_recall.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def small_plot():
    recall_file1 = "/root/support-sparse-dataset/results/query_cut/base_small_recall.pkl"
    doc_recall = load_list_from_pickle(recall_file1)
    doc_cut = [3, 4, 5, 6, 7, 8, 9]
    plt.plot(doc_cut, doc_recall, marker='o')

    # Labels and title
    plt.xlabel('query cut')
    plt.ylabel('recall')
    plt.title('impact query cut to recall')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('results/query_cut/small_query_cut_recall.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

if __name__ == '__main__':
    safe_plot()