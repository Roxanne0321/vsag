import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_list_from_pickle(filename):
    with open(filename, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

def safe_plot():
    recall_file1 = "/root/support-sparse-dataset/results/doc_cut/safe_recall.pkl"
    doc_recall = load_list_from_pickle(recall_file1)
    doc_cut = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
    plt.plot(doc_cut, doc_recall, marker='o')

    # Labels and title
    plt.xlabel('doc cut')
    plt.ylabel('recall')
    plt.title('impact doc cut to recall')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('results/doc_cut/safe_doc_cut_recall.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def small_plot():
    recall_file1 = "/root/support-sparse-dataset/results/doc_cut/base_small_recall.pkl"
    doc_recall = load_list_from_pickle(recall_file1)
    doc_cut = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    plt.plot(doc_cut, doc_recall, marker='o')

    # Labels and title
    plt.xlabel('doc cut')
    plt.ylabel('recall')
    plt.title('impact doc cut to recall')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('results/doc_cut/small_doc_cut_recall.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

if __name__ == '__main__':
    small_plot()