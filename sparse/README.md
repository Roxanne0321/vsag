# Project Overview

Welcome to the Sparse Index Testing Demo. This project provides a comprehensive framework for testing sparse indexes using various datasets and configurations. The primary objective is to efficiently manage datasets, generate indexes, and evaluate query performance.

## Directory Structure

Here's a brief overview of the key directories and files contained within this project:

### `sparse/`

This directory contains essential elements for sparse index testing.

- **`data/`**: Dedicated to storing necessary datasets required for testing:
  - `base`: The base dataset used for index construction.
  - `query`: The dataset containing queries that are run against the sparse index.
  - `groundtruth`: The reference dataset used for verifying the accuracy of the query results.

- **`index/`**: Stores the generated indexes from the test processes, used subsequently for efficient query execution.

- **`results/`**: Contains the results of running queries against the sparse indexes, providing insight into performance and accuracy based on various retrieval parameters.

### `recall_cal.py`

This Python script calculates the recall rate of query results, measuring the accuracy of queries performed against the groundtruth dataset. Utilize this script to understand and enhance your index retrieval methods' precision.

## Usage

### Setup

Begin by ensuring all necessary datasets are placed within their respective directories in `sparse/data/`.

### Index Creation

Construct sparse indexes using the given datasets.

**Parameters:**

- `--dataset`: The dataset file used for building the index (e.g., `base_1M.csr`).
- `--window_size`: Specifies the window size for data set partitioning, related to cache size (e.g., `100000`).
- `--n_cut`: ; commonly set between 25-40 for `base_1M`.

**Command Line Example:**

```bash
./build-release/sparse/examples/sparse_index_build --dataset base_1M --window_size 100000 --n_cut 40
```

### Query Execution

Execute queries against the constructed indexes to retrieve results.

**Parameters:**

- `--topk`: Number of nearest neighbors to recall (e.g., `10`).
- `--query_cut`: Proportion of the query to retain, where `0.2` retains 20% of the query.
- `--reorder_k`: Number of data points to reorder, typically between 300 and 1000.

**Command Line Example:**

```bash
./build-release/sparse/examples/sparse_index_search --dataset base_1M --window_size 100000 --n_cut 40 --topk 10 --query_cut 0.2 --reorder_k 300
```

### Recall Calculation

Use this step to calculate the recall rate of the query results.

**Command Line Example:**

```bash
python sparse/recall_cal.py --dataset base_1M --topk 10 --query_cut 20 --n_cut 40 --reorder_k 300
```