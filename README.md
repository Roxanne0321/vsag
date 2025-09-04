# SINDI: Sparse Inverted Non-redundant Distance Index

<div align="center">
  <h3>Proof-of-Concept – Integrated into <a href="https://github.com/antgroup/vsag">VSAG</a></h3>
</div>

## Introduction

This repository hosts the **proof-of-concept and research prototype implementation** of the **SINDI**, an efficient algorithm for **Approximate Maximum Inner Product Search** (AMIPS) on sparse vectors. It was proposed in the paper *"SINDI: An Efficient Index for Sparse Vector Approximate Maximum Inner Product Search"*.

⚠️ **Note for Practitioners:**  
The SINDI index has been **fully integrated into the [VSAG](https://github.com/antgroup/vsag) framework**, which provides a **production-grade implementation** with ongoing maintenance, cross-version compatibility, and Python/C++ APIs.  
For deployment, benchmarking, or large-scale production scenarios, please refer to the  
**`Sindi` index within the VSAG repository** instead of this prototype.

---

## Parameters
### **Index Construction Parameters**
- **`--window_size` (λ)**：Window size controls how many vector IDs are processed per cache-local segment. Smaller λ ⇒ fewer random accesses but more window switches; larger λ ⇒ better posting list locality but more random writes. λ is tuned to the target hardware’s cache capacity and latency characteristics.
- **`--doc_prune_ratio` (α)**：Document pruning ratio. Retains the minimal set of high-value non-zero entries whose cumulative mass ≥ α × (total vector mass). Higher α preserves more entries (↑ recall, ↓ QPS); lower α prunes more aggressively (↑ QPS, slight recall drop). Typical: `0.1-1`.

### **Query/Search Parameters**
- **`--query_prune_ratio` (β)**：Query pruning ratio keeps the minimal set of high-value query entries covering β × (total query mass) during coarse retrieval. Lower β yields faster coarse search but may reduce recall before reranking. Typical: `0.1-1`.
- **`--reorder_size` (γ)**：Reordering candidate pool size. Number of coarse-stage candidates reranked with exact inner product to produce final top-k results. Larger γ increases recall but adds refinement cost. Common: k × 5 to k × 50. Typical: `50-2000`.

## Installation

You can build SINDI as part of the VSAG framework from source.

```bash
git clone -b sparse --single-branch https://github.com/Roxanne0321/vsag.git
cd vsag
make release
```

## 📂 Offline Evaluation Datasets

The offline benchmark datasets used in this work are derived from the **BigANN Sparse Vector Track**,  
which in turn is based on the **MSMARCO Passage Ranking** corpus encoded with the **SPLADE** model.

- **Base datasets** contain SPLADE-encoded sparse vectors for MS MARCO passages.
- **Query dataset** contains SPLADE-encoded sparse vectors for 6,980 development queries.
- Vectors are stored in **Compressed Sparse Row (CSR)** format, with dimensionality up to ~100,000  
  and an average of ~120 non-zero entries per base vector, ~49 per query vector.

### Download Links

| Dataset Name | Type | Size (vectors) | Download URL |
| :---: | :---: | :---: | :---: |
| `base_small` | Base | 100,000 | [Download](https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.csr.gz) |
| `base_1M`    | Base | 1,000,000 | [Download](https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_1M.csr.gz) |
| `base_full`  | Base   | 8,841,823 | [Download](https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.csr.gz) |
| `queries`    | Query  | 6,980 | [Download](https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz) |

## Usage
### Build Index
```bash
./build-release/sparse/scripts/sindi_index_build <basefile> <lambda> <alpha> <index_path>
```

### Generate Groung Truth
```base
./build-release/sparse/scripts/generate_gt <basefile> <queryfile> <gtfile> <topk>
```

### Search Index
```bash
./build-release/sparse/scripts/sindi_index_search <index_path> <queryfile> <gtfile> <beta> <gamma> <topk> <num_threads>
```