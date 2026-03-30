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
- **`lambda` (λ)**：Window size used for segmented accumulation. **Valid range: `[10000, 100000]`**.
- **`alpha` (α)**：Document pruning ratio for base vectors. For each base vector, keep the minimal prefix (after sorting by value descending) whose cumulative mass reaches `alpha * total_mass`. **Valid range: `[0, 1]`**.
- **`use_reorder`**：Whether to perform exact reranking on coarse candidates. Default is `true`.

### **Query/Search Parameters**
- **`beta` (β)**：Query term retain ratio in coarse retrieval. Keep top `int(query_dim * beta)` terms by value in each query. **Valid range: `[0, 1]`**.
- **`gamma` (γ)**：Candidate pool size kept after coarse stage. If `gamma < topk`, internal behavior is `gamma = topk`.
- **`num_threads`**：Thread count used in search (`omp_set_num_threads`).

> Note: `prune_stragy` is **not** used by current SINDI parameter parser.

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
./build-release/sparse/scripts/sindi_index_build <basefile> <lambda> <alpha> <use_reorder> <index_path>
```

### Generate Groung Truth
```base
./build-release/sparse/scripts/generate_gt <basefile> <queryfile> <gtfile> <topk>
```

### Search Index
```bash
./build-release/sparse/scripts/sindi_index_search <index_path> <queryfile> <gtfile> <beta> <gamma> <topk> <num_threads>
```