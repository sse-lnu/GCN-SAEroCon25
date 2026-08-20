# Code-to-Architecture Mapping with Graph Neural Networks

## Project Overview

A semi-supervised pipeline for mapping source files to architectural modules from a file-level dependency graph. Two Graph Neural Network encoders — **GCN** and **GAT** — are trained with an iterative self-training loop, alongside an **MLP** baseline (features only, no graph) and an **NBA** (Naive Bayes) baseline for comparison against a non-neural method.

### Motivation

Software architecture documentation tends to go stale as a system evolves, and manually recovering the file-to-module mapping does not scale to large codebases. We address this by inferring the mapping directly from the dependency structure and the code's own textual content, with an approach that can extend a partial, incomplete mapping to unlabeled files without requiring full ground truth for the entire system.

### Approach

Each system is modeled as a heterogeneous, multi-relational file-dependency graph:
- **Nodes** are source files.
- **Edges** are typed static dependencies (e.g., calls, imports, inheritance) between files.
- **Node features** combine Word2Vec embeddings of the code's identifiers/text with a folder-location signal derived from each file's path.

GCN/GAT encoders run an iterative self-training loop: warm up on a labeled seed set, then repeatedly promote high-confidence pseudo-labels and retrain. The MLP baseline uses the same features without any graph structure. NBA runs an independent Naive-Bayes-based iterative mapping for comparison.

### Repository Contents

```
c2a_mapping/
├── baselines/                       # NBA (Naive Bayes) baseline
├── data/                            # Sample datasets
├── data_pipeline/                   # Feature extraction, graph construction, label encoding
├── evaluation/                      # Metric helpers, misclassification analysis
├── experiments/                     # config.json + GNN experiment runner
├── models/                          # GCN, GAT, MLP definitions
├── training/                        # Self-training loop
├── run_mapping.py                   # Get automatic source file mapping
└── requirements.txt
```

---

## Research Questions

- **RQ1 — Effectiveness & Scaling** *(primary)*: How effectively does GNN-based mapping perform relative to non-graph baselines, and how does it scale?
- **RQ2 — Dependency-graph contribution** *(secondary, still verifiable)*: What does the dependency graph contribute beyond textual features, and where does it help or hurt on a per-file basis?
- **RQ3 — Design sensitivity**: How sensitive is the approach to modeling dependency types separately, aggregation depth, and edge direction?

### RQ1 — Effectiveness & Scaling

```bash
cd c2a_mapping/
python experiments/run_experiments.py          # GCN, GAT, MLP — all datasets, default config
python baselines/run_nba.py                    # NBA baseline, same datasets
```

Compare `results/{gcn,gat}_hetero_results.csv` and `results/mlp_results.csv` against `results/nba_results.csv` on `f1_macro`/`f1_micro`/`precision_macro`/`recall_macro`.

### RQ2 — Dependency-graph contribution

```bash
cd c2a_mapping/
python evaluation/save_misclassified.py bash   # per-file misclassification rate, from an existing predictions.csv
```

Reads `results/{stem}_predictions.csv` (written by `run_experiments.py`) and writes `results/{stem}_misclassified.csv`: how often each file was misclassified across runs, letting you see where the model — and by extension the dependency graph — struggles on a given system.

### RQ3 — Design sensitivity

```bash
python experiments/run_experiments.py --ablation homo_directed
python experiments/run_experiments.py --ablation hetero_undirected
python experiments/run_experiments.py --ablation hetero_reversed
python experiments/run_experiments.py --ablation hetero_directed_1layer
```

See the Ablations table below for what each one varies.

### Mapping without evaluation

```bash
cd c2a_mapping/
python run_mapping.py --dataset bash --model gat
```

Every file with a known Module is used as training data (no held-out split — there's nothing to hold out for); every other file gets a `predicted_module`. No metrics are computed, since there's no assumption that a full ground truth exists. Output: `results/{stem}_mapping.csv` (`File`, `predicted_module` only). The NBA baseline has the same option: `python baselines/run_nba.py --mapping-only`.

---

## Setup

```bash
cd c2a_mapping/
pip install -r requirements.txt
```

Tested against `torch==2.4.0` / `torch-geometric==2.7.0`; both support CUDA or CPU.

---

## Quick Start

```bash
cd c2a_mapping/
python experiments/run_experiments.py --dataset bash --runs 5
```
---

## Running Experiments

`experiments/run_experiments.py`:
- `--ablation <name>` — one of the ablations below (default: `hetero_directed`).
- `--dataset <stem>` — a single dataset instead of every configured one.
- `--runs <n>` — override the run count for that dataset.

`baselines/run_nba.py`:
- `--datasets <stem,stem,...>` or `all`.
- `--runs <n>` — runs per dataset (default: `nba.num_runs` in `config.json`).
- `--mapping-only` — skip evaluation, predict every unlabeled file (see above).

`run_mapping.py`:
- `--dataset <stem>` (required).
- `--model gcn|gat|mlp` (default: `gat`).
- `--data-dir` / `--cache-dir` — override the default `data/` / `cache/w2v` locations.

Results are written per model/ablation to `results/{gcn,gat}_{hetero,homo,undirected,reversed}_results.csv` and `results/mlp_results.csv`.

### Ablations

| Name | Mode | Directed |
|------|------|----------|
| `hetero_directed` *(default)* | one conv per relation type | yes |
| `homo_directed` | merged adjacency | yes |
| `hetero_undirected` | one conv per relation type | no |
| `hetero_reversed` | one conv per relation type, edges flipped | yes |
| `hetero_directed_1layer` | one conv per relation type, 1 message-passing layer | yes |

### Default Configuration

From `experiments/config.json`:
- **GCN / GAT**: hidden size 256, 2 message-passing layers, dropout 0.01 (GAT: 4 attention heads).
- **MLP**: hidden size 256, dropout 0.0.
- **Training**: learning rate 0.001, self-training promotion threshold 0.95, 100 warmup epochs, 4 self-training rounds of 30 epochs each.
- **Runs**: 200 per dataset/model by default.

---

## Input and Output

Each dataset consists of two CSVs:

```
c2a_mapping/data/
├── {stem}.csv         # columns: File, Entity, Module, Member_Name
└── {stem}_deps.csv    # columns: Source_File, Target_File, Dependency_Type, Dependency_Count
```

`Module` may be missing for some files — those become unlabeled nodes that `run_mapping.py` / `run_nba.py --mapping-only` will predict rather than requiring for evaluation.

Output files, all under `results/`:
- `{model}_{ablation}_results.csv` — metrics per run (`f1_macro`, `f1_micro`, `precision_macro`, `recall_macro`, plus `mapped_`-prefixed variants for the confidently-mapped subset).
- `{stem}_predictions.csv` — per-file, per-run `true_module` / `pred_module` / `correct`, the input to RQ2's misclassification analysis.
- `{stem}_mapping.csv` / `{stem}_nba_mapping.csv` — `File` / `predicted_module` only, no ground truth required.
- `nba_results.csv` — NBA's own results, with the same mapped-subset naming as above and `full_`-prefixed forced-prediction variants.

---

## Data

Most systems used here are available in the [GAER repository](https://github.com/sse-lnu/GAER/tree/master/data). `argouml`, `commons`, and `sweetHome` are included directly in `c2a_mapping/data/` because they aren't present there.
