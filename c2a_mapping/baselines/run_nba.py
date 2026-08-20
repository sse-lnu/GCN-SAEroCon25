"""
Runner for the NBA (Naive Bayes Architecture) baseline.

Configuration is loaded from experiments/config.json ("nba" section).
CLI flags override config values where provided.

By default, each run pulls its training set from the same pre-generated splits
GCN/GAT/MLP use (HetMap/data/splits/{Display}_splits.json), so NBA run i and
GCN/GAT/MLP run i see the literal same labeled set for a paired comparison. Pass
--use-fixed-splits=false to fall back to the old behavior (a fresh unseeded random
split per run).

Usage:
    python baselines/run_nba.py                         # all datasets, num_runs from config
    python baselines/run_nba.py --datasets ant,jabref   # specific datasets
    python baselines/run_nba.py --runs 100              # override number of runs
    python baselines/run_nba.py --out path/to/out.csv   # custom output path
    python baselines/run_nba.py --use-fixed-splits false # old unseeded-random splits

Results are saved to results/nba_results.csv (separate from GNN results).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.datasets import discover_datasets, load_file_level, display_name, split_ratio
from baselines.nba import (
    prepare_dataset, generate_file_split, run_iterative_nb_mapping,
    evaluate_mapping_results, predict_mapping, _worker_run,
)

_CFG_PATH   = Path(__file__).parent.parent / "experiments" / "config.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Full 11-system data folder (the module-level default in data_pipeline/datasets.py
# only has 3 systems checked into c2a_mapping/c2a_mapping/data/) — same folder
# fixed_splits_experiment.ipynb points GCN/GAT/MLP at.
DATA_DIR = Path(r"C:\Users\JABEERAK\Architecture_Recovery\data\datasets")

# Same splits directory and dataset-name resolution as fixed_splits_experiment.ipynb
# (cell-3's DISPLAY_TO_STEM) — the splits filenames on disk use "TeamMates"/"Chrome",
# which differ from data_pipeline/datasets.py's DISPLAY_NAMES ("T.Mates"/"Chromium").
SPLITS_DIR = Path(r"C:\Users\JABEERAK\Architecture_Recovery\HetMap\data\splits")
STEM_TO_SPLITS_DISPLAY = {
    "ant": "Ant", "argouml": "A.UML", "commons": "C.Img",
    "jabref": "Jabref", "lucene": "Lucene", "sweetHome": "SH-3D",
    "teammates": "TeamMates", "bash": "Bash",
    "hdf": "HDF", "hdc": "HDC", "chromium": "Chrome",
}


def _load_nba_cfg():
    return json.loads(_CFG_PATH.read_text())["nba"]


def _load_fixed_splits(stem):
    display = STEM_TO_SPLITS_DISPLAY[stem]
    path = SPLITS_DIR / f"{display}_splits.json"
    if not path.exists():
        raise FileNotFoundError(f"No splits file for {stem} at {path}")
    with open(path) as f:
        return json.load(f)


def run_nba_dataset(stem, n_runs=None, n_workers=None, use_fixed_splits=True):
    nba_cfg = _load_nba_cfg()
    n_runs  = n_runs or nba_cfg["num_runs"]

    file_df, file_dep = load_file_level(stem, DATA_DIR)
    prepared = prepare_dataset(file_df, file_dep)

    config = SimpleNamespace(
        use_cda                 = nba_cfg["use_cda"],
        use_node_text           = nba_cfg["use_node_text"],
        use_node_name           = nba_cfg["use_node_name"],
        use_arch_component_name = nba_cfg["use_arch_component_name"],
        min_word_length         = 3,
        mapping_threshold       = nba_cfg["mapping_threshold"],
        word_count              = nba_cfg["word_count"],
        use_stemming            = True,
        random_seed             = None,
    )

    args_list = []
    if use_fixed_splits:
        splits = _load_fixed_splits(stem)
        for run_id in range(n_runs):   # 0-indexed, matches GCN/GAT/MLP's RUN_IDS
            train_files = splits[str(run_id)]["train"]
            args_list.append((prepared, train_files, config, run_id, stem, display_name(stem)))
    else:
        for run_id in range(1, n_runs + 1):
            train_files, _ = generate_file_split(
                file_df,
                split_ratio      = nba_cfg["split_ratio"],
                min_train_points = nba_cfg["min_train_points"],
                max_train_points = nba_cfg["max_train_points"],
            )
            args_list.append((prepared, train_files, config, run_id, stem, display_name(stem)))

    workers = n_workers or max(1, mp.cpu_count() - 1)
    with mp.Pool(workers) as pool:
        raw = pool.map(_worker_run, args_list)

    return [r for r in raw if r is not None]


def run_nba_mapping_only(stem):
    """Predict modules for every file that doesn't already have one — no
    evaluation, no fixed splits, no repeats. Every file with a known Module
    is used as training data; the rest get a predicted_module."""
    nba_cfg = _load_nba_cfg()
    file_df, file_dep = load_file_level(stem, DATA_DIR)
    prepared = prepare_dataset(file_df, file_dep)

    config = SimpleNamespace(
        use_cda                 = nba_cfg["use_cda"],
        use_node_text           = nba_cfg["use_node_text"],
        use_node_name           = nba_cfg["use_node_name"],
        use_arch_component_name = nba_cfg["use_arch_component_name"],
        min_word_length         = 3,
        mapping_threshold       = nba_cfg["mapping_threshold"],
        word_count              = nba_cfg["word_count"],
        use_stemming            = True,
        random_seed             = None,
    )

    train_files = [f for f, m in prepared.file_labels.items() if m]
    predicted   = predict_mapping(prepared, train_files, config)
    return pd.DataFrame(
        {"File": list(predicted), "predicted_module": list(predicted.values())}
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the NBA baseline. Config from experiments/config.json."
    )
    parser.add_argument("--datasets", default="all",
                        help="comma-separated stems or 'all'")
    parser.add_argument("--runs", type=int, default=None,
                        help="runs per dataset (default: nba.num_runs in config.json)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: cpu_count - 1)")
    parser.add_argument("--out", default=str(RESULTS_DIR / "nba_results.csv"),
                        help="output CSV path")
    parser.add_argument("--use-fixed-splits", type=str, default="true",
                        choices=["true", "false"],
                        help="use the same pre-generated splits as GCN/GAT/MLP "
                             "(default: true). 'false' restores the old unseeded "
                             "random split per run.")
    parser.add_argument("--mapping-only", action="store_true",
                        help="skip evaluation entirely — predict a module for every "
                             "unlabeled file using all labeled files as training data. "
                             "No fixed splits, no repeats, no metrics required.")
    args = parser.parse_args()
    use_fixed_splits = args.use_fixed_splits.lower() == "true"

    stems = (
        [s for s in discover_datasets(DATA_DIR) if s != "ant"]
        if args.datasets == "all" else args.datasets.split(",")
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mapping_only:
        for stem in stems:
            print(f"\n--- {display_name(stem)} (mapping only) ---")
            df = run_nba_mapping_only(stem)
            out = RESULTS_DIR / f"{stem}_nba_mapping.csv"
            df.to_csv(out, index=False)
            print(f"  {len(df)} files mapped -> {out}")
        return

    all_rows = []
    for stem in stems:
        print(f"\n--- {display_name(stem)} ({args.runs or _load_nba_cfg()['num_runs']} runs, "
              f"{'fixed' if use_fixed_splits else 'random'} splits) ---")
        rows = run_nba_dataset(stem, n_runs=args.runs, n_workers=args.workers,
                                use_fixed_splits=use_fixed_splits)
        all_rows.extend(rows)
        if rows:
            f1_vals = [r["f1_macro"] for r in rows]
            print(f"  f1_macro: {sum(f1_vals)/len(f1_vals):.3f} (mean over {len(rows)} runs)")

    df = pd.DataFrame(all_rows)
    if os.path.exists(args.out):
        df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
