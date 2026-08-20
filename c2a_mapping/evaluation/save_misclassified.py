"""
RQ2 helper: per-file misclassification rate across runs, from an existing
{stem}_predictions.csv (written by experiments/run_experiments.py). Answers
"which files does the model get wrong, and how often" — the dependency-graph
contribution question — without needing anything beyond the predictions
already saved for RQ1.

Usage:
    python evaluation/save_misclassified.py bash
    python evaluation/save_misclassified.py bash --results-dir path/to/results
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR  = Path(__file__).parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"


def save_misclassified(stem, results_dir=None):
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    pred_path   = results_dir / f"{stem}_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions file at {pred_path} — run "
                                 f"experiments/run_experiments.py first.")

    df = pd.read_csv(pred_path)
    df = df[df["true_module"].notna()]  # only rows with ground truth can be scored

    summary = (
        df.groupby(["file", "model"])
        .agg(
            true_module=("true_module", "first"),
            times_in_test=("correct", "size"),
            times_correct=("correct", "sum"),
        )
        .reset_index()
    )
    summary["misclassification_rate"] = 1 - summary["times_correct"] / summary["times_in_test"]
    summary = summary.sort_values("misclassification_rate", ascending=False)

    out_path = results_dir / f"{stem}_misclassified.csv"
    summary.to_csv(out_path, index=False)
    print(f"{len(summary)} (file, model) rows -> {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Per-file misclassification rate from predictions.csv.")
    parser.add_argument("dataset", help="dataset stem, e.g. bash")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    save_misclassified(args.dataset, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
