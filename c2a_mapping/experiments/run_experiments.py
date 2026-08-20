"""
Config-driven, resumable experiment runner.

Reads experiments/config.json, runs the models listed in 'models_to_run'
for the ablation set in 'active_ablation', and saves CSVs to results/.

Usage:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --ablation homo_directed
    python experiments/run_experiments.py --dataset ant --runs 5

To switch ablation, either pass --ablation <name> or change 'active_ablation'
in config.json. All ablation definitions are in the 'ablations' list.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.datasets import discover_datasets, load_file_level
from training.train import train

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def load_config(path):
    with open(path) as f:
        return json.load(f)


def get_ablation(config, name=None):
    target = name or config["active_ablation"]
    for a in config["ablations"]:
        if a["name"] == target:
            return {k: v for k, v in a.items() if k != "name"}
    raise ValueError(f"Ablation '{target}' not found in config. "
                     f"Available: {[a['name'] for a in config['ablations']]}")


def load_existing(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def completed_runs(df, stem, model, ablation):
    if df.empty:
        return 0
    mask = (
        (df["data_name"] == stem) &
        (df["model"] == model.upper()) &
        (df["mode"] == ablation["mode"]) &
        (df["directed"] == ablation["directed"]) &
        (df["reverse_relations"] == ablation["reverse_relations"])
    )
    subset = df[mask]
    if subset.empty:
        return 0
    return subset["run_id"].nunique() if "run_id" in subset.columns else len(subset)


def run(config_path, ablation_override=None, dataset_override=None, runs_override=None):
    config   = load_config(config_path)
    ablation = get_ablation(config, ablation_override)

    results_base = Path(__file__).parent.parent / config.get("results_dir", "results")
    cache_dir    = Path(__file__).parent.parent / config.get("cache_dir", "cache/w2v")
    results_base.mkdir(parents=True, exist_ok=True)

    stems = (
        [dataset_override] if dataset_override
        else (discover_datasets() if config["datasets"] == "all" else config["datasets"])
    )
    dataset_overrides = config.get("dataset_overrides", {})
    flush_every = config.get("flush_every", 5)
    train_cfg   = config.get("training", {})
    model_params = config.get("model_params", {})

    for stem in stems:
        results_path     = results_base / f"{stem}_results.csv"
        predictions_path = results_base / f"{stem}_predictions.csv"
        df_results      = load_existing(results_path)
        df_predictions  = load_existing(predictions_path)

        file_df, _ = load_file_level(stem)

        n_runs = runs_override or dataset_overrides.get(stem, {}).get("runs", config["default_runs"])

        for model_type in config["models_to_run"]:
            completed = completed_runs(df_results, stem, model_type, ablation)
            if completed >= n_runs:
                print(f"{stem} [{model_type}] already {completed}/{n_runs} — skipping.")
                continue

            remaining  = n_runs - completed
            next_start = completed + 1
            model_cfg  = model_params.get(model_type, {})
            print(f"\n{stem} [{model_type}] running {remaining} runs from {next_start}")

            while remaining > 0:
                chunk = min(flush_every, remaining)
                rows, lenc, fdf = train(
                    stem, model_type, ablation,
                    model_cfg=model_cfg, train_cfg=train_cfg,
                    n_runs=chunk, start_run=next_start,
                    cache_dir=str(cache_dir), return_predictions=True,
                )

                pred_rows = []
                for row in rows:
                    run_id = row["run_id"]
                    for p in row.pop("predictions", []):
                        true_m = (lenc.inverse_transform([p["true_label"]])[0]
                                  if p["true_label"] != -1 else None)
                        pred_m = (lenc.inverse_transform([p["pred_label"]])[0]
                                  if p["pred_label"] is not None else None)
                        pred_rows.append({
                            "run_id": run_id, "data_name": stem,
                            "model": model_type.lower(),
                            "file": fdf["File"].iloc[p["node_idx"]],
                            "true_module": true_m, "pred_module": pred_m,
                            "correct": pred_m == true_m,
                        })

                df_results = pd.concat([df_results, pd.DataFrame(rows)], ignore_index=True)
                df_results = df_results.sort_values("run_id").drop_duplicates(
                    subset=["data_name", "model", "mode", "directed",
                            "reverse_relations", "run_id"],
                    keep="first",
                )
                df_results.to_csv(results_path, index=False)

                if pred_rows:
                    df_predictions = pd.concat([df_predictions, pd.DataFrame(pred_rows)],
                                               ignore_index=True)
                    df_predictions = df_predictions.drop_duplicates(
                        subset=["data_name", "model", "run_id", "file"], keep="first"
                    )
                    df_predictions.to_csv(predictions_path, index=False)

                next_start += chunk
                remaining  -= chunk

        print(f"\n{stem}: {len(df_results)} rows saved -> {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=str(DEFAULT_CONFIG))
    parser.add_argument("--ablation", default=None,
                        help="Ablation name from config (overrides active_ablation)")
    parser.add_argument("--dataset",  default=None)
    parser.add_argument("--runs",     type=int, default=None)
    args = parser.parse_args()
    run(args.config, ablation_override=args.ablation,
        dataset_override=args.dataset, runs_override=args.runs)


if __name__ == "__main__":
    main()
