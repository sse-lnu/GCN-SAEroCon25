"""
Shared evaluation utilities: per-run metrics and multi-run aggregation.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(ground_truth: Dict[str, str],
             predicted: Dict[str, Optional[str]],
             forced_pred: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Compute metrics for one run.

    ground_truth : file_path -> true component
    predicted    : file_path -> predicted component, or None if unmapped
    forced_pred  : file_path -> top-1 prediction regardless of threshold
    """
    mapped_true, mapped_pred = [], []
    forced_true, forced_list = [], []

    for fp, true in ground_truth.items():
        forced_true.append(true)
        forced_list.append(forced_pred[fp])
        p = predicted.get(fp)
        if p is not None:
            mapped_true.append(true)
            mapped_pred.append(p)

    if not mapped_true:
        return None

    total = len(forced_true)
    mapped_labels = sorted(set(mapped_true))
    full_labels = sorted(set(forced_true))

    def _scores(y_t, y_p, labels):
        kw = dict(labels=labels, zero_division=0)
        return {
            "f1_micro":        f1_score(y_t, y_p, average="micro",  **kw),
            "f1_macro":        f1_score(y_t, y_p, average="macro",  **kw),
            "precision_micro": precision_score(y_t, y_p, average="micro",  **kw),
            "precision_macro": precision_score(y_t, y_p, average="macro",  **kw),
            "recall_micro":    recall_score(y_t, y_p, average="micro",  **kw),
            "recall_macro":    recall_score(y_t, y_p, average="macro",  **kw),
        }

    m = _scores(mapped_true, mapped_pred, mapped_labels)
    f = _scores(forced_true, forced_list, full_labels)
    return {
        "coverage":           len(mapped_true) / total,
        **m,
        **{f"full_{k}": v for k, v in f.items()},
        "mapped_count":       len(mapped_true),
        "unmapped_count":     total - len(mapped_true),
        "test_size":          total,
        "mapped_class_count": len(mapped_labels),
        "test_class_count":   len(full_labels),
    }


def aggregate_runs(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean ± std across multiple runs."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metrics_list])),
            "std":  float(np.std([m[k] for m in metrics_list], ddof=min(1, len(metrics_list) - 1))),
        }
        for k in keys
    }
