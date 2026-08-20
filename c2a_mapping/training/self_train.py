"""
Two-phase self-training loop — exact match to C2A_pipeline.ipynb.

Phase 1 — Warmup : train on seed nodes only for warmup_epochs.
Phase 2 — Rounds : for self_train_rounds rounds:
              * revise existing pseudo-labels the model now disagrees with
              * promote new high-confidence orphans (conf > threshold)
              * retrain on seed + pseudo for self_train_epochs epochs
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score


def _edge_index_dict(data):
    """HeteroData.edge_index_dict raises KeyError when zero edge types are
    registered at all (e.g. MLP data with no edges) — fall back to {}."""
    try:
        return data.edge_index_dict
    except KeyError:
        return {}


def _labeled(true, pred):
    """Exclude nodes with no ground truth (sentinel -1) before scoring —
    metrics can only be computed where a true label actually exists."""
    mask = true != -1
    return true[mask], pred[mask]


def _score(true, pred, labels=None):
    """f1/precision/recall, or None-filled if there's nothing to score
    (e.g. every node in this subset is unlabeled)."""
    if len(true) == 0:
        return {k: None for k in (
            "f1_macro", "f1_micro", "precision_macro", "precision_micro",
            "recall_macro", "recall_micro",
        )}
    kw = dict(zero_division=1) if labels is None else dict(labels=labels, zero_division=1)
    return {
        "f1_macro":        f1_score(true, pred, average="macro",  **kw),
        "f1_micro":        f1_score(true, pred, average="micro",  **kw),
        "precision_macro": precision_score(true, pred, average="macro",  **kw),
        "precision_micro": precision_score(true, pred, average="micro",  **kw),
        "recall_macro":    recall_score(true, pred, average="macro",  **kw),
        "recall_micro":    recall_score(true, pred, average="micro",  **kw),
    }


def self_train(
    data,
    model,
    train_idx,               # seed indices — LongTensor already on device
    lr,
    device,
    threshold          = 0.95,
    warmup_epochs      = 100,
    self_train_rounds  = 4,
    self_train_epochs  = 30,
    verbose            = False,
    return_predictions = False,
    initial_pseudo     = None,
):
    model = model.to(device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    true_labels  = data["file"].y.clone()
    train_labels = torch.full_like(true_labels, -100)
    train_labels[train_idx] = true_labels[train_idx]

    num_files   = data["file"].x.size(0)
    seed_mask   = torch.zeros(num_files, dtype=torch.bool, device=device)
    seed_mask[train_idx] = True
    pseudo_mask = torch.zeros(num_files, dtype=torch.bool, device=device)

    if initial_pseudo:
        for idx, lbl in initial_pseudo.items():
            train_labels[idx] = lbl
            pseudo_mask[idx]  = True

    metrics_history = []

    # ── Phase 1: Warmup ───────────────────────────────────────────────────────
    if verbose:
        print(f"  Warmup: {warmup_epochs} epochs on {int(seed_mask.sum())} seed nodes")
    for _ in range(warmup_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, _edge_index_dict(data))
        loss   = F.cross_entropy(logits[train_idx], train_labels[train_idx])
        loss.backward()
        optimizer.step()

    # ── Phase 2: Self-training rounds ────────────────────────────────────────
    for rd in range(self_train_rounds):
        model.eval()
        with torch.no_grad():
            logits     = model(data.x_dict, _edge_index_dict(data))
            probs      = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        # Revise existing pseudo-labels the model now disagrees with
        existing = torch.where(pseudo_mask)[0]
        if existing.numel() > 0:
            high_conf = conf[existing] > threshold
            to_check  = existing[high_conf]
            changed   = pred[to_check] != train_labels[to_check]
            to_update = to_check[changed]
            if to_update.numel() > 0:
                train_labels[to_update] = pred[to_update]

        # Promote new high-confidence orphans
        candidate_mask = ~(seed_mask | pseudo_mask)
        candidates     = torch.where(candidate_mask)[0]
        if candidates.numel() == 0:
            break

        new_hi  = conf[candidates] > threshold
        new_idx = candidates[new_hi]
        if new_idx.numel() == 0:
            break

        prev_added = int(new_idx.numel())
        pseudo_mask[new_idx]  = True
        train_labels[new_idx] = pred[new_idx]

        train_idx = torch.where(seed_mask | pseudo_mask)[0]
        for _ in range(self_train_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x_dict, _edge_index_dict(data))
            loss   = F.cross_entropy(logits[train_idx], train_labels[train_idx])
            loss.backward()
            optimizer.step()

        pseudo_idx = torch.where(pseudo_mask)[0].tolist()
        pred_lbl   = train_labels[pseudo_idx].cpu().numpy()
        true_lbl   = true_labels[pseudo_idx].cpu().numpy()
        true_lbl, pred_lbl = _labeled(true_lbl, pred_lbl)
        mapped     = int((seed_mask | pseudo_mask).sum())
        remaining  = int((~(seed_mask | pseudo_mask)).sum())

        metrics = {
            "round":             rd,
            "new_pseudo":        prev_added,
            **_score(true_lbl, pred_lbl),
            "mapped":            mapped,
            "orphans_remaining": remaining,
        }
        metrics_history.append(metrics)
        if verbose:
            f1_str = f"{metrics['f1_macro']:.3f}" if metrics["f1_macro"] is not None else "n/a"
            print(f"  round {rd}: +{prev_added} pseudo  "
                  f"f1_macro={f1_str}  "
                  f"mapped={mapped}  orphans={remaining}")

    # ── Final evaluation on all non-seed nodes ────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x_dict, _edge_index_dict(data))
        final_pred   = final_logits.argmax(dim=1).cpu().numpy()
    final_true = true_labels.cpu().numpy()

    test_idx_np   = torch.where(~seed_mask)[0].cpu().numpy()
    mapped_idx_np = torch.where(pseudo_mask)[0].cpu().numpy()

    test_true,   test_pred   = final_true[test_idx_np],   final_pred[test_idx_np]
    mapped_true, mapped_pred = final_true[mapped_idx_np], final_pred[mapped_idx_np]

    test_true_l,   test_pred_l   = _labeled(test_true, test_pred)
    mapped_true_l, mapped_pred_l = _labeled(mapped_true, mapped_pred)

    test_labels_present   = np.unique(test_true_l) if len(test_true_l) > 0 else None
    mapped_labels_present = np.unique(mapped_true_l) if len(mapped_true_l) > 0 else test_labels_present

    final_mapped = int((seed_mask | pseudo_mask).sum())
    coverage     = final_mapped / num_files

    result = {
        "seed_size":      int(seed_mask.sum()),
        "n_test_nodes":   int(len(test_idx_np)),
        "n_mapped_nodes": int(len(mapped_idx_np)),
        "unmapped_count": int((~(seed_mask | pseudo_mask)).sum()),
        "coverage":       coverage,
        "rounds_run":     len(metrics_history),
        **{f"mapped_{k}": v for k, v in _score(mapped_true_l, mapped_pred_l, mapped_labels_present).items()},
        **_score(test_true_l, test_pred_l, test_labels_present),
        "metrics_history": metrics_history,
    }

    if return_predictions:
        pseudo_idx = torch.where(pseudo_mask)[0].tolist()
        result["predictions"] = (
            [{"node_idx": i, "true_label": true_labels[i].item(),
              "pred_label": train_labels[i].item(),
              "forced_pred_label": int(final_pred[i])} for i in pseudo_idx]
            + [{"node_idx": i, "true_label": true_labels[i].item(),
                "pred_label": None,
                "forced_pred_label": int(final_pred[i])}
               for i in torch.where(~(seed_mask | pseudo_mask))[0].tolist()]
        )

    return result
