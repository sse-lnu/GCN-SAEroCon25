"""
Seed-set construction: every class gets >=1 representative and total
seed size is always >= MIN_SEED_SIZE.
"""
import numpy as np
import torch

MIN_SEED_SIZE = 25


def generate_split(labels, split_ratio, min_seed_size=MIN_SEED_SIZE):
    """labels: 1D array/tensor of int class labels for every node.
    Returns (seed_idx, orphan_idx) as LongTensors."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    n = len(labels)
    all_nodes = np.arange(n)

    target_train = max(int(np.ceil(split_ratio * n)), min_seed_size)
    target_train = min(target_train, n)

    shuffled = all_nodes.copy()
    np.random.shuffle(shuffled)
    train_set = set(shuffled[:target_train].tolist())

    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        if not any(idx in train_set for idx in idxs):
            train_set.add(int(np.random.choice(idxs)))

    if len(train_set) < min_seed_size:
        remaining = [i for i in shuffled if i not in train_set]
        train_set.update(remaining[: min_seed_size - len(train_set)])

    train_idx = np.array(sorted(train_set), dtype=np.int64)
    test_idx = np.setdiff1d(all_nodes, train_idx)
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def assert_hard_constraints(labels, seed_idx, min_seed_size=MIN_SEED_SIZE):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    seed_labels = labels[seed_idx.cpu().numpy() if isinstance(seed_idx, torch.Tensor) else seed_idx]
    assert len(seed_idx) >= min_seed_size, f"seed size {len(seed_idx)} < {min_seed_size}"
    assert set(seed_labels.tolist()) == set(labels.tolist()), "not every class represented in seed set"
