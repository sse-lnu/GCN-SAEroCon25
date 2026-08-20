"""
Shared label encoding for files whose Module is only partially known.

Files with a missing/blank Module get the sentinel -1 instead of a class
index, so they can flow through graph construction and self-training as
unlabeled nodes rather than crashing at LabelEncoder.fit/transform time.
"""
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

UNLABELED = -1


def encode_labels(modules):
    """Encode a Module Series to a LongTensor, sentinel UNLABELED (-1) for
    missing values. LabelEncoder is fit only on the known (non-null) labels.

    Returns (y: LongTensor, label_encoder: LabelEncoder).
    """
    modules   = modules.astype(object)
    known     = modules.notna() & (modules.astype(str).str.strip() != "") & (modules.astype(str) != "None")
    encoder   = LabelEncoder().fit(modules[known].astype(str))
    encoded   = np.full(len(modules), UNLABELED, dtype=np.int64)
    encoded[known.to_numpy()] = encoder.transform(modules[known].astype(str))
    return torch.tensor(encoded, dtype=torch.long), encoder
