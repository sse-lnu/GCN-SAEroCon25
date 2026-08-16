"""
File-level folder-location signal features: splits each file's path into
folder-name tokens, then binary-vectorizes across all files.
"""
import os
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

_CAMEL_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _folder_tokens(f, segs, drop):
    f = f.replace("\\", "/").strip("/")
    if not f:
        return []

    name = f.split("/")[-1]
    base, ext = os.path.splitext(name)
    ext = ext.lower()

    if ext == ".java":
        out, seen = [], set()
        for s in segs.get(f, []):
            if s and s not in drop and s not in seen:
                out.append(s)
                seen.add(s)
        if out:
            return out
        return _CAMEL_SPLIT.findall(base.lower())

    parts = [p for p in f.split("/") if p]
    toks = list(parts[:-1])
    if ext in {".c", ".h", ".cpp", ".hpp"}:
        toks.extend(t for t in base.lower().split(".") if t)
    else:
        toks.append(base.lower())

    out, seen = [], set()
    for t in toks:
        if t and t not in drop and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_loc_features(file_df):
    """Returns binary float tensor of shape (len(file_df), vocab_size), row-aligned to file_df."""
    files = file_df["File"].astype(str).str.replace("\\", "/", regex=False).tolist()
    segs = {f: [s for s in "/".join(f.split("/")[:-1]).split("/") if s] for f in files}

    seg_lists = list(segs.values())
    drop = {"src", "main", "java"}
    if seg_lists and all(seg_lists):
        for tokens in zip(*seg_lists):
            if all(t == tokens[0] for t in tokens):
                drop.add(tokens[0])
            else:
                break

    loc_texts = [" ".join(_folder_tokens(f, segs, drop)) or "root" for f in files]
    features = CountVectorizer(binary=True).fit_transform(loc_texts).toarray().astype(np.float32)
    return torch.tensor(features, dtype=torch.float)
