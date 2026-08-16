"""
File-level Word2Vec semantic features.

- trains W2V on member-level data (all entities, not just first per file)
- file embedding = mean of all its entity embeddings
- L2-normalized before caching

Cache: cache_dir/{stem}_w2v.npy. Delete the cache to force recomputation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline.w2v_embeddings import W2VEmbeddingGenerator

_DATA_DIR = Path(__file__).parent.parent / "data"


def build_w2v_features(file_df, stem, cache_dir=None, data_dir=None,
                       vector_size=128, window=5, min_count=5, epochs=20,
                       max_df=0.9, max_vocab_size=2000):
    """Returns L2-normalized float tensor (len(file_df), vector_size), row-aligned to file_df."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "cache" / "w2v"
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{stem}_w2v.npy"

    if cache_path.exists():
        return torch.tensor(np.load(cache_path), dtype=torch.float)

    data_dir = Path(data_dir) if data_dir else _DATA_DIR

    # Load member-level CSV to average all entity embeddings per file
    raw_df = pd.read_csv(data_dir / f"{stem}.csv")
    raw_df["Member_Name"] = raw_df["Member_Name"].astype(str)
    raw_df["Code"] = raw_df.groupby("File")["Member_Name"].transform(
        lambda x: " ".join(x.dropna().astype(str))
    )
    base = raw_df[["File", "Entity", "Code"]].copy()
    base["File"]   = base["File"].astype(str)
    base["Entity"] = base["Entity"].astype(str)
    base["Code"]   = base["Code"].fillna("").astype(str)

    gen = W2VEmbeddingGenerator(base[["Entity", "Code"]], max_df=max_df)
    emb_map = gen.generate(
        vector_size=vector_size, window=window, min_count=min_count,
        sg=1, epochs=epochs, max_vocab_size=max_vocab_size,
    )

    zero = np.zeros(vector_size, dtype=np.float32)
    file2ents = base.groupby("File")["Entity"].apply(list).to_dict()
    file_vecs = {}
    for f, ents in file2ents.items():
        vecs = [emb_map.get(e, zero) for e in ents if e in emb_map]
        file_vecs[f] = np.mean(vecs, axis=0).astype(np.float32) if vecs else zero

    rows = [file_vecs.get(str(f), zero) for f in file_df["File"].astype(str)]
    vecs = np.stack(rows).astype(np.float32)
    normed = torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float), p=2, dim=1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, normed.numpy())

    return normed
