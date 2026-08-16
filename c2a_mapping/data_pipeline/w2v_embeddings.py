from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def build_default_stopwords(extra: Optional[Iterable[str]] = None) -> Set[str]:
    """
    Returns a default English stopword set + common code/Java boilerplate terms.
    Tries NLTK stopwords first; falls back to a small built-in set if NLTK is unavailable.
    """
    base: Set[str] = set()
    try:
        from nltk.corpus import stopwords  # type: ignore

        base = set(stopwords.words("english"))
    except Exception:
        base = {
            "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "as",
            "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
            "from", "by", "at", "not", "no", "but", "if", "then", "else", "when",
        }

    code_terms = {
        "name", "logger", "log", "class", "public", "private", "protected",
        "void", "static", "final", "return", "int", "string", "boolean",
        "true", "false", "null", "new", "get", "set", "value", "data",
        "object", "var", "args", "this", "main", "system", "out", "print", "println",
        "import", "package", "extends", "implements", "throws", "try", "catch",
        "exception", "file", "read", "write", "input", "output", "create", "loader",
        "strict", "update", "field", "default", "comment", "response", "entry",
        "edit", "copy", "start", "button", "check", "delete", "show",
        "begin", "double", "float", "char", "interface", "enum", "goto",
        "super", "abstract", "synchronized", "volatile", "finally", "throw", "case",
        "break", "continue", "switch", "while", "for", "do", "instanceof", "assert",
        "const", "unknown", "clinit", "init", "<clinit>", "<init>", "self",
        "getname", "setname", "getvalue", "setvalue", "header", "footer",
    }

    out = set(w.lower() for w in base) | set(w.lower() for w in code_terms)
    if extra:
        out |= set(w.lower() for w in extra)
    return out


DEFAULT_STOPWORDS: Set[str] = build_default_stopwords()


class W2VEmbeddingGenerator:
    """
    Learns word2vec embeddings over identifier tokens extracted from code and averages them per entity.
    Expects a dataframe with columns:
      - Entity: unique identifier
      - Code: raw code string
    """

    IDENT_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    CAMEL_SNAKE_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")

    def __init__(
        self,
        df: pd.DataFrame,
        max_df: float = 0.9,
        stop_words: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
    ):
        if "Entity" not in df.columns or "Code" not in df.columns:
            raise ValueError("DataFrame must contain 'Entity' and 'Code' columns.")

        self.df = df.copy()
        self.df["Entity"] = self.df["Entity"].astype(str)
        self.df["Code"] = self.df["Code"].fillna("").astype(str)
        self.df.drop_duplicates(subset=["Entity"], inplace=True)

        self.max_df = float(max_df)
        self.min_token_len = int(min_token_len)
        if stop_words is None:
            stop_words = DEFAULT_STOPWORDS
        self.stop_words = set(w.lower() for w in stop_words)

        self.tokens_per_entity: Dict[str, List[str]] = {}
        self._build_corpus()

    def _split_identifier(self, text: str) -> List[str]:
        out: List[str] = []
        for piece in text.split("_"):
            for part in self.CAMEL_SNAKE_SPLIT.findall(piece):
                t = part.lower()
                if len(t) >= self.min_token_len and t not in self.stop_words:
                    out.append(t)
        return out

    def _tokens_from_code(self, code: str) -> List[str]:
        if not code:
            return []
        toks: List[str] = []
        for ident in self.IDENT_REGEX.findall(code):
            if ident.isupper():
                continue
            toks.extend(self._split_identifier(ident))
        return toks

    def _build_corpus(self) -> None:
        raw_tokens: Dict[str, List[str]] = {}
        for _, row in self.df.iterrows():
            entity = row["Entity"]
            raw_tokens[entity] = self._tokens_from_code(row["Code"])

        token_df: Dict[str, int] = {}
        for toks in raw_tokens.values():
            for tok in set(toks):
                token_df[tok] = token_df.get(tok, 0) + 1

        num_entities = max(1, len(raw_tokens))
        threshold = self.max_df * num_entities
        allowed = {tok for tok, c in token_df.items() if c < threshold}

        self.tokens_per_entity = {
            ent: [t for t in toks if t in allowed] for ent, toks in raw_tokens.items()
        }

    def generate(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        sg: int = 1,
        epochs: int = 10,
        max_vocab_size: Optional[int] = 2000,
        workers: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        sentences: List[Sequence[str]] = list(self.tokens_per_entity.values())
        if not sentences:
            return {}

        if workers is None:
            cpu = os.cpu_count() or 1
            workers = max(1, cpu - 1)

        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=workers,
            max_vocab_size=max_vocab_size,
        )

        zero = np.zeros(vector_size, dtype=np.float32)
        embeddings: Dict[str, np.ndarray] = {}
        for entity, toks in self.tokens_per_entity.items():
            vecs = [model.wv[t] for t in toks if t in model.wv]
            embeddings[entity] = np.mean(vecs, axis=0).astype(np.float32) if vecs else zero
        return embeddings
