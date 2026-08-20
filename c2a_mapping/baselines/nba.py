"""
NBA: Naive Bayes Architecture mapping (fast, config-driven re-implementation).

Incorporates all performance fixes over the original nba_mapping_experiments.py:
  1. Dependency indices built once per round  — O(D + N*C*deg) vs O(N*C*D)
  2. node_text computed once per orphan       — not once per (orphan, component)
  3. Batch sklearn inference                  — one matrix call per orphan, not C calls
  4. No token-count explosion when word_count=False

Usage:
    python nba.py --config config.json
    python nba.py --config config.json --save-mappings
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


STOP_WORDS = {"<init>", "<clinit>", "tmp", "temp"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeClass:
    name: str
    texts: List[str] = field(default_factory=list)
    is_inner: bool = False


@dataclass
class Node:
    name: str
    classes: List[CodeClass] = field(default_factory=list)
    mapping: str = ""
    clustering: str = ""
    clustering_type: str = ""
    attractions: List[float] = field(default_factory=list)

    def logic_name(self) -> str:
        for code_class in self.classes:
            if not code_class.is_inner:
                return code_class.name
        return self.name


@dataclass(frozen=True)
class Dependency:
    source: str
    target: str
    dep_type: str
    count: int = 1


@dataclass
class Component:
    name: str


@dataclass
class Architecture:
    components: List[Component]

    def component_names(self) -> List[str]:
        return [c.name for c in self.components]


@dataclass
class MappingResult:
    node_name: str
    predicted_component: Optional[str]
    attractions: Dict[str, float]


# ---------------------------------------------------------------------------
# Java-compatible RNG (matches original NBA paper implementation)
# ---------------------------------------------------------------------------

class JavaRandom:
    def __init__(self, seed: int) -> None:
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)

    def next_int(self, bound: Optional[int] = None) -> int:
        if bound is None:
            value = self._next(32)
            return value - (1 << 32) if value >= (1 << 31) else value
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & (bound - 1)) == 0:
            return (bound * self._next(31)) >> 31
        while True:
            bits  = self._next(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value


def java_shuffle(values: List[object], rand: JavaRandom) -> None:
    for i in range(len(values), 1, -1):
        j = rand.next_int(i)
        values[i - 1], values[j] = values[j], values[i - 1]


# ---------------------------------------------------------------------------
# Vectorizer
# ---------------------------------------------------------------------------

class PythonWordVectorizer:
    DEFAULT_DELIMITERS = " \r\n\t.,;:'\"()?!"

    def __init__(self, *, word_count: bool = False) -> None:
        self.word_count = word_count

    def fit_transform(self, texts, *, labels=None, class_names=None):
        return [self._vectorize(t) for t in texts]

    def transform(self, texts, *, labels=None, class_names=None):
        return [self._vectorize(t) for t in texts]

    def _vectorize(self, text: str) -> Counter:
        tokens = [t for t in self._tokenize(text) if t]
        return Counter(tokens) if self.word_count else Counter(set(tokens))

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens, start = [], 0
        for i, ch in enumerate(text):
            if ch in self.DEFAULT_DELIMITERS:
                if start < i:
                    tokens.append(text[start:i])
                start = i + 1
        if start < len(text):
            tokens.append(text[start:])
        return tokens


# ---------------------------------------------------------------------------
# Stemmer
# ---------------------------------------------------------------------------

class NLTKPorterStemmer:
    def __init__(self) -> None:
        try:
            from nltk.stem import PorterStemmer
        except ImportError as exc:
            raise ImportError(
                "NLTK is not installed. Use: conda install nltk  or  pip install nltk"
            ) from exc
        self._stemmer = PorterStemmer()

    def stem_many(self, words: Sequence[str]) -> List[str]:
        return [self._stemmer.stem(w) for w in words]

    def __call__(self, word: str) -> str:
        return self._stemmer.stem(word)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class SklearnMultinomialNaiveBayes:
    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_names: List[str] = []
        self._documents: List[Tuple[str, Counter]] = []
        self._vectorizer = None
        self._classifier = None

    def fit(self, documents: Sequence[Tuple[str, Counter]]) -> None:
        self._documents = list(documents)
        if not self._documents:
            raise ValueError("Cannot fit classifier with no documents")
        self._fit_classifier(class_priors=None)

    def set_class_priors(self, priors: Dict[str, float]) -> None:
        if not self._documents:
            raise ValueError("Classifier must be fitted before setting priors")
        self._fit_classifier(class_priors=priors)

    def predict_proba(self, token_counts: Counter) -> Dict[str, float]:
        features = self._vectorizer.transform([dict(token_counts)])
        probs    = self._classifier.predict_proba(features)[0]
        return {label: float(p) for label, p in zip(self._classifier.classes_, probs)}

    def predict_proba_batch(self, token_counts_list: List[Counter]) -> List[Dict[str, float]]:
        """One sklearn call for all samples — much faster than N single calls."""
        features    = self._vectorizer.transform([dict(c) for c in token_counts_list])
        prob_matrix = self._classifier.predict_proba(features)
        classes     = self._classifier.classes_
        return [{label: float(p) for label, p in zip(classes, row)} for row in prob_matrix]

    def _fit_classifier(self, class_priors) -> None:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.naive_bayes import MultinomialNB

        labels       = [lbl for lbl, _ in self._documents]
        feature_dicts = [dict(cnt) for _, cnt in self._documents]
        class_names  = sorted(self.class_names) if self.class_names else sorted(set(labels))

        self._vectorizer = DictVectorizer(sparse=True)
        features = self._vectorizer.fit_transform(feature_dicts)

        prior_values = (
            None if class_priors is None
            else [class_priors.get(lbl, 0.0) for lbl in class_names]
        )

        self._classifier = MultinomialNB(alpha=self.alpha, fit_prior=True, class_prior=prior_values)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log",
                                    category=RuntimeWarning)
            self._classifier.partial_fit(features, labels, classes=class_names)


# ---------------------------------------------------------------------------
# NBMapper
# ---------------------------------------------------------------------------

class NBMapper:
    """Multinomial NB mapper with all fast-path optimisations applied."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        use_cda: bool = True,
        use_node_text: bool = True,
        use_node_name: bool = True,
        use_arch_component_name: bool = True,
        min_word_length: int = 3,
        mapping_threshold: float = 0.9,
        word_count: bool = False,
        stemmer=None,
        use_stemming: bool = True,
        vectorizer=None,
        classifier=None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.architecture            = architecture
        self.use_cda                 = use_cda
        self.use_node_text           = use_node_text
        self.use_node_name           = use_node_name
        self.use_arch_component_name = use_arch_component_name
        self.min_word_length         = min_word_length
        self.mapping_threshold       = mapping_threshold if 0.0 <= mapping_threshold <= 1.0 else 0.9
        self.word_count              = word_count
        self.stemmer                 = stemmer if stemmer is not None else (NLTKPorterStemmer() if use_stemming else None)
        self.vectorizer              = vectorizer if vectorizer is not None else PythonWordVectorizer(word_count=word_count)
        self.classifier              = classifier if classifier is not None else SklearnMultinomialNaiveBayes(alpha=1.0)
        self.random_seed             = random_seed

    def map(self, nodes: Sequence[Node], dependencies: Sequence[Dependency]) -> List[MappingResult]:
        nodes_by_name    = {n.name: n for n in nodes}
        initially_mapped = [n for n in nodes if n.mapping and n.clustering]
        orphans          = [n for n in nodes if n.mapping and not n.clustering]

        if self.random_seed is not None:
            rand = JavaRandom(self.random_seed)
            java_shuffle(initially_mapped, rand)
            java_shuffle(orphans, rand)

        if not initially_mapped:
            raise ValueError("At least one initially clustered node is required")

        # Build dep indices once — avoids O(N*C*D) re-scanning.
        deps_by_source: Dict[str, List[Dependency]] = defaultdict(list)
        deps_by_target: Dict[str, List[Dependency]] = defaultdict(list)
        for dep in dependencies:
            deps_by_source[dep.source].append(dep)
            deps_by_target[dep.target].append(dep)
        clustered_names: Set[str] = {n.name for n in initially_mapped}

        training_texts = self._get_training_texts(
            initially_mapped, nodes_by_name, deps_by_source, deps_by_target, clustered_names
        )
        training_docs = self._vectorize_documents(training_texts, fit=True)
        self.classifier.class_names = self.architecture.component_names()
        self.classifier.fit(training_docs)
        self.classifier.set_class_priors(self._initial_class_distribution(initially_mapped))

        results: List[MappingResult] = []
        component_names = self.architecture.component_names()

        for orphan in orphans:
            node_text = self._get_node_words(orphan)  # computed once per orphan

            docs: List[Tuple[str, str]] = []
            for comp in component_names:
                cda = self._get_unmapped_cda_words(
                    orphan=orphan, candidate_component=comp,
                    clustered_names=clustered_names, nodes_by_name=nodes_by_name,
                    deps_by_source=deps_by_source, deps_by_target=deps_by_target,
                )
                docs.append((comp, f"{cda}{node_text}".strip()))

            vectorized = self._vectorize_documents(docs, fit=False)
            all_probs  = self.classifier.predict_proba_batch([cnt for _, cnt in vectorized])

            attractions = {component_names[i]: all_probs[i][component_names[i]]
                           for i in range(len(component_names))}
            orphan.attractions = [attractions[c] for c in component_names]
            predicted = self._do_auto_mapping_abs_threshold(orphan, component_names)
            results.append(MappingResult(
                node_name=orphan.name, predicted_component=predicted, attractions=attractions
            ))

        return results

    def _do_auto_mapping_abs_threshold(self, orphan: Node, component_names: Sequence[str]) -> Optional[str]:
        if len(orphan.attractions) < 2:
            if orphan.attractions and orphan.attractions[0] > self.mapping_threshold:
                orphan.clustering = component_names[0]
                orphan.clustering_type = "Automatic"
                return component_names[0]
            return None

        sorted_idx = sorted(range(len(orphan.attractions)),
                            key=lambda i: orphan.attractions[i], reverse=True)
        best, second = sorted_idx[0], sorted_idx[1]

        if (orphan.attractions[best] > self.mapping_threshold
                and orphan.attractions[best] > orphan.attractions[second]):
            orphan.clustering = component_names[best]
            orphan.clustering_type = "Automatic"
            return component_names[best]
        return None

    def _get_training_texts(self, clustered_nodes, nodes_by_name,
                            deps_by_source, deps_by_target, clustered_names):
        docs = []
        if self.use_arch_component_name:
            for comp in self.architecture.components:
                text = self._get_arch_component_words(comp.name)
                if text:
                    docs.append((comp.name, text))
        for node in clustered_nodes:
            cda   = self._get_mapped_cda_words(source=node, clustered_names=clustered_names,
                                               nodes_by_name=nodes_by_name,
                                               deps_by_source=deps_by_source,
                                               deps_by_target=deps_by_target)
            docs.append((node.clustering, cda))
            docs.append((node.clustering, self._get_node_words(node)))
        return docs

    def _vectorize_documents(self, documents, *, fit: bool):
        texts  = [text for _, text in documents]
        labels = [lbl  for lbl,  _ in documents]
        cnames = self.architecture.component_names()
        counters = (
            self.vectorizer.fit_transform(texts, labels=labels, class_names=cnames)
            if fit else
            self.vectorizer.transform(texts, labels=labels, class_names=cnames)
        )
        return [(lbl, cnt) for (lbl, _), cnt in zip(documents, counters)]

    def _initial_class_distribution(self, clustered_nodes):
        total  = len(clustered_nodes)
        counts = Counter(n.clustering for n in clustered_nodes)
        return {c.name: counts.get(c.name, 0) / total for c in self.architecture.components}

    def _get_node_words(self, node: Node) -> str:
        parts = []
        if self.use_node_text:
            for cc in node.classes:
                for text in cc.texts:
                    if text.lower() not in STOP_WORDS:
                        t = self._de_camel_case(text, self.min_word_length)
                        if t:
                            parts.append(t)
        if self.use_node_name:
            t = self._de_camel_case(node.logic_name().replace(".", " "), 0)
            if t:
                parts.append(t)
        return " ".join(parts).strip()

    def _get_arch_component_words(self, name: str) -> str:
        return self._de_camel_case(name, 0) if self.use_arch_component_name else ""

    def _get_unmapped_cda_words(self, *, orphan, candidate_component,
                                clustered_names, nodes_by_name, deps_by_source, deps_by_target):
        if not self.use_cda:
            return ""
        out = self._dependency_string_from_node(
            source_name=orphan.name, source_component=candidate_component,
            clustered_names=clustered_names, nodes_by_name=nodes_by_name,
            deps_by_source=deps_by_source)
        inc = self._dependency_string_to_node(
            target_name=orphan.name, target_component=candidate_component,
            clustered_names=clustered_names, nodes_by_name=nodes_by_name,
            deps_by_target=deps_by_target)
        return f"{out} {inc}"

    def _get_mapped_cda_words(self, *, source, clustered_names, nodes_by_name,
                              deps_by_source, deps_by_target):
        if not self.use_cda:
            return ""
        out = self._dependency_string_from_node(
            source_name=source.name, source_component=source.clustering,
            clustered_names=clustered_names, nodes_by_name=nodes_by_name,
            deps_by_source=deps_by_source)
        inc = self._dependency_string_to_node(
            target_name=source.name, target_component=source.clustering,
            clustered_names=clustered_names, nodes_by_name=nodes_by_name,
            deps_by_target=deps_by_target)
        return f"{out} {inc}"

    def _dependency_string_from_node(self, *, source_name, source_component,
                                     clustered_names, nodes_by_name, deps_by_source):
        tokens = []
        for dep in deps_by_source.get(source_name, []):
            if dep.target == source_name or dep.target not in clustered_names:
                continue
            rel = self._rel(source_component, dep.dep_type,
                            nodes_by_name[dep.target].clustering)
            if self.word_count:
                tokens.extend([rel] * dep.count)
            else:
                tokens.append(rel)
        return " ".join(tokens)

    def _dependency_string_to_node(self, *, target_name, target_component,
                                   clustered_names, nodes_by_name, deps_by_target):
        tokens = []
        for dep in deps_by_target.get(target_name, []):
            if dep.source == target_name or dep.source not in clustered_names:
                continue
            src_comp = nodes_by_name[dep.source].clustering
            if src_comp == target_component:
                continue
            rel = self._rel(src_comp, dep.dep_type, target_component)
            if self.word_count:
                tokens.extend([rel] * dep.count)
            else:
                tokens.append(rel)
        return " ".join(tokens)

    def _rel(self, from_comp: str, dep_type: str, to_comp: str) -> str:
        return from_comp.replace(".", "") + dep_type + to_comp.replace(".", "")

    def _de_camel_case(self, text: str, min_length: int) -> str:
        for ch in "0123456789":
            text = text.replace(ch, " ")
        text = text.replace("_", " ").replace("-", " ").replace(".", " ")

        parts = []
        for piece in text.split():
            words = []
            for w in self._split_camel(piece):
                if not w:
                    continue
                lo = w.lower()
                if len(lo) < min_length or "$" in lo:
                    continue
                words.append(lo)
            if not words:
                continue
            if self.stemmer is not None and hasattr(self.stemmer, "stem_many"):
                words = list(self.stemmer.stem_many(words))
            elif self.stemmer is not None:
                words = [self.stemmer(w) for w in words]
            parts.extend(words)
        return " ".join(parts)

    def _split_camel(self, piece: str) -> List[str]:
        if not piece:
            return []
        words, start = [], 0
        for i in range(1, len(piece)):
            cur, prev = piece[i], piece[i - 1]
            nxt = piece[i + 1] if i + 1 < len(piece) else ""
            if cur.isupper() and (not prev.isupper() or (nxt and nxt.islower())):
                words.append(piece[start:i])
                start = i
        words.append(piece[start:])
        return words


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def morgan_file_logic_name(file_path: str) -> str:
    base = Path(str(file_path).replace("\\", "/")).name
    return base.rsplit(".", 1)[0] if "." in base else base


def clean_file_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep every file with a File id — Module may be missing (unlabeled
    file, no ground truth), in which case it's kept as an empty string
    rather than dropped, so it still becomes a mappable node."""
    c = df.dropna(subset=["File"]).copy()
    c["File"]   = c["File"].astype(str)
    c["Module"] = c["Module"].apply(lambda m: str(m) if pd.notna(m) and str(m).strip() else "")
    return c


def build_architecture(df: pd.DataFrame) -> Architecture:
    modules = clean_file_table(df)["Module"]
    modules = sorted(m for m in modules.unique() if m)
    return Architecture(components=[Component(name=m) for m in modules])


def build_nodes(df: pd.DataFrame, clustered_files: Dict[str, str]) -> Tuple[List[Node], Dict[str, str]]:
    cleaned = clean_file_table(df)
    file_labels = (
        cleaned[["File", "Module"]]
        .drop_duplicates(subset=["File"])
        .set_index("File")["Module"]
        .to_dict()
    )
    member_names: Dict[str, List[str]] = {}
    if "Member_Name" in cleaned.columns:
        member_names = (
            cleaned.dropna(subset=["Member_Name"])
            .drop_duplicates(subset=["File", "Member_Name"])
            .groupby("File")["Member_Name"]
            .apply(lambda vals: sorted({str(v) for v in vals if str(v).strip()}))
            .to_dict()
        )

    nodes = []
    for fp, module in file_labels.items():
        cc = CodeClass(name=morgan_file_logic_name(str(fp)),
                       texts=member_names.get(fp, []), is_inner=False)
        nodes.append(Node(
            name=str(fp), classes=[cc],
            mapping=str(fp),
            clustering=clustered_files.get(str(fp), ""),
        ))
    return nodes, {str(fp): str(m) for fp, m in file_labels.items()}


def collapse_dependencies_to_file_level(
    df_dep: pd.DataFrame, valid_files: Sequence[str]
) -> List[Dependency]:
    valid = set(map(str, valid_files))
    deps  = df_dep.copy()
    deps["Source_File"] = deps["Source_File"].astype(str)
    deps["Target_File"] = deps["Target_File"].astype(str)
    deps = deps[deps["Source_File"].isin(valid) & deps["Target_File"].isin(valid)].copy()

    if "Dependency_Count" in deps.columns:
        deps["Dependency_Count"] = pd.to_numeric(deps["Dependency_Count"], errors="coerce").fillna(1).astype(float)
    else:
        deps["Dependency_Count"] = 1.0

    grouped = (
        deps.groupby(["Source_File", "Target_File", "Dependency_Type"], as_index=False)
            ["Dependency_Count"].sum().reset_index(drop=True)
    )
    return [
        Dependency(source=str(r.Source_File), target=str(r.Target_File),
                   dep_type=str(r.Dependency_Type), count=max(1, int(round(float(r.Dependency_Count)))))
        for r in grouped.itertuples(index=False)
    ]


@dataclass
class PreparedDataset:
    architecture:   Architecture
    dependencies:   List[Dependency]
    file_labels:    Dict[str, str]
    node_templates: List[Node]


def prepare_dataset(df: pd.DataFrame, df_dep: pd.DataFrame) -> PreparedDataset:
    node_templates, file_labels = build_nodes(df, {})
    return PreparedDataset(
        architecture=build_architecture(df),
        dependencies=collapse_dependencies_to_file_level(df_dep, file_labels.keys()),
        file_labels=file_labels,
        node_templates=node_templates,
    )


def instantiate_nodes(prepared: PreparedDataset, clustered_files: Dict[str, str]) -> List[Node]:
    return [
        Node(
            name=t.name,
            classes=[CodeClass(name=cc.name, texts=list(cc.texts), is_inner=cc.is_inner)
                     for cc in t.classes],
            mapping=t.mapping,
            clustering=clustered_files.get(t.name, ""),
        )
        for t in prepared.node_templates
    ]


def generate_file_split(
    df: pd.DataFrame, *, split_ratio: float, min_train_points: int, max_train_points: int
) -> Tuple[List[str], List[str]]:
    file_labels = (
        clean_file_table(df)[["File", "Module"]]
        .drop_duplicates(subset=["File"])
        .reset_index(drop=True)
    )
    file_labels = file_labels[file_labels["Module"] != ""]  # seed candidates need a real label
    files  = file_labels["File"].astype(str).to_numpy()
    labels = file_labels["Module"].astype(str).to_numpy()

    indices = np.arange(len(files))
    np.random.shuffle(indices)

    n_train = min(max(int(np.ceil(split_ratio * len(files))), min_train_points), max_train_points)
    train_set = set(indices[:n_train].tolist())

    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        if not any(i in train_set for i in label_idx):
            train_set.add(int(np.random.choice(label_idx)))

    train_idx = np.array(sorted(train_set), dtype=np.int64)
    test_idx  = np.setdiff1d(np.arange(len(files)), train_idx)
    return files[train_idx].tolist(), files[test_idx].tolist()


# ---------------------------------------------------------------------------
# Iterative mapping
# ---------------------------------------------------------------------------

def _run_nb_mapper_once(
    prepared: PreparedDataset, clustered_files: Dict[str, str], config
) -> Tuple[List[MappingResult], Dict[str, str]]:
    nodes  = instantiate_nodes(prepared, clustered_files)
    mapper = NBMapper(
        architecture=prepared.architecture,
        use_cda=config.use_cda,
        use_node_text=config.use_node_text,
        use_node_name=config.use_node_name,
        use_arch_component_name=config.use_arch_component_name,
        min_word_length=config.min_word_length,
        mapping_threshold=config.mapping_threshold,
        word_count=config.word_count,
        use_stemming=config.use_stemming,
        random_seed=config.random_seed,
    )
    return mapper.map(nodes, prepared.dependencies), prepared.file_labels


def run_iterative_nb_mapping(
    prepared: PreparedDataset, train_files: Sequence[str], config
) -> Tuple[List[MappingResult], Dict[str, str], Dict[str, int]]:
    file_labels     = prepared.file_labels
    clustered_files = {fp: file_labels[fp] for fp in train_files if fp in file_labels}

    rounds_done      = 0
    total_pseudo     = 0
    accepted_results: Dict[str, MappingResult] = {}

    while True:
        results, file_labels = _run_nb_mapper_once(prepared, clustered_files, config)
        newly_mapped = {}
        for r in results:
            if r.predicted_component is None or r.node_name in clustered_files:
                continue
            newly_mapped[r.node_name]       = r.predicted_component
            accepted_results[r.node_name]   = r
        rounds_done += 1

        if not newly_mapped:
            final_results = list(accepted_results.values()) + results
            break

        clustered_files.update(newly_mapped)
        total_pseudo += len(newly_mapped)

    remaining = [r for r in final_results if r.node_name not in set(train_files)]
    return remaining, file_labels, {
        "self_mapping_rounds_completed": rounds_done,
        "pseudo_mapped_count":           total_pseudo,
        "final_training_size":           len(clustered_files),
    }


def evaluate_mapping_results(
    results: Sequence[MappingResult], file_labels: Dict[str, str]
) -> Optional[Dict[str, float]]:
    mapped_true, mapped_pred = [], []
    forced_true, forced_pred = [], []

    for r in results:
        true = file_labels.get(r.node_name)
        if not true:  # no ground truth for this file — can't be scored
            continue
        forced_true.append(true)
        forced_pred.append(max(r.attractions, key=r.attractions.get))
        if r.predicted_component is not None:
            mapped_true.append(true)
            mapped_pred.append(r.predicted_component)

    if not mapped_true:
        return None

    total         = len(forced_true)
    mapped_labels = sorted(set(mapped_true))
    full_labels   = sorted(set(forced_true))

    def _s(y_t, y_p, labels):
        kw = dict(labels=labels, zero_division=0)
        return {
            "f1_micro":        f1_score(y_t, y_p, average="micro",  **kw),
            "f1_macro":        f1_score(y_t, y_p, average="macro",  **kw),
            "precision_micro": precision_score(y_t, y_p, average="micro",  **kw),
            "precision_macro": precision_score(y_t, y_p, average="macro",  **kw),
            "recall_micro":    recall_score(y_t, y_p, average="micro",  **kw),
            "recall_macro":    recall_score(y_t, y_p, average="macro",  **kw),
        }

    m = _s(mapped_true, mapped_pred, mapped_labels)
    f = _s(forced_true, forced_pred, full_labels)
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


def predict_mapping(prepared: PreparedDataset, train_files: Sequence[str], config) -> Dict[str, str]:
    """Run the iterative mapping and return {file: predicted_module} for every
    non-seed file — no evaluation, no ground truth required for any of them.
    Every file gets a mapping: confidently-mapped files use
    predicted_component, anything left over falls back to the highest-
    attraction module (same forced-choice convention as the full_ metrics)."""
    results, _, _ = run_iterative_nb_mapping(prepared, train_files, config)
    return {
        r.node_name: r.predicted_component or max(r.attractions, key=r.attractions.get)
        for r in results
    }


# ---------------------------------------------------------------------------
# Worker (top-level for Windows multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker_run(args: tuple) -> Optional[dict]:
    prepared, train_files, config, run_id, data_name, dataset_name = args
    t0 = time.time()
    results, file_labels, stats = run_iterative_nb_mapping(prepared, train_files, config)
    elapsed = time.time() - t0
    metrics = evaluate_mapping_results(results, file_labels)
    if metrics is None:
        return None
    return {
        **metrics,
        "dataset":         dataset_name,
        "data_name":       data_name,
        "run_id":          run_id,
        "train_size":      len(train_files),
        "elapsed_seconds": round(elapsed, 3),
        **stats,
        "mapping_threshold": config.mapping_threshold,
    }


