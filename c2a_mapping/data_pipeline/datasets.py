"""
Dataset discovery and member-level -> file-level aggregation.

Discovers every {stem}.csv / {stem}_deps.csv pair from DATA_DIR so any
system dropped into data/ is automatically loadable.
"""
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

DISPLAY_NAMES = {
    "argouml": "A.UML",
    "commons": "C.Img", 
    "jabref": "Jabref",
    "lucene": "Lucene", 
    "sweetHome": "SH-3D", 
    "teammates": "T.Mates",
    "bash": "Bash", 
    "hdc": "HDC", 
    "hdf": "HDF", 
    "chromium": "Chromium",
}

SPLIT_RATIOS = {"argouml": 0.05, "commons": 0.05, "jabref": 0.05, "lucene": 0.05,
    "sweetHome": 0.05, "teammates": 0.05,
}
DEFAULT_SPLIT_RATIO = 0.05


def discover_datasets(data_dir=None):
    """Return sorted list of stems that have a matching {stem}.csv + {stem}_deps.csv pair."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    stems = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_deps.csv"):
            stem = f[: -len("_deps.csv")]
            if (data_dir / f"{stem}.csv").exists():
                stems.append(stem)
    return stems


def display_name(stem):
    return DISPLAY_NAMES.get(stem, stem)


def split_ratio(stem):
    return SPLIT_RATIOS.get(stem, DEFAULT_SPLIT_RATIO)


def load_file_level(stem, data_dir=None):
    """Aggregate member-level CSVs for `stem` to file-level entity/dep tables.

    Returns (file_df, file_dep) where file_df has one row per File and
    file_dep has file-level dependency edges with Source_ID/Target_ID columns.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(data_dir / f"{stem}.csv")
    dep = pd.read_csv(data_dir / f"{stem}_deps.csv")

    df["Member_Name"] = df["Member_Name"].astype(str)
    file_df = (
        df.groupby("File", as_index=False)
        .agg(
            Entity=("Entity", "first"),
            Module=("Module", lambda s: (lambda vals: sorted(vals)[0] if vals else None)(
                set(str(m) for m in s if pd.notna(m) and str(m).strip())
            )),
            Code=("Member_Name", lambda s: " ".join(sorted(set(str(x) for x in s if pd.notna(x) and str(x).strip())))),
        )
    )
    file_df["File_ID"] = range(len(file_df))
    file_id_map = dict(zip(file_df["File"], file_df["File_ID"]))

    dep = dep.dropna(subset=["Source_File", "Target_File", "Dependency_Type"])
    file_dep = (
        dep.groupby(["Source_File", "Target_File", "Dependency_Type"], as_index=False)
        .agg(Dependency_Count=("Dependency_Count", "sum"))
    )
    file_dep = file_dep[file_dep["Source_File"] != file_dep["Target_File"]]
    file_dep["Source_ID"] = file_dep["Source_File"].map(file_id_map)
    file_dep["Target_ID"] = file_dep["Target_File"].map(file_id_map)
    file_dep = file_dep.dropna(subset=["Source_ID", "Target_ID"])
    file_dep["Source_ID"] = file_dep["Source_ID"].astype(int)
    file_dep["Target_ID"] = file_dep["Target_ID"].astype(int)

    return file_df, file_dep
