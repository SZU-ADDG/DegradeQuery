from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import tarfile
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from tqdm.auto import tqdm

PROJECT_VERSION = "0.1.5"
DEFAULT_CONFIG_PATH = "config.yaml"
SOURCE_SNAPSHOT_FILES = [
    "dq.py",
    "dq_data.py",
    "dq_model.py",
    "config.yaml",
    "requirements.txt",
    "README.md",
    "docs/CHANGELOG.md",
    "docs/RUN_NOTES.md",
]
BASE_STANDARD_COLUMNS = [
    "row_id",
    "dataset",
    "source",
    "split",
    "is_labeled",
    "label",
    "smiles",
    "target_id",
    "target_name",
    "target_sequence",
    "recruiter_id",
    "recruiter_name",
    "recruiter_sequence",
    "cell_line",
    "dc50_nm",
    "dmax_percent",
    "degradation_percent",
    "assay_text",
    "task_id",
]
PROTAC_EXTRA_COLUMNS = [
    "benchmark_ready",
    "data_protocol",
    "split_policy",
    "label_source",
    "target_id_source",
    "recruiter_id_source",
    "target_sequence_source",
    "recruiter_sequence_source",
    "protac_path",
    "target_pocket_path",
    "target_ligand_path",
    "recruiter_pocket_path",
    "recruiter_ligand_path",
]
STANDARD_COLUMNS = BASE_STANDARD_COLUMNS + PROTAC_EXTRA_COLUMNS
BOOLEAN_COLUMNS = {"is_labeled", "benchmark_ready"}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
CLASSIC_UNIPROT = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])$"
)
EXTENDED_UNIPROT = re.compile(r"^A0A[A-Z0-9]{7}$")
PROTAC_NUMERIC_ID = re.compile(r"([0-9]+)_protac$")
LABEL_TRUE = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "active",
    "positive",
    "high",
    "degrader",
    "degradable",
}
LABEL_FALSE = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "inactive",
    "negative",
    "low",
    "nondegrader",
    "non-degrader",
    "undegradable",
}
RUNTIME_EXCEL_SUFFIXES = {".xlsx", ".xls"}
DATASET_FILE_SUFFIXES = {".csv", ".json", ".xlsx", ".xls", ".pkl", ".pt", ".npy", ".mol2", ".pdb"}
PDB_FULL_PATH_COLUMNS = [
    "target_structure_path",
    "target_full_path",
    "target_protein_path",
    "tar_path",
    "e3_ligase_path",
    "recruiter_structure_path",
    "recruiter_full_path",
]
PDB_TARGET_FALLBACK_COLUMNS = ["target_structure_path", "target_full_path", "target_protein_path", "tar_path", "target_pocket_path"]
PDB_RECRUITER_FALLBACK_COLUMNS = [
    "recruiter_structure_path",
    "recruiter_full_path",
    "e3_ligase_path",
    "recruiter_pocket_path",
]
THREE_TO_ONE_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
}

warnings.filterwarnings("ignore", category=FutureWarning)
UNIPROT_RUNTIME = {"disabled": False, "disable_reason": ""}

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDKIT_AVAILABLE = True
RDKIT_IMPORT_ERROR: Optional[Exception] = None
RDLogger.DisableLog("rdApp.*")

SMILES_ALIASES = [
    "smiles",
    "smile",
    "canonical_smiles",
    "protac_smiles",
    "compound_smiles",
    "query_smiles",
]
LABEL_ALIASES = [
    "label",
    "labels",
    "class",
    "activity",
    "active",
    "status",
    "degradation_label",
    "degradation_status",
    "binary_label",
    "y",
]
SPLIT_ALIASES = ["split", "set", "subset", "partition", "group"]
TARGET_ID_ALIASES = [
    "target_id",
    "target_uniprot",
    "target_uniprot_id",
    "target_accession",
    "target_protein_id",
]
TARGET_NAME_ALIASES = [
    "target_name",
    "target",
    "protein",
    "protein_name",
    "target_protein",
]
TARGET_SEQUENCE_ALIASES = [
    "target_sequence",
    "target_seq",
    "protein_sequence",
    "sequence",
]
RECRUITER_ID_ALIASES = [
    "recruiter_id",
    "e3_id",
    "e3_uniprot",
    "e3_uniprot_id",
    "ligase_id",
]
RECRUITER_NAME_ALIASES = [
    "recruiter_name",
    "recruiter",
    "e3",
    "e3_ligase",
    "ligase",
    "recruiting_protein",
]
RECRUITER_SEQUENCE_ALIASES = [
    "recruiter_sequence",
    "recruiter_seq",
    "e3_sequence",
    "ligase_sequence",
]
CELL_LINE_ALIASES = ["cell_line", "cellline", "cell", "cell_type"]
DC50_ALIASES = ["dc50_nm", "dc50", "dc_50", "dc50(nm)", "dc50_nM"]
DMAX_ALIASES = ["dmax_percent", "dmax", "d_max", "dmax(%)", "dmax_percent(%)"]
DEGRADATION_PERCENT_ALIASES = [
    "degradation_percent",
    "degradation",
    "degradation_capacity",
    "deg_percent",
    "degradation_%",
]
ASSAY_TEXT_ALIASES = ["assay_text", "assay", "assay_description", "comment", "notes"]
ROW_ID_ALIASES = ["row_id", "id", "compound_id", "entry_id", "record_id"]
TASK_ID_ALIASES = ["task_id"]
ANALYSIS_SPLIT_COLUMNS = [
    "row_id",
    "split",
    "analysis_scheme",
    "seed",
    "is_labeled",
    "label",
    "target_id",
    "recruiter_id",
    "canonical_smiles",
    "scaffold",
    "pretrain_use",
    "supervised_use",
    "label_visible",
    "fewshot_role",
    "meta_target_side",
    "label_fraction",
    "test_frac",
    "shots_per_class",
]


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def canonical_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def sequence_kmer_set(sequence: str, k: int = 3) -> set[str]:
    seq = "".join(ch for ch in safe_text(sequence).upper() if ch.isalpha())
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[idx : idx + k] for idx in range(len(seq) - k + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / max(len(left | right), 1))


def target_sequence_cluster_map(frame: pd.DataFrame, threshold: float = 0.70) -> Dict[str, str]:
    work = frame[["target_id", "target_sequence"]].copy()
    work["target_id"] = work["target_id"].fillna("").astype(str)
    work["target_sequence"] = work["target_sequence"].fillna("").astype(str)
    work = work[work["target_id"] != ""].drop_duplicates("target_id").reset_index(drop=True)
    if work.empty:
        return {}
    target_ids = work["target_id"].tolist()
    kmers = [sequence_kmer_set(sequence) for sequence in work["target_sequence"].tolist()]
    neighbors: Dict[int, List[int]] = defaultdict(list)
    for left in range(len(target_ids)):
        for right in range(left + 1, len(target_ids)):
            if jaccard_similarity(kmers[left], kmers[right]) >= float(threshold):
                neighbors[left].append(right)
                neighbors[right].append(left)
    cluster_map: Dict[str, str] = {}
    seen: set[int] = set()
    cluster_idx = 0
    for start_idx, target_id in enumerate(target_ids):
        if start_idx in seen:
            continue
        cluster_idx += 1
        label = f"sequence_cluster_family_{cluster_idx:03d}"
        queue = [start_idx]
        seen.add(start_idx)
        while queue:
            current = queue.pop()
            cluster_map[target_ids[current]] = label
            for nxt in sorted(neighbors.get(current, []), key=lambda item: target_ids[item]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if target_id not in cluster_map:
            cluster_map[target_id] = label
    return cluster_map


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        if pd.isna(value):
            return None
        return float(value)
    text = safe_text(value)
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_text(value).lower()
    return text in {"1", "true", "t", "yes", "y"}


def is_probable_uniprot(accession: str) -> bool:
    text = safe_text(accession).upper()
    return bool(CLASSIC_UNIPROT.match(text) or EXTENDED_UNIPROT.match(text))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return bytes_sha256(blob)


def require_rdkit() -> None:
    if not RDKIT_AVAILABLE:
        raise RuntimeError(
            "RDKit is required for molecule parsing but is unavailable in this environment: "
            f"{RDKIT_IMPORT_ERROR}"
        )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def runtime_root(config: Dict[str, Any]) -> Path:
    configured = Path(config["project"]["root"])
    if configured.exists():
        return configured
    return Path(__file__).resolve().parent


def resolve_path(config: Dict[str, Any], raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return runtime_root(config) / path


def ensure_project_dirs(config: Dict[str, Any]) -> None:
    resolve_path(config, config["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_path(config, config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_path(config, config["paths"]["docs_dir"]).mkdir(parents=True, exist_ok=True)


def notes_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, "docs/RUN_NOTES.md")


def changelog_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, "docs/CHANGELOG.md")


def output_dir(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["output_dir"])


def data_dir(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["data_dir"])


def feature_cache_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["data"]["feature_cache"])


def sequence_cache_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["data"]["sequence_cache"])


def protein_emb_cache_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["data"]["protein_emb_cache"])


def esm2_embedding_cache_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["data"].get("esm2_650m_embeddings", "data/esm2_650m_embeddings.pt"))


def official_name_json_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["data"]["protac8k_official_name"])


def audit_report_path(config: Dict[str, Any]) -> Path:
    return output_dir(config) / "protac8k_audit.json"


def split_report_path(config: Dict[str, Any]) -> Path:
    return output_dir(config) / "protac8k_split_report.json"


def leakage_report_path(config: Dict[str, Any]) -> Path:
    return output_dir(config) / "protac8k_leakage_report.json"


def sequence_report_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_sequence_report.json"


def baseline_report_json_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_baseline_report.json"


def baseline_report_md_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_baseline_report.md"


def benchmark_report_json_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_benchmark_report.json"


def benchmark_report_md_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_benchmark_report.md"


def compare_runs_report_json_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_compare_runs_report.json"


def compare_runs_report_md_path(config: Dict[str, Any], dataset: str) -> Path:
    return output_dir(config) / f"{dataset}_compare_runs_report.md"


def named_json_output_path(config: Dict[str, Any], stem: str) -> Path:
    return output_dir(config) / f"{stem}.json"


def named_md_output_path(config: Dict[str, Any], stem: str) -> Path:
    return output_dir(config) / f"{stem}.md"


def analysis_split_csv_path(config: Dict[str, Any], dataset: str, scheme: str, seed: int) -> Path:
    return output_dir(config) / f"{timestamp_now()}_{dataset}_{scheme}_seed{seed}_split.csv"


def analysis_split_report_path(config: Dict[str, Any], dataset: str, scheme: str, seed: int) -> Path:
    return output_dir(config) / f"{timestamp_now()}_{dataset}_{scheme}_seed{seed}_split_report.json"


def esm2_embedding_report_json_path(config: Dict[str, Any]) -> Path:
    return named_json_output_path(config, "esm2_650m_embedding_report")


def esm2_embedding_report_md_path(config: Dict[str, Any]) -> Path:
    return named_md_output_path(config, "esm2_650m_embedding_report")


def esm2_failure_report_path(config: Dict[str, Any]) -> Path:
    return named_md_output_path(config, "esm2_650m_failure_report")


def fixed05_error_profile_path(config: Dict[str, Any]) -> Path:
    return named_md_output_path(config, "fixed05_error_profile_best_model")


def normalized_dataset_path(config: Dict[str, Any], dataset: str) -> Path:
    if dataset == "protac8k":
        return resolve_path(config, config["data"]["normalized_protac8k"])
    if dataset == "molglue":
        return resolve_path(config, config["data"]["normalized_molglue"])
    if dataset == "mgtbind":
        return resolve_path(config, config["data"]["normalized_mgtbind"])
    return resolve_path(config, f"data/{dataset}_normalized.csv")


def protac8k_root(config: Dict[str, Any]) -> Path:
    return resolve_path(config, config["data"]["protac8k_dir"])


def append_note(config: Dict[str, Any], title: str, lines: Sequence[str]) -> None:
    ensure_project_dirs(config)
    path = notes_path(config)
    block = [f"\n## {title}", ""]
    block.extend([line.rstrip() for line in lines if line is not None])
    block.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


def load_feature_cache(config: Dict[str, Any]) -> Dict[str, Any]:
    path = feature_cache_path(config)
    if not path.exists():
        return {"datasets": {}}
    return torch.load(path, map_location="cpu", weights_only=False)


def save_feature_cache(config: Dict[str, Any], cache: Dict[str, Any]) -> None:
    ensure_project_dirs(config)
    torch.save(cache, feature_cache_path(config))


def append_manifest(config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    ensure_project_dirs(config)
    manifest_path = output_dir(config) / "run_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def compute_source_tree_sha256(config: Dict[str, Any]) -> str:
    root = runtime_root(config)
    digest = hashlib.sha256()
    for rel_path in SOURCE_SNAPSHOT_FILES:
        path = root / rel_path
        if not path.exists():
            continue
        digest.update(rel_path.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_source_bundle_sha256(config: Dict[str, Any]) -> str:
    bundle_path = output_dir(config) / "source_last_good.tar.gz"
    if bundle_path.exists():
        return file_sha256(bundle_path)
    return compute_source_tree_sha256(config)


def compute_source_hash(config: Dict[str, Any]) -> str:
    return compute_source_tree_sha256(config)


def compute_dataset_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.copy()
    ordered = ordered[sorted(ordered.columns)]
    blob = ordered.to_csv(index=False).encode("utf-8")
    return bytes_sha256(blob)


def resolve_device(config: Dict[str, Any]) -> torch.device:
    requested = safe_text(config["project"].get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretty_print_stats(stats: Dict[str, Any]) -> None:
    for key in [
        "n_total",
        "n_labeled",
        "n_unlabeled",
        "n_positive",
        "n_negative",
        "n_targets",
        "n_recruiters",
        "n_cell_lines",
        "missing_sequence_count",
        "split_policy",
    ]:
        if key in stats:
            print(f"{key}: {stats[key]}")


def list_dataset_files(dataset_root: Path) -> List[Path]:
    return sorted([path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in DATASET_FILE_SUFFIXES])


def read_dataframe_from_path(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return pd.read_json(path, lines=True)
        if isinstance(payload, dict):
            if all(isinstance(v, dict) for v in payload.values()):
                return pd.DataFrame.from_dict(payload, orient="index").reset_index(drop=True)
            return pd.json_normalize(payload)
        return pd.DataFrame(payload)
    if ext in RUNTIME_EXCEL_SUFFIXES:
        return pd.read_excel(path)
    if ext == ".pkl":
        payload = pd.read_pickle(path)
        if isinstance(payload, pd.DataFrame):
            return payload
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame(payload)
        raise ValueError(f"Unsupported pickle content for DataFrame conversion: {type(payload)}")
    if ext == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, pd.DataFrame):
            return payload
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame(payload)
        raise ValueError(f"Unsupported pt content for DataFrame conversion: {type(payload)}")
    raise ValueError(f"Unsupported file type: {path}")


def find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    alias_set = {canonical_name(alias) for alias in aliases}
    for column in frame.columns:
        if canonical_name(column) in alias_set:
            return column
    return None


def normalize_label_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        if value in {0, 1}:
            return int(value)
        if value == -1:
            return None
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return None
        if abs(value - 0) < 1e-8:
            return 0
        if abs(value - 1) < 1e-8:
            return 1
        if abs(value + 1) < 1e-8:
            return None
    text = safe_text(value).lower()
    if not text:
        return None
    if text in LABEL_TRUE:
        return 1
    if text in LABEL_FALSE:
        return 0
    numeric = safe_float(text)
    if numeric is None:
        return None
    if abs(numeric - 0) < 1e-8:
        return 0
    if abs(numeric - 1) < 1e-8:
        return 1
    return None


def standardize_split_value(value: Any) -> str:
    text = safe_text(value).lower()
    if not text:
        return ""
    if text in {"train", "training"}:
        return "train"
    if text in {"val", "valid", "validation", "dev"}:
        return "val"
    if text in {"test", "testing", "holdout"}:
        return "test"
    if text in {"unlabeled", "unlabelled"}:
        return "unlabeled"
    return text


def canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def bemis_murcko_scaffold(smiles: str) -> str:
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return ""
    require_rdkit()
    try:
        return safe_text(MurckoScaffold.MurckoScaffoldSmiles(smiles=canonical))
    except Exception:
        return ""


def mol_from_smiles(smiles: str):
    if not smiles:
        return None
    require_rdkit()
    return Chem.MolFromSmiles(smiles)


def mol_from_mol2(path: Path):
    require_rdkit()
    if not path.exists():
        return None
    try:
        return Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    except Exception:
        return None


def count_atoms_from_mol2(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("@<TRIPOS>ATOM") + 1
        end = lines.index("@<TRIPOS>BOND")
    except ValueError:
        return 0
    return sum(1 for line in lines[start:end] if line.strip())


def extract_sample_id_from_protac_file(path: Path) -> int:
    match = PROTAC_NUMERIC_ID.search(path.stem)
    if match:
        return int(match.group(1))
    return safe_int(path.stem, 0)


def sort_protac_mol2_files(paths: Sequence[Path]) -> List[Path]:
    return sorted(paths, key=lambda item: (extract_sample_id_from_protac_file(item), item.name))


def aux_vector_key(vector: np.ndarray) -> str:
    return bytes_sha256(np.asarray(vector, dtype=np.float32).tobytes())


def stable_cluster_ids(vectors: np.ndarray, prefix: str) -> List[str]:
    mapping: Dict[str, str] = {}
    labels: List[str] = []
    for vector in vectors:
        key = aux_vector_key(vector)
        if key not in mapping:
            mapping[key] = f"{prefix}_{len(mapping) + 1:03d}"
        labels.append(mapping[key])
    return labels


def load_json_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fasta_to_sequence(text: str) -> str:
    return "".join(line.strip() for line in text.splitlines() if line and not line.startswith(">"))


def fetch_uniprot_sequence(uniprot_id: str, timeout: int = 10, retries: int = 3, sleep_seconds: float = 1.0) -> Optional[str]:
    if UNIPROT_RUNTIME["disabled"]:
        return None
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                sequence = fasta_to_sequence(response.text)
                return sequence or None
            if response.status_code == 404:
                return None
        except requests.RequestException as exc:
            message = safe_text(exc).lower()
            if "connection refused" in message or "failed to establish a new connection" in message:
                UNIPROT_RUNTIME["disabled"] = True
                UNIPROT_RUNTIME["disable_reason"] = safe_text(exc)
                return None
        if attempt + 1 < retries:
            time.sleep(sleep_seconds)
    return None


def pdb_sequence_from_atom_lines(path: Path) -> str:
    if not path.exists() or path.suffix.lower() != ".pdb":
        return ""
    sequence_by_chain: Dict[str, List[Tuple[Tuple[str, str, str], str, bool]]] = defaultdict(list)
    seen: set[Tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_name = safe_text(line[17:20]).upper()
            aa = THREE_TO_ONE_AA.get(residue_name, "")
            if not aa:
                continue
            atom_name = safe_text(line[12:16]).upper()
            chain_id = safe_text(line[21:22]) or "_"
            residue_number = safe_text(line[22:26])
            insertion_code = safe_text(line[26:27])
            residue_key = (chain_id, residue_number, insertion_code)
            seen_key = (chain_id, residue_number, insertion_code, aa)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            sequence_by_chain[chain_id].append((residue_key, aa, atom_name == "CA"))
    best_sequence = ""
    for chain_id, entries in sequence_by_chain.items():
        ordered: Dict[Tuple[str, str, str], Tuple[str, bool]] = {}
        for residue_key, aa, is_ca in entries:
            if residue_key not in ordered:
                ordered[residue_key] = (aa, is_ca)
            elif is_ca and not ordered[residue_key][1]:
                ordered[residue_key] = (aa, True)
        sequence = "".join(aa for _, (aa, _) in sorted(ordered.items(), key=lambda item: (item[0][1], item[0][2])))
        if len(sequence) > len(best_sequence):
            best_sequence = sequence
    return best_sequence


def choose_first_existing_pdb(row: pd.Series, columns: Sequence[str]) -> str:
    for column in columns:
        raw = safe_text(row.get(column, ""))
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path.suffix.lower() == ".pdb":
            return str(path)
    return ""


def classify_pdb_sequence_source(path: Path, sequence: str) -> str:
    lowered = "/".join(part.lower() for part in path.parts)
    if "data/protac8k/" in lowered or lowered.endswith(".pdb"):
        return "pdb_full"
    if any(token in lowered for token in ["pocket", "fragment", "site"]) and len(sequence) < 100:
        return "pdb_pocket"
    return "pdb_full"


def build_sequence_report(frame: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
    target_present = frame["target_sequence"].fillna("").astype(str).str.len() > 0
    recruiter_present = frame["recruiter_sequence"].fillna("").astype(str).str.len() > 0
    target_ids = frame["target_id"].fillna("").astype(str)
    recruiter_ids = frame["recruiter_id"].fillna("").astype(str)
    unique_target_ids = sorted({safe_text(value) for value in target_ids.tolist() if safe_text(value)})
    unique_recruiter_ids = sorted({safe_text(value) for value in recruiter_ids.tolist() if safe_text(value)})
    target_cov_unique = 0.0
    recruiter_cov_unique = 0.0
    if unique_target_ids:
        covered = {
            safe_text(row["target_id"])
            for _, row in frame.loc[target_present, ["target_id"]].drop_duplicates().iterrows()
            if safe_text(row["target_id"])
        }
        target_cov_unique = len(covered) / len(unique_target_ids)
    if unique_recruiter_ids:
        covered = {
            safe_text(row["recruiter_id"])
            for _, row in frame.loc[recruiter_present, ["recruiter_id"]].drop_duplicates().iterrows()
            if safe_text(row["recruiter_id"])
        }
        recruiter_cov_unique = len(covered) / len(unique_recruiter_ids)
    missing_examples = []
    missing_mask = ~(target_present & recruiter_present)
    for _, row in frame.loc[missing_mask, ["row_id", "target_id", "recruiter_id", "target_sequence_source", "recruiter_sequence_source"]].head(10).iterrows():
        missing_examples.append(
            {
                "row_id": safe_text(row["row_id"]),
                "target_id": safe_text(row["target_id"]),
                "recruiter_id": safe_text(row["recruiter_id"]),
                "target_sequence_source": safe_text(row["target_sequence_source"]),
                "recruiter_sequence_source": safe_text(row["recruiter_sequence_source"]),
            }
        )
    return {
        "n_rows": int(len(frame)),
        "unique_target_ids": int(len(unique_target_ids)),
        "unique_recruiter_ids": int(len(unique_recruiter_ids)),
        "target_sequence_coverage_row_level": float(target_present.mean()) if len(frame) else 0.0,
        "recruiter_sequence_coverage_row_level": float(recruiter_present.mean()) if len(frame) else 0.0,
        "target_sequence_coverage_unique_level": float(target_cov_unique),
        "recruiter_sequence_coverage_unique_level": float(recruiter_cov_unique),
        "n_uniprot_success": int(stats.get("n_uniprot_success", 0)),
        "n_pdb_success": int(stats.get("n_pdb_success", 0)),
        "n_missing": int(missing_mask.sum()),
        "examples_missing": missing_examples,
        "target_sequence_source_counts": Counter(frame["target_sequence_source"].fillna("").astype(str).tolist()),
        "recruiter_sequence_source_counts": Counter(frame["recruiter_sequence_source"].fillna("").astype(str).tolist()),
    }


def ensure_sequences(
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_path = sequence_cache_path(config)
    cache = load_json_cache(cache_path)
    stats: Dict[str, Any] = {
        "cache_hits": 0,
        "existing_rows": 0,
        "n_uniprot_success": 0,
        "n_pdb_success": 0,
        "n_missing": 0,
        "n_uniprot_failed": 0,
    }
    resolved_id_cache: Dict[str, Tuple[str, str]] = {}
    if not isinstance(cache, dict):
        cache = {}

    def resolve(sequence: str, protein_id: str, current_source: str, row: pd.Series, path_columns: Sequence[str]) -> Tuple[str, str]:
        sequence = safe_text(sequence)
        current_source = safe_text(current_source)
        protein_id = safe_text(protein_id).upper()
        if sequence:
            stats["existing_rows"] += 1
            if current_source.startswith("pdb_"):
                pdb_path = choose_first_existing_pdb(row, path_columns)
                if pdb_path:
                    return sequence, classify_pdb_sequence_source(Path(pdb_path), sequence)
            return sequence, current_source or "manual"
        if protein_id and protein_id in resolved_id_cache:
            cached_sequence, cached_source = resolved_id_cache[protein_id]
            return cached_sequence, cached_source
        if protein_id and is_probable_uniprot(protein_id):
            if protein_id in cache:
                stats["cache_hits"] += 1
                cached_value = safe_text(cache.get(protein_id, ""))
                if cached_value:
                    resolved_id_cache[protein_id] = (cached_value, "uniprot")
                    return cached_value, "uniprot"
            else:
                fetched = fetch_uniprot_sequence(protein_id, timeout=10, retries=3, sleep_seconds=1.0)
                cache[protein_id] = fetched or ""
                if fetched:
                    stats["n_uniprot_success"] += 1
                    resolved_id_cache[protein_id] = (fetched, "uniprot")
                    return fetched, "uniprot"
                stats["n_uniprot_failed"] += 1
        pdb_path = choose_first_existing_pdb(row, path_columns)
        if pdb_path:
            sequence = pdb_sequence_from_atom_lines(Path(pdb_path))
            if sequence:
                pdb_source = classify_pdb_sequence_source(Path(pdb_path), sequence)
                stats["n_pdb_success"] += 1
                if protein_id:
                    resolved_id_cache[protein_id] = (sequence, pdb_source)
                return sequence, pdb_source
        stats["n_missing"] += 1
        if protein_id:
            resolved_id_cache[protein_id] = ("", "missing")
        return "", "missing"

    for seq_col, id_col, source_col, path_columns in [
        ("target_sequence", "target_id", "target_sequence_source", PDB_TARGET_FALLBACK_COLUMNS),
        ("recruiter_sequence", "recruiter_id", "recruiter_sequence_source", PDB_RECRUITER_FALLBACK_COLUMNS),
    ]:
        sequences: List[str] = []
        sources: List[str] = []
        for _, row in frame.iterrows():
            resolved_sequence, resolved_source = resolve(row.get(seq_col, ""), row.get(id_col, ""), row.get(source_col, ""), row, path_columns)
            sequences.append(resolved_sequence)
            sources.append(resolved_source)
        frame[seq_col] = sequences
        frame[source_col] = sources

    save_json_cache(cache_path, cache)
    return frame, stats


def fetchseq_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    dataset = args.dataset
    frame = load_normalized_dataset(config, dataset)
    frame, stats = ensure_sequences(frame, config)
    write_normalized_dataset(frame, normalized_dataset_path(config, dataset))
    report = build_sequence_report(frame, stats)
    write_json(sequence_report_path(config, dataset), report)
    append_note(
        config,
        f"Sequence Recovery {timestamp_now()}",
        [
            f"- dataset: {dataset}",
            f"- unique_target_ids: {report['unique_target_ids']}",
            f"- unique_recruiter_ids: {report['unique_recruiter_ids']}",
            f"- target_row_coverage: {report['target_sequence_coverage_row_level']:.4f}",
            f"- recruiter_row_coverage: {report['recruiter_sequence_coverage_row_level']:.4f}",
            f"- n_uniprot_success: {report['n_uniprot_success']}",
            f"- n_pdb_success: {report['n_pdb_success']}",
            f"- n_missing: {report['n_missing']}",
        ],
    )
    print(f"sequence_report: {sequence_report_path(config, dataset)}")


def empty_standard_row(dataset: str, source: str, row_id: str) -> Dict[str, Any]:
    row = {column: "" for column in STANDARD_COLUMNS}
    row["row_id"] = row_id
    row["dataset"] = dataset
    row["source"] = source
    row["is_labeled"] = False
    row["benchmark_ready"] = False
    return row


def infer_label_and_rule(
    dataset: str,
    raw_label: Any,
    dc50_nm: Optional[float],
    dmax_percent: Optional[float],
    degradation_percent: Optional[float],
    config: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    parsed = normalize_label_value(raw_label)
    if parsed is not None:
        return parsed, "original_dataset_label"
    if dataset == "protac8k":
        if dc50_nm is not None and dmax_percent is not None:
            low = (
                dc50_nm >= float(config["labeling"]["protac_low_dc50_nm"])
                and dmax_percent < float(config["labeling"]["protac_low_dmax_percent"])
            )
            return (0 if low else 1), "recomputed_from_dc50_dmax"
        return None, "missing_label"
    if dc50_nm is not None and dmax_percent is not None:
        low = (
            dc50_nm >= float(config["labeling"]["protac_low_dc50_nm"])
            and dmax_percent < float(config["labeling"]["protac_low_dmax_percent"])
        )
        return (0 if low else 1), "dc50_dmax_threshold"
    if degradation_percent is not None:
        threshold = float(config["labeling"]["degradation_percent_positive_threshold"])
        return (1 if degradation_percent >= threshold else 0), "degradation_percent_threshold"
    return None, "missing_label"


def summarize_standard_frame(frame: pd.DataFrame, split_policy: str) -> Dict[str, Any]:
    labeled = frame[frame["is_labeled"] == True]
    label_series = labeled["label"].astype(int) if not labeled.empty else pd.Series(dtype=int)
    missing_sequence = (
        ((frame["target_sequence"].fillna("") == "") | (frame["recruiter_sequence"].fillna("") == "")).sum()
    )
    return {
        "n_total": int(len(frame)),
        "n_labeled": int(frame["is_labeled"].sum()),
        "n_unlabeled": int((~frame["is_labeled"]).sum()),
        "n_positive": int((label_series == 1).sum()) if not label_series.empty else 0,
        "n_negative": int((label_series == 0).sum()) if not label_series.empty else 0,
        "n_targets": int(frame["task_id"].replace("", np.nan).nunique(dropna=True)),
        "n_recruiters": int(
            frame["recruiter_id"]
            .replace("", np.nan)
            .fillna(frame["recruiter_name"].replace("", np.nan))
            .nunique(dropna=True)
        ),
        "n_cell_lines": int(frame["cell_line"].replace("", np.nan).nunique(dropna=True)),
        "missing_sequence_count": int(missing_sequence),
        "split_policy": split_policy,
    }


def assign_random_split(
    frame: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Tuple[pd.DataFrame, str]:
    work = frame.copy()
    work["split"] = work["split"].fillna("").astype(str)
    labeled_indices = work.index[work["is_labeled"] == True].tolist()
    unlabeled_indices = work.index[work["is_labeled"] == False].tolist()
    rng = random.Random(seed)
    train_ratio, val_ratio, test_ratio = ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError("train/val/test ratios must sum to 1.0")
    label_groups: Dict[int, List[int]] = defaultdict(list)
    for idx in labeled_indices:
        label_groups[int(work.loc[idx, "label"])].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    for group_indices in label_groups.values():
        rng.shuffle(group_indices)
        total = len(group_indices)
        test_count = int(round(total * test_ratio))
        val_count = int(round(total * val_ratio))
        test_indices.extend(group_indices[:test_count])
        val_indices.extend(group_indices[test_count : test_count + val_count])
        train_indices.extend(group_indices[test_count + val_count :])

    work.loc[:, "split"] = ""
    work.loc[train_indices, "split"] = "train"
    work.loc[val_indices, "split"] = "val"
    work.loc[test_indices, "split"] = "test"
    work.loc[unlabeled_indices, "split"] = "unlabeled"
    return work, f"random_split_seed{seed}"


def make_degrademaster_code_split(
    frame: pd.DataFrame,
    train_rate: float,
    seed: int,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    work = frame.copy().reset_index(drop=True)
    total_indices = list(range(len(work)))
    random.seed(seed)
    random.shuffle(total_indices)
    n_labeled = int(work["is_labeled"].sum())
    test_size = int(n_labeled * (1.0 - train_rate))
    test_count = 0
    splits = ["train"] * len(work)
    for idx in total_indices:
        if bool(work.at[idx, "is_labeled"]) and test_count < test_size:
            splits[idx] = "test"
            test_count += 1
        else:
            splits[idx] = "train"
    work["split"] = splits
    labeled = work[work["is_labeled"] == True]
    train_labeled = labeled[labeled["split"] == "train"]
    test_labeled = labeled[labeled["split"] == "test"]
    report = {
        "protocol": f"degrademaster_code_split_seed{seed}",
        "seed": int(seed),
        "train_rate": float(train_rate),
        "n_total": int(len(work)),
        "n_labeled": int(n_labeled),
        "n_unlabeled": int((~work["is_labeled"]).sum()),
        "n_train": int((work["split"] == "train").sum()),
        "n_test": int((work["split"] == "test").sum()),
        "n_train_labeled": int(len(train_labeled)),
        "n_train_unlabeled": int(((work["split"] == "train") & (~work["is_labeled"])).sum()),
        "n_test_labeled": int(len(test_labeled)),
        "n_test_unlabeled": int(((work["split"] == "test") & (~work["is_labeled"])).sum()),
        "train_pos": int((train_labeled["label"].astype(int) == 1).sum()) if not train_labeled.empty else 0,
        "train_neg": int((train_labeled["label"].astype(int) == 0).sum()) if not train_labeled.empty else 0,
        "test_pos": int((test_labeled["label"].astype(int) == 1).sum()) if not test_labeled.empty else 0,
        "test_neg": int((test_labeled["label"].astype(int) == 0).sum()) if not test_labeled.empty else 0,
    }
    return work, report["protocol"], report


def detect_explicit_split(frame: pd.DataFrame) -> Optional[pd.Series]:
    column = find_column(frame, SPLIT_ALIASES)
    if not column:
        return None
    standardized = frame[column].apply(standardize_split_value)
    allowed = {"train", "val", "test", "unlabeled", ""}
    if standardized.isin(allowed).mean() < 0.80:
        return None
    return standardized


def standardize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in work.columns:
            work[column] = False if column in BOOLEAN_COLUMNS else ""
    work["is_labeled"] = work["is_labeled"].apply(as_bool)
    work["benchmark_ready"] = work["benchmark_ready"].apply(as_bool)
    labels = []
    for is_labeled, value in zip(work["is_labeled"].tolist(), work["label"].tolist()):
        labels.append(safe_int(value, -1) if as_bool(is_labeled) else "")
    work["label"] = labels
    for column in STANDARD_COLUMNS:
        if column in BOOLEAN_COLUMNS or column == "label":
            continue
        work[column] = work[column].fillna("").astype(str)
    return work[STANDARD_COLUMNS]


def standardize_analysis_split_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ANALYSIS_SPLIT_COLUMNS:
        if column not in work.columns:
            if column in {"pretrain_use", "supervised_use", "label_visible"}:
                work[column] = False
            elif column in {"seed", "shots_per_class"}:
                work[column] = 0
            elif column in {"label_fraction", "test_frac"}:
                work[column] = ""
            else:
                work[column] = ""
    work["row_id"] = work["row_id"].fillna("").astype(str)
    work["split"] = work["split"].apply(standardize_split_value)
    work["analysis_scheme"] = work["analysis_scheme"].fillna("").astype(str)
    work["seed"] = work["seed"].apply(lambda value: safe_int(value, 0))
    work["is_labeled"] = work["is_labeled"].apply(as_bool)
    labels: List[Any] = []
    for is_labeled, value in zip(work["is_labeled"].tolist(), work["label"].tolist()):
        parsed = normalize_label_value(value)
        labels.append(parsed if parsed is not None and is_labeled else "")
    work["label"] = labels
    work["target_id"] = work["target_id"].fillna("").astype(str)
    work["recruiter_id"] = work["recruiter_id"].fillna("").astype(str)
    work["canonical_smiles"] = work["canonical_smiles"].fillna("").astype(str)
    work["scaffold"] = work["scaffold"].fillna("").astype(str)
    work["pretrain_use"] = work["pretrain_use"].apply(as_bool)
    work["supervised_use"] = work["supervised_use"].apply(as_bool)
    work["label_visible"] = work["label_visible"].apply(as_bool)
    work["fewshot_role"] = work["fewshot_role"].fillna("").astype(str)
    work["meta_target_side"] = work["meta_target_side"].fillna("").astype(str)
    work["label_fraction"] = [safe_float(value) if safe_text(value) else "" for value in work["label_fraction"].tolist()]
    work["test_frac"] = [safe_float(value) if safe_text(value) else "" for value in work["test_frac"].tolist()]
    work["shots_per_class"] = work["shots_per_class"].apply(lambda value: safe_int(value, 0))
    return work[ANALYSIS_SPLIT_COLUMNS]


def default_analysis_overlay(frame: pd.DataFrame, scheme: str = "official", seed: int = 0) -> pd.DataFrame:
    overlay = pd.DataFrame(
        {
            "row_id": frame["row_id"].fillna("").astype(str),
            "split": frame["split"].apply(standardize_split_value),
            "analysis_scheme": scheme,
            "seed": int(seed),
            "is_labeled": frame["is_labeled"].apply(as_bool),
            "label": frame["label"].tolist(),
            "target_id": frame["target_id"].fillna("").astype(str),
            "recruiter_id": frame["recruiter_id"].fillna("").astype(str),
            "canonical_smiles": [canonicalize_smiles(safe_text(value)) for value in frame["smiles"].tolist()],
            "scaffold": [bemis_murcko_scaffold(safe_text(value)) for value in frame["smiles"].tolist()],
        }
    )
    overlay["pretrain_use"] = overlay["split"].eq("train")
    overlay["supervised_use"] = overlay["split"].eq("train") & overlay["is_labeled"]
    overlay["label_visible"] = overlay["is_labeled"]
    overlay["fewshot_role"] = "none"
    overlay["meta_target_side"] = "none"
    overlay["label_fraction"] = ""
    overlay["test_frac"] = ""
    overlay["shots_per_class"] = 0
    return standardize_analysis_split_overlay(overlay)


def load_analysis_split_overlay(config: Dict[str, Any], raw_path: str) -> pd.DataFrame:
    path = resolve_path(config, raw_path)
    frame = read_dataframe_from_path(path)
    return standardize_analysis_split_overlay(frame)


def apply_analysis_split_overlay(frame: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    base = frame.copy()
    base["row_id"] = base["row_id"].fillna("").astype(str)
    merged = base.merge(overlay, on="row_id", how="left", suffixes=("", "__overlay"))
    if merged["split__overlay"].isna().all():
        raise RuntimeError("Split overlay does not match any dataset row_id values.")
    for column in ["split", "is_labeled", "label", "target_id", "recruiter_id"]:
        overlay_column = f"{column}__overlay"
        if overlay_column in merged.columns:
            merged[column] = merged[overlay_column]
    merged["analysis_scheme"] = merged["analysis_scheme"].fillna("").astype(str)
    merged["analysis_seed"] = merged["seed"].apply(lambda value: safe_int(value, 0))
    merged["canonical_smiles"] = merged["canonical_smiles"].fillna("").astype(str)
    merged["scaffold"] = merged["scaffold"].fillna("").astype(str)
    merged["pretrain_use"] = merged["pretrain_use"].apply(as_bool)
    merged["supervised_use"] = merged["supervised_use"].apply(as_bool)
    merged["label_visible"] = merged["label_visible"].apply(as_bool)
    merged["fewshot_role"] = merged["fewshot_role"].fillna("none").astype(str)
    merged["meta_target_side"] = merged["meta_target_side"].fillna("none").astype(str)
    merged["label_fraction"] = [safe_float(value) if safe_text(value) else None for value in merged["label_fraction"].tolist()]
    merged["test_frac"] = [safe_float(value) if safe_text(value) else None for value in merged["test_frac"].tolist()]
    merged["shots_per_class"] = merged["shots_per_class"].apply(lambda value: safe_int(value, 0))
    drop_columns = [column for column in merged.columns if column.endswith("__overlay") or column == "seed"]
    merged = merged.drop(columns=drop_columns, errors="ignore")
    return merged


def write_normalized_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_normalized_dataset(config: Dict[str, Any], dataset: str) -> pd.DataFrame:
    path = normalized_dataset_path(config, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Normalized dataset not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = False if column in BOOLEAN_COLUMNS else ""
    frame["is_labeled"] = frame["is_labeled"].apply(as_bool)
    frame["benchmark_ready"] = frame["benchmark_ready"].apply(as_bool)
    labels = []
    for is_labeled, value in zip(frame["is_labeled"].tolist(), frame["label"].tolist()):
        labels.append(safe_int(value, -1) if as_bool(is_labeled) else "")
    frame["label"] = labels
    return frame[STANDARD_COLUMNS]


def choose_group_holdout(
    labeled_frame: pd.DataFrame,
    group_column: str,
    test_frac: float,
    seed: int,
    attempts: int = 512,
) -> List[str]:
    if labeled_frame.empty:
        return []
    work = labeled_frame.copy()
    work[group_column] = work[group_column].fillna("").astype(str)
    work = work[work[group_column] != ""].reset_index(drop=True)
    if work.empty:
        return []
    target_rows = int(round(len(work) * float(test_frac)))
    target_rows = max(1, min(target_rows, len(work) - 1))
    grouped = work.groupby(group_column)
    overall_pos = int((work["label"].astype(int) == 1).sum())
    overall_ratio = overall_pos / max(len(work), 1)
    group_stats: List[Dict[str, Any]] = []
    for group_value, group_frame in grouped:
        labels = group_frame["label"].astype(int)
        group_stats.append(
            {
                "group": safe_text(group_value),
                "count": int(len(group_frame)),
                "pos": int((labels == 1).sum()),
                "neg": int((labels == 0).sum()),
            }
        )
    rng = random.Random(seed)
    best_groups: List[str] = []
    best_score = float("inf")
    for _ in range(attempts):
        shuffled = list(group_stats)
        rng.shuffle(shuffled)
        chosen: List[Dict[str, Any]] = []
        total = 0
        for item in shuffled:
            if total >= target_rows:
                break
            chosen.append(item)
            total += int(item["count"])
        improved = True
        while improved and chosen:
            improved = False
            current_total = sum(int(item["count"]) for item in chosen)
            current_error = abs(current_total - target_rows)
            current_pos = sum(int(item["pos"]) for item in chosen)
            current_ratio = current_pos / max(current_total, 1)
            for candidate in list(chosen):
                next_total = current_total - int(candidate["count"])
                if next_total <= 0:
                    continue
                next_error = abs(next_total - target_rows)
                next_pos = current_pos - int(candidate["pos"])
                next_ratio = next_pos / max(next_total, 1)
                if (next_error, abs(next_ratio - overall_ratio)) <= (current_error, abs(current_ratio - overall_ratio)):
                    chosen.remove(candidate)
                    improved = True
                    break
        total = sum(int(item["count"]) for item in chosen)
        pos = sum(int(item["pos"]) for item in chosen)
        ratio = pos / max(total, 1)
        score = abs(total - target_rows) + 0.25 * abs(ratio - overall_ratio) * len(work)
        if score < best_score:
            best_score = score
            best_groups = [safe_text(item["group"]) for item in chosen]
    return sorted({item for item in best_groups if item})


def stratified_label_sample(
    labeled_train: pd.DataFrame,
    label_fraction: float,
    seed: int,
) -> List[str]:
    if labeled_train.empty:
        return []
    rng = random.Random(seed)
    chosen: List[str] = []
    fraction = max(0.0, min(float(label_fraction), 1.0))
    for label_value in [0, 1]:
        group = labeled_train[labeled_train["label"].astype(int) == label_value]["row_id"].astype(str).tolist()
        rng.shuffle(group)
        if fraction >= 1.0:
            take = len(group)
        else:
            take = max(1, int(round(len(group) * fraction)))
            take = min(take, len(group))
        chosen.extend(group[:take])
    return sorted(set(chosen))


def fewshot_support_query_overlay(
    frame: pd.DataFrame,
    shots_per_class: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    overlay = default_analysis_overlay(frame, "fewshot_target", seed)
    overlay["split"] = "train"
    overlay["pretrain_use"] = True
    overlay["supervised_use"] = overlay["is_labeled"]
    overlay["label_visible"] = overlay["is_labeled"]
    overlay["fewshot_role"] = "base_train"
    overlay["meta_target_side"] = "base"
    overlay["shots_per_class"] = int(shots_per_class)

    labeled = overlay[overlay["is_labeled"] == True].copy().reset_index(drop=True)
    labeled["label"] = labeled["label"].astype(int)
    nested_max_shots = max(int(shots_per_class), 8)
    grouped = labeled.groupby("target_id")["label"].agg(
        total="count",
        pos=lambda s: int((s == 1).sum()),
        neg=lambda s: int((s == 0).sum()),
    )
    eligible = grouped[
        (grouped["pos"] >= nested_max_shots + 1) & (grouped["neg"] >= nested_max_shots + 1)
    ].index.tolist()
    rng = random.Random(seed)
    rng.shuffle(eligible)
    meta_target_count = max(1, int(math.ceil(len(eligible) * 0.20))) if eligible else 0
    meta_targets = sorted(eligible[:meta_target_count])
    for target_id in meta_targets:
        target_rows = labeled[labeled["target_id"] == target_id].copy()
        pos_rows = target_rows[target_rows["label"] == 1]["row_id"].astype(str).tolist()
        neg_rows = target_rows[target_rows["label"] == 0]["row_id"].astype(str).tolist()
        rng.shuffle(pos_rows)
        rng.shuffle(neg_rows)
        support_ids = set(pos_rows[:shots_per_class] + neg_rows[:shots_per_class])
        query_fixed_exclusion_ids = set(pos_rows[:nested_max_shots] + neg_rows[:nested_max_shots])
        target_mask = overlay["target_id"].fillna("").astype(str).eq(target_id)
        overlay.loc[target_mask, "split"] = "meta_test"
        overlay.loc[target_mask, "pretrain_use"] = False
        overlay.loc[target_mask, "supervised_use"] = False
        overlay.loc[target_mask, "meta_target_side"] = "meta_test"
        overlay.loc[target_mask, "fewshot_role"] = "none"
        support_mask = target_mask & overlay["row_id"].isin(support_ids)
        query_mask = target_mask & overlay["is_labeled"] & ~overlay["row_id"].isin(query_fixed_exclusion_ids)
        overlay.loc[support_mask, "fewshot_role"] = "support"
        overlay.loc[query_mask, "fewshot_role"] = "query"
    report = {
        "scheme": "fewshot_target",
        "seed": int(seed),
        "shots_per_class": int(shots_per_class),
        "nested_query_max_shots_per_class": int(nested_max_shots),
        "eligible_targets": int(len(eligible)),
        "meta_test_targets": int(len(meta_targets)),
        "meta_test_target_ids": meta_targets,
        "support_rows": int((overlay["fewshot_role"] == "support").sum()),
        "query_rows": int((overlay["fewshot_role"] == "query").sum()),
        "base_train_rows": int((overlay["fewshot_role"] == "base_train").sum()),
    }
    return standardize_analysis_split_overlay(overlay), report


def build_analysis_split_overlay(
    config: Dict[str, Any],
    frame: pd.DataFrame,
    scheme: str,
    seed: int,
    label_fraction: Optional[float] = None,
    test_frac: Optional[float] = None,
    shots_per_class: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    overlay = default_analysis_overlay(frame, scheme, seed)
    report: Dict[str, Any] = {"scheme": scheme, "seed": int(seed)}
    if scheme == "official":
        report.update(
            {
                "n_rows": int(len(overlay)),
                "n_train": int((overlay["split"] == "train").sum()),
                "n_test": int((overlay["split"] == "test").sum()),
            }
        )
        return overlay, report
    if scheme == "low_label":
        if label_fraction is None:
            raise ValueError("low_label split requires --label-fraction")
        official_train_mask = overlay["split"].eq("train")
        labeled_train = overlay[official_train_mask & overlay["is_labeled"]].copy().reset_index(drop=True)
        chosen_ids = set(stratified_label_sample(labeled_train, float(label_fraction), seed))
        overlay["pretrain_use"] = overlay["split"].eq("train")
        overlay["supervised_use"] = overlay["split"].eq("train") & overlay["row_id"].isin(chosen_ids)
        overlay["label_visible"] = overlay["is_labeled"] & (overlay["split"].ne("train") | overlay["row_id"].isin(chosen_ids))
        overlay["label_fraction"] = float(label_fraction)
        report.update(
            {
                "label_fraction": float(label_fraction),
                "official_test_rows": int((overlay["split"] == "test").sum()),
                "visible_labeled_train_rows": int(overlay["supervised_use"].sum()),
                "hidden_labeled_train_rows": int((official_train_mask & overlay["is_labeled"] & ~overlay["supervised_use"]).sum()),
                "pretrain_train_rows": int(overlay["pretrain_use"].sum()),
            }
        )
        return standardize_analysis_split_overlay(overlay), report
    if scheme in {
        "target_holdout",
        "target_e3_holdout",
        "recruiter_holdout",
        "target_family_holdout",
        "scaffold_holdout",
    }:
        if test_frac is None:
            raise ValueError(f"{scheme} requires --test-frac")
        labeled = overlay[overlay["is_labeled"]].copy().reset_index(drop=True)
        if scheme == "target_holdout":
            group_column = "target_id"
        elif scheme == "target_e3_holdout":
            overlay["target_e3_pair"] = overlay["target_id"].fillna("").astype(str) + "||" + overlay["recruiter_id"].fillna("").astype(str)
            labeled = overlay[overlay["is_labeled"]].copy().reset_index(drop=True)
            group_column = "target_e3_pair"
        elif scheme == "recruiter_holdout":
            group_column = "recruiter_id"
        elif scheme == "target_family_holdout":
            cluster_map = target_sequence_cluster_map(frame, threshold=0.70)
            overlay["sequence_cluster_family"] = overlay["target_id"].fillna("").astype(str).map(cluster_map).fillna("")
            labeled = overlay[overlay["is_labeled"]].copy().reset_index(drop=True)
            group_column = "sequence_cluster_family"
        else:
            group_column = "scaffold"
        held_out_groups = choose_group_holdout(labeled, group_column, float(test_frac), seed)
        held_out_mask = overlay[group_column].fillna("").astype(str).isin(held_out_groups)
        overlay["split"] = np.where(held_out_mask, "test", "train")
        overlay["pretrain_use"] = ~held_out_mask
        overlay["supervised_use"] = (~held_out_mask) & overlay["is_labeled"]
        overlay["label_visible"] = overlay["is_labeled"]
        overlay["fewshot_role"] = "none"
        overlay["meta_target_side"] = "none"
        overlay["test_frac"] = float(test_frac)
        report.update(
            {
                "test_frac": float(test_frac),
                "held_out_groups": held_out_groups,
                "n_test_rows": int(held_out_mask.sum()),
                "n_test_labeled": int((held_out_mask & overlay["is_labeled"]).sum()),
                "n_train_rows": int((~held_out_mask).sum()),
                "n_train_labeled": int(((~held_out_mask) & overlay["is_labeled"]).sum()),
            }
        )
        return standardize_analysis_split_overlay(
            overlay.drop(columns=["target_e3_pair", "sequence_cluster_family"], errors="ignore")
        ), report
    if scheme == "fewshot_target":
        if shots_per_class is None:
            raise ValueError("fewshot_target requires --shots-per-class")
        return fewshot_support_query_overlay(frame, int(shots_per_class), seed)
    raise ValueError(f"Unsupported analysis split scheme: {scheme}")


def write_analysis_split_files(
    config: Dict[str, Any],
    dataset: str,
    scheme: str,
    seed: int,
    overlay: pd.DataFrame,
    report: Dict[str, Any],
) -> Tuple[Path, Path]:
    stamp = timestamp_now()
    csv_path = output_dir(config) / f"{stamp}_{dataset}_{scheme}_seed{seed}_split.csv"
    report_path = output_dir(config) / f"{stamp}_{dataset}_{scheme}_seed{seed}_split_report.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(csv_path, index=False)
    write_json(report_path, report)
    return csv_path, report_path


def split_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    ensure_project_dirs(config)
    dataset = safe_text(args.dataset)
    if dataset != "protac8k":
        raise ValueError("split currently supports only --dataset protac8k")
    scheme = safe_text(args.scheme)
    seed = int(args.seed)
    frame = load_normalized_dataset(config, dataset)
    overlay, report = build_analysis_split_overlay(
        config,
        frame,
        scheme,
        seed,
        label_fraction=getattr(args, "label_fraction", None),
        test_frac=getattr(args, "test_frac", None),
        shots_per_class=getattr(args, "shots_per_class", None),
    )
    csv_path, report_path = write_analysis_split_files(config, dataset, scheme, seed, overlay, report)
    print(f"split_csv: {csv_path}")
    print(f"split_report: {report_path}")


def describe_array_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, np.ndarray):
        return {"type": "ndarray", "shape": list(payload.shape), "dtype": str(payload.dtype)}
    if isinstance(payload, torch.Tensor):
        return {"type": "tensor", "shape": list(payload.shape), "dtype": str(payload.dtype)}
    if isinstance(payload, (list, tuple)):
        description: Dict[str, Any] = {"type": type(payload).__name__, "length": len(payload)}
        if payload and isinstance(payload[0], (torch.Tensor, np.ndarray)):
            description["first_item_shape"] = list(payload[0].shape)
        return description
    if isinstance(payload, dict):
        return {"type": "dict", "keys": list(payload.keys())[:20], "length": len(payload)}
    return {"type": type(payload).__name__}


def inventory_dataset_root(dataset_root: Path) -> Dict[str, Any]:
    files_by_suffix: Dict[str, List[str]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in DATASET_FILE_SUFFIXES.union({".zip"}):
            files_by_suffix[path.suffix.lower()].append(str(path.relative_to(dataset_root)))
    top_level_folders = sorted([path.name for path in dataset_root.iterdir() if path.is_dir()]) if dataset_root.exists() else []
    return {
        "top_level_folders": top_level_folders,
        "files_by_suffix": dict(files_by_suffix),
    }


def extract_protac_zip(zip_path: Path, destination: Path, force: bool) -> int:
    import zipfile

    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_name = member.filename
            if member_name.endswith("/") or member_name.startswith("__MACOSX/"):
                continue
            if member_name.startswith("PROTAC-8K/"):
                member_name = member_name[len("PROTAC-8K/") :]
            if not member_name:
                continue
            out_path = destination / member_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not force:
                continue
            with archive.open(member) as source, out_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    return extracted


def download_with_progress(url: str, target_path: Path, desc: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    with tmp_path.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))
    tmp_path.replace(target_path)


def resolve_local_name_json(dataset_root: Path) -> Optional[Path]:
    candidates = [path for path in dataset_root.rglob("name.json") if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def fetch_official_name_json(config: Dict[str, Any], force: bool = False) -> Path:
    target = official_name_json_path(config)
    if target.exists() and not force:
        return target
    primary_url = config["benchmark"]["official_name_json_url"]
    fallback_urls = [
        "https://cdn.jsdelivr.net/gh/ABILiLab/DegradeMaster@main/data/PROTAC/name.json",
        primary_url,
        "https://github.com/ABILiLab/DegradeMaster/raw/main/data/PROTAC/name.json",
    ]
    last_error: Optional[Exception] = None
    for url in fallback_urls:
        for timeout in (60, 180):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                return target
            except requests.RequestException as exc:
                last_error = exc
                continue
    raise RuntimeError(f"Failed to fetch official name.json from all configured mirrors: {last_error}")


def resolve_official_name_json(
    config: Dict[str, Any],
    dataset_root: Path,
    auto_fetch: bool,
) -> Tuple[Optional[Path], str]:
    local = resolve_local_name_json(dataset_root)
    if local is not None:
        return local, "local_name_json"
    cached = official_name_json_path(config)
    if cached.exists():
        return cached, "cached_official_name_json"
    if auto_fetch:
        fetched = fetch_official_name_json(config, force=False)
        return fetched, "fetched_official_name_json"
    return None, "missing_name_json"


def ordered_name_json_items(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, Any]:
        key = safe_text(item[0])
        if key.isdigit():
            return (0, int(key))
        return (1, key)

    return sorted(payload.items(), key=sort_key)


def parse_named_accession(raw_path: str) -> Tuple[str, str]:
    stem = Path(safe_text(raw_path)).stem
    if "_" not in stem:
        return stem, ""
    name, accession = stem.rsplit("_", 1)
    accession = accession.upper()
    if is_probable_uniprot(accession):
        return name, accession
    return stem, ""


def resolve_asset_path(dataset_root: Path, asset_dir: str, relative_name: str) -> str:
    filename = Path(safe_text(relative_name)).name
    if not filename:
        return ""
    return str(dataset_root / asset_dir / filename)


def maybe_load_aux_assets(dataset_root: Path) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}
    asset_files = {
        "protac_feature": dataset_root / "features" / "protac_feature.npy",
        "target_feature": dataset_root / "features" / "target_feature.npy",
        "e3_feature": dataset_root / "features" / "e3_feature.npy",
        "labels": dataset_root / "processed" / "label.pt",
    }
    for key, path in asset_files.items():
        if not path.exists():
            continue
        if path.suffix.lower() == ".npy":
            assets[key] = np.load(path, allow_pickle=True)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, torch.Tensor):
                assets[key] = payload.detach().cpu().numpy()
            else:
                assets[key] = np.asarray(payload)
    return assets


def protac_feature_row_cache(
    aux_assets: Dict[str, Any],
    index: int,
    protac_path: str,
    target_path: str,
    recruiter_path: str,
) -> Dict[str, Any]:
    row_cache: Dict[str, Any] = {
        "protac_path": protac_path,
        "target_pocket_path": target_path,
        "recruiter_pocket_path": recruiter_path,
        "source_index": index,
    }
    if "protac_feature" in aux_assets and index < len(aux_assets["protac_feature"]):
        row_cache["aux_protac_feature"] = np.asarray(aux_assets["protac_feature"][index], dtype=np.float32)
    if "target_feature" in aux_assets and index < len(aux_assets["target_feature"]):
        row_cache["aux_target_feature"] = np.asarray(aux_assets["target_feature"][index], dtype=np.float32)
    if "e3_feature" in aux_assets and index < len(aux_assets["e3_feature"]):
        row_cache["aux_recruiter_feature"] = np.asarray(aux_assets["e3_feature"][index], dtype=np.float32)
    return row_cache


def build_protac_name_json_frame(
    config: Dict[str, Any],
    dataset_root: Path,
    payload: Dict[str, Any],
    metadata_source: str,
    hydrate_sequences: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ordered_items = ordered_name_json_items(payload)
    aux_assets = maybe_load_aux_assets(dataset_root)
    rows: List[Dict[str, Any]] = []
    dataset_cache = {"rows": {}, "meta": {}}
    canonical_failures = 0
    target_parse_failures = 0
    recruiter_parse_failures = 0

    for index, (row_key, record) in enumerate(ordered_items):
        row = empty_standard_row("protac8k", "degrademaster_name_json", safe_text(row_key))
        label = normalize_label_value(record.get("label"))
        row["label"] = label if label is not None else ""
        row["is_labeled"] = label is not None
        row["smiles"] = safe_text(
            record.get("smiles")
            or record.get("SMILES")
            or record.get("link_smile")
            or record.get("protac_smile")
            or record.get("PROTAC SMILES")
        )
        row["data_protocol"] = "degrademaster_name_json"
        row["label_source"] = "official_name_json_label"

        target_name, target_id = parse_named_accession(record.get("tar_path", ""))
        if not target_id and safe_text(record.get("target_pocket_path")):
            target_name, target_id = parse_named_accession(record.get("target_pocket_path", ""))
        recruiter_name, recruiter_id = parse_named_accession(record.get("e3_ligase_path", ""))
        if not recruiter_id and safe_text(record.get("recruiter_pocket_path")):
            recruiter_name, recruiter_id = parse_named_accession(record.get("recruiter_pocket_path", ""))
        if not target_id:
            target_parse_failures += 1
        if not recruiter_id:
            recruiter_parse_failures += 1

        row["target_id"] = target_id
        row["target_name"] = safe_text(target_name or target_id)
        row["recruiter_id"] = recruiter_id
        row["recruiter_name"] = safe_text(recruiter_name or recruiter_id)
        row["target_id_source"] = "parsed_tar_path" if target_id else "missing"
        row["recruiter_id_source"] = "parsed_e3_ligase_path" if recruiter_id else "missing"
        row["target_sequence_source"] = "missing"
        row["recruiter_sequence_source"] = "missing"

        protac_path = resolve_asset_path(dataset_root, "protac", record.get("protac_path", ""))
        target_pocket_path = resolve_asset_path(dataset_root, "target_pocket", record.get("tar_path", ""))
        target_ligand_path = resolve_asset_path(dataset_root, "target_ligand", record.get("war_path", ""))
        recruiter_pocket_path = resolve_asset_path(dataset_root, "ligase_pocket", record.get("e3_ligase_path", ""))
        recruiter_ligand_path = resolve_asset_path(dataset_root, "ligase_ligand", record.get("e3_lig_path", ""))
        row["protac_path"] = protac_path
        row["target_pocket_path"] = target_pocket_path
        row["target_ligand_path"] = target_ligand_path
        row["recruiter_pocket_path"] = recruiter_pocket_path
        row["recruiter_ligand_path"] = recruiter_ligand_path
        row["task_id"] = row["target_id"] or row["target_name"] or row["row_id"]
        row["benchmark_ready"] = bool(target_id and recruiter_id)

        canonical_smiles = ""
        if row["smiles"] and RDKIT_AVAILABLE:
            canonical_smiles = canonicalize_smiles(row["smiles"])
            if not canonical_smiles:
                canonical_failures += 1
        elif row["smiles"]:
            canonical_failures += 1
        row_cache = protac_feature_row_cache(
            aux_assets,
            index,
            protac_path,
            target_pocket_path,
            recruiter_pocket_path,
        )
        if canonical_smiles:
            row_cache["canonical_smiles"] = canonical_smiles
        dataset_cache["rows"][row["row_id"]] = row_cache
        rows.append(row)

    frame = pd.DataFrame(rows)
    split_frame, split_policy, split_report = make_degrademaster_code_split(
        frame,
        float(config["benchmark"]["train_rate"]),
        int(config["benchmark"]["seed"]),
    )
    split_frame["split_policy"] = split_policy
    if hydrate_sequences:
        split_frame, sequence_stats = ensure_sequences(split_frame, config)
    else:
        sequence_stats = {"fetched": 0, "cache_hits": 0, "failed": 0}
    split_frame["benchmark_ready"] = (
        split_frame["data_protocol"].eq("degrademaster_name_json")
        & split_frame["target_id_source"].ne("missing")
        & split_frame["recruiter_id_source"].ne("missing")
        & split_frame["split_policy"].eq(split_policy)
    )
    split_frame = standardize_frame_columns(split_frame)
    dataset_cache["meta"] = {
        "source": "degrademaster_name_json",
        "metadata_source": metadata_source,
        "split_policy": split_policy,
        "benchmark_ready": bool(split_frame["benchmark_ready"].all()),
        "feature_alignment_assumption": "official_name_json_order_to_feature_row_index",
    }
    metadata = {
        "source": "degrademaster_name_json",
        "metadata_source": metadata_source,
        "row_count": int(len(split_frame)),
        "official_entry_count": int(len(ordered_items)),
        "split_policy": split_policy,
        "sequence_stats": sequence_stats,
        "canonical_smiles_failures": int(canonical_failures),
        "target_parse_failures": int(target_parse_failures),
        "recruiter_parse_failures": int(recruiter_parse_failures),
        "feature_shapes": {
            key: list(value.shape) if isinstance(value, np.ndarray) else describe_array_payload(value)
            for key, value in aux_assets.items()
        },
        "feature_alignment_assumption": "official_name_json_order_to_feature_row_index",
    }
    return split_frame, metadata, dataset_cache, split_report


def build_leakage_report(frame: pd.DataFrame, split_policy: str) -> Dict[str, Any]:
    train_frame = frame[frame["split"] == "train"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    def canonical_smiles_value(row: pd.Series) -> str:
        smiles = safe_text(row.get("smiles", ""))
        if not smiles:
            return ""
        if RDKIT_AVAILABLE:
            canonical = canonicalize_smiles(smiles)
            if canonical:
                return canonical
        return smiles

    for work in [train_frame, test_frame]:
        work["canonical_smiles"] = work.apply(canonical_smiles_value, axis=1)

    def overlap_count(columns: Sequence[str]) -> int:
        train_keys = {
            tuple(safe_text(value) for value in row)
            for row in train_frame[list(columns)].itertuples(index=False, name=None)
            if any(safe_text(value) for value in row)
        }
        test_keys = {
            tuple(safe_text(value) for value in row)
            for row in test_frame[list(columns)].itertuples(index=False, name=None)
            if any(safe_text(value) for value in row)
        }
        return int(len(train_keys & test_keys))

    return {
        "protocol": split_policy,
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "overlap_counts": {
            "canonical_smiles": overlap_count(["canonical_smiles"]),
            "target_id": overlap_count(["target_id"]),
            "target_id_canonical_smiles": overlap_count(["target_id", "canonical_smiles"]),
            "target_id_recruiter_id_canonical_smiles": overlap_count(
                ["target_id", "recruiter_id", "canonical_smiles"]
            ),
        },
        "note": (
            "These overlaps are audited only. The default code split is not leakage-filtered because "
            "the visible DegradeMaster split function does not implement the paper-described exclusion."
        ),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def normalize_external_dataset(
    raw_frame: pd.DataFrame,
    dataset: str,
    source: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    column_map = {
        "row_id": find_column(raw_frame, ROW_ID_ALIASES),
        "label": find_column(raw_frame, LABEL_ALIASES),
        "split": find_column(raw_frame, SPLIT_ALIASES),
        "smiles": find_column(raw_frame, SMILES_ALIASES),
        "target_id": find_column(raw_frame, TARGET_ID_ALIASES),
        "target_name": find_column(raw_frame, TARGET_NAME_ALIASES),
        "target_sequence": find_column(raw_frame, TARGET_SEQUENCE_ALIASES),
        "recruiter_id": find_column(raw_frame, RECRUITER_ID_ALIASES),
        "recruiter_name": find_column(raw_frame, RECRUITER_NAME_ALIASES),
        "recruiter_sequence": find_column(raw_frame, RECRUITER_SEQUENCE_ALIASES),
        "cell_line": find_column(raw_frame, CELL_LINE_ALIASES),
        "dc50_nm": find_column(raw_frame, DC50_ALIASES),
        "dmax_percent": find_column(raw_frame, DMAX_ALIASES),
        "degradation_percent": find_column(raw_frame, DEGRADATION_PERCENT_ALIASES),
        "assay_text": find_column(raw_frame, ASSAY_TEXT_ALIASES),
        "task_id": find_column(raw_frame, TASK_ID_ALIASES),
    }
    rows = []
    label_rule_counts: Dict[str, int] = defaultdict(int)
    for idx, record in raw_frame.iterrows():
        row_id = safe_text(record[column_map["row_id"]]) if column_map["row_id"] else f"{dataset}_{idx:05d}"
        row = empty_standard_row(dataset, source, row_id)
        row["smiles"] = safe_text(record[column_map["smiles"]]) if column_map["smiles"] else ""
        row["target_id"] = safe_text(record[column_map["target_id"]]) if column_map["target_id"] else ""
        row["target_name"] = safe_text(record[column_map["target_name"]]) if column_map["target_name"] else ""
        row["target_sequence"] = (
            safe_text(record[column_map["target_sequence"]]) if column_map["target_sequence"] else ""
        )
        row["recruiter_id"] = safe_text(record[column_map["recruiter_id"]]) if column_map["recruiter_id"] else ""
        row["recruiter_name"] = (
            safe_text(record[column_map["recruiter_name"]]) if column_map["recruiter_name"] else ""
        )
        row["recruiter_sequence"] = (
            safe_text(record[column_map["recruiter_sequence"]]) if column_map["recruiter_sequence"] else ""
        )
        row["cell_line"] = safe_text(record[column_map["cell_line"]]) if column_map["cell_line"] else ""
        row["dc50_nm"] = safe_float(record[column_map["dc50_nm"]]) if column_map["dc50_nm"] else None
        row["dmax_percent"] = (
            safe_float(record[column_map["dmax_percent"]]) if column_map["dmax_percent"] else None
        )
        row["degradation_percent"] = (
            safe_float(record[column_map["degradation_percent"]]) if column_map["degradation_percent"] else None
        )
        row["assay_text"] = safe_text(record[column_map["assay_text"]]) if column_map["assay_text"] else ""
        raw_label = record[column_map["label"]] if column_map["label"] else None
        label, label_rule = infer_label_and_rule(
            dataset,
            raw_label,
            row["dc50_nm"],
            row["dmax_percent"],
            row["degradation_percent"],
            config,
        )
        label_rule_counts[label_rule] += 1
        row["is_labeled"] = label is not None
        row["label"] = label if label is not None else ""
        row["split"] = safe_text(record[column_map["split"]]) if column_map["split"] else ""
        row["split"] = standardize_split_value(row["split"])
        row["task_id"] = safe_text(record[column_map["task_id"]]) if column_map["task_id"] else ""
        row["data_protocol"] = f"{dataset}_external_validation"
        row["label_source"] = label_rule
        row["target_id_source"] = "table" if row["target_id"] else "missing"
        row["recruiter_id_source"] = "table" if row["recruiter_id"] else "missing"
        row["target_sequence_source"] = "manual" if row["target_sequence"] else "missing"
        row["recruiter_sequence_source"] = "manual" if row["recruiter_sequence"] else "missing"
        if not row["task_id"]:
            row["task_id"] = row["target_id"] or row["target_name"] or f"{dataset}_task_{idx:05d}"
        if not row["target_name"] and row["target_id"]:
            row["target_name"] = row["target_id"]
        if not row["recruiter_name"] and row["recruiter_id"]:
            row["recruiter_name"] = row["recruiter_id"]
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame, seq_stats = ensure_sequences(frame, config)
    explicit = detect_explicit_split(frame)
    if explicit is not None and explicit.astype(str).str.len().sum() > 0:
        frame["split"] = explicit
        split_policy = "official_split_from_dataset"
    else:
        ratios = tuple(float(x) for x in config["benchmark"].get("train_val_test", [0.7, 0.15, 0.15]))
        frame, split_policy = assign_random_split(frame, int(config["benchmark"]["seed"]), ratios)
    frame["split_policy"] = split_policy
    frame = standardize_frame_columns(frame)
    metadata = {
        "column_map": column_map,
        "label_rule_counts": dict(label_rule_counts),
        "sequence_stats": seq_stats,
        "split_policy": split_policy,
    }
    return frame, metadata


def prepare_protac_asset_fallback(
    config: Dict[str, Any],
    dataset_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    protac_feature_path = dataset_root / "features" / "protac_feature.npy"
    target_feature_path = dataset_root / "features" / "target_feature.npy"
    e3_feature_path = dataset_root / "features" / "e3_feature.npy"
    label_path = dataset_root / "processed" / "label.pt"
    missing = [
        str(path)
        for path in [protac_feature_path, target_feature_path, e3_feature_path, label_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"PROTAC-8K asset fallback files missing: {missing}")

    protac_feature = np.load(protac_feature_path, allow_pickle=True)
    target_feature = np.load(target_feature_path, allow_pickle=True)
    e3_feature = np.load(e3_feature_path, allow_pickle=True)
    labels = np.array(torch.load(label_path, map_location="cpu", weights_only=False))
    protac_files = sort_protac_mol2_files(list((dataset_root / "protac").glob("*_protac.mol2")))
    row_count = min(len(protac_feature), len(target_feature), len(e3_feature), len(labels), len(protac_files))
    target_clusters = stable_cluster_ids(target_feature[:row_count], "target_cluster")
    recruiter_clusters = stable_cluster_ids(e3_feature[:row_count], "recruiter_cluster")

    rows = []
    dataset_cache = {"rows": {}, "meta": {}}
    smiles_failures = 0
    atom_counts: List[int] = []
    for idx in tqdm(range(row_count), desc="prep protac8k asset fallback"):
        mol_path = protac_files[idx]
        row_id = f"protac8k_{extract_sample_id_from_protac_file(mol_path):05d}"
        row = empty_standard_row("protac8k", "zenodo_processed_assets_fallback", row_id)
        mol = mol_from_mol2(mol_path) if RDKIT_AVAILABLE else None
        smiles = ""
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                atom_counts.append(int(mol.GetNumAtoms()))
            except Exception:
                atom_counts.append(count_atoms_from_mol2(mol_path))
                smiles_failures += 1
        else:
            atom_counts.append(count_atoms_from_mol2(mol_path))
            smiles_failures += 1
        label_value = normalize_label_value(labels[idx])
        row["is_labeled"] = label_value is not None
        row["label"] = label_value if label_value is not None else ""
        row["smiles"] = smiles
        row["target_id"] = target_clusters[idx]
        row["target_name"] = target_clusters[idx]
        row["recruiter_id"] = recruiter_clusters[idx]
        row["recruiter_name"] = recruiter_clusters[idx]
        row["target_sequence_source"] = "fallback"
        row["recruiter_sequence_source"] = "fallback"
        row["task_id"] = row["target_id"]
        row["data_protocol"] = "asset_fallback_surrogate_target"
        row["split_policy"] = f"degrademaster_code_split_seed{int(config['benchmark']['seed'])}"
        row["label_source"] = "original_dataset_label"
        row["target_id_source"] = "clustered_target_feature"
        row["recruiter_id_source"] = "clustered_e3_feature"
        row["benchmark_ready"] = False
        row["protac_path"] = str(mol_path)
        dataset_cache["rows"][row_id] = {
            "protac_path": str(mol_path),
            "aux_protac_feature": protac_feature[idx].astype(np.float32),
            "aux_target_feature": target_feature[idx].astype(np.float32),
            "aux_recruiter_feature": e3_feature[idx].astype(np.float32),
            "target_cluster": target_clusters[idx],
            "recruiter_cluster": recruiter_clusters[idx],
            "source_index": idx,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame, split_policy, split_report = make_degrademaster_code_split(
        frame,
        float(config["benchmark"]["train_rate"]),
        int(config["benchmark"]["seed"]),
    )
    frame["split_policy"] = split_policy
    frame = standardize_frame_columns(frame)
    dataset_cache["meta"] = {
        "source": "zenodo_processed_assets_fallback",
        "split_policy": split_policy,
        "benchmark_ready": False,
        "reason": "surrogate target/recruiter IDs; not comparable to DegradeMaster",
    }
    metadata = {
        "source": "zenodo_processed_assets_fallback",
        "split_policy": split_policy,
        "row_count": row_count,
        "smiles_failures": smiles_failures,
        "max_atoms": max(atom_counts) if atom_counts else 0,
        "avg_atoms": float(np.mean(atom_counts)) if atom_counts else 0.0,
        "target_cluster_count": len(set(target_clusters)),
        "recruiter_cluster_count": len(set(recruiter_clusters)),
        "feature_shapes": {
            "protac_feature": list(protac_feature.shape),
            "target_feature": list(target_feature.shape),
            "e3_feature": list(e3_feature.shape),
            "labels": list(labels.shape),
        },
        "reason": "surrogate target/recruiter IDs; not comparable to DegradeMaster",
    }
    return frame, metadata, dataset_cache, split_report


def maybe_note_asset_fallback_warning(config: Dict[str, Any]) -> None:
    append_note(
        config,
        f"Asset Fallback Warning {timestamp_now()}",
        [
            "- This run is a smoke test only and must not be used for paper benchmark comparison.",
            "- benchmark_ready: false",
            "- data_protocol: asset_fallback_surrogate_target",
            "- reason: surrogate target/recruiter IDs; not comparable to DegradeMaster",
        ],
    )


def build_protac8k_from_official_metadata(
    config: Dict[str, Any],
    dataset_root: Path,
    hydrate_sequences: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    name_path, source_kind = resolve_official_name_json(config, dataset_root, auto_fetch=True)
    if name_path is None:
        if not as_bool(config["benchmark"].get("allow_asset_fallback", False)):
            raise FileNotFoundError(
                "Official PROTAC metadata not found and asset fallback is disabled. "
                "Run audit to inspect assets or enable benchmark.allow_asset_fallback for smoke tests."
            )
        print("WARNING: falling back to surrogate asset protocol. This is not benchmark-comparable.")
        maybe_note_asset_fallback_warning(config)
        return prepare_protac_asset_fallback(config, dataset_root)

    payload = json.loads(name_path.read_text(encoding="utf-8"))
    return build_protac_name_json_frame(config, dataset_root, payload, source_kind, hydrate_sequences)


def write_split_report(config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    write_json(split_report_path(config), payload)


def write_leakage_report(config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    write_json(leakage_report_path(config), payload)


def build_path_coverage(dataset_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "protac_path": ("protac", "protac_path"),
        "tar_path": ("target_pocket", "tar_path"),
        "war_path": ("target_ligand", "war_path"),
        "e3_ligase_path": ("ligase_pocket", "e3_ligase_path"),
        "e3_lig_path": ("ligase_ligand", "e3_lig_path"),
    }
    rows = ordered_name_json_items(payload)
    coverage: Dict[str, Any] = {}
    for report_key, (asset_dir, json_key) in mapping.items():
        present_rows = 0
        missing_rows = 0
        unique_present = set()
        unique_missing = set()
        for _, record in rows:
            relative_name = safe_text(record.get(json_key, ""))
            if not relative_name:
                missing_rows += 1
                continue
            asset_path = Path(resolve_asset_path(dataset_root, asset_dir, relative_name))
            if asset_path.exists():
                present_rows += 1
                unique_present.add(asset_path.name)
            else:
                missing_rows += 1
                unique_missing.add(asset_path.name)
        coverage[report_key] = {
            "present_rows": int(present_rows),
            "missing_rows": int(missing_rows),
            "unique_present": int(len(unique_present)),
            "unique_missing": int(len(unique_missing)),
        }
    return coverage


def audit_protac8k(config: Dict[str, Any], check_leakage: bool) -> Dict[str, Any]:
    dataset_root = protac8k_root(config)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")
    inventory = inventory_dataset_root(dataset_root)
    zip_files = sorted(dataset_root.glob("*.zip"))
    official_path, official_source = resolve_official_name_json(config, dataset_root, auto_fetch=True)
    official_payload: Dict[str, Any] = {}
    official_summary: Dict[str, Any] = {"source": official_source, "path": str(official_path) if official_path else ""}
    if official_path is not None:
        official_payload = json.loads(official_path.read_text(encoding="utf-8"))
        label_counter = Counter(safe_int(value.get("label"), -999) for value in official_payload.values())
        official_summary.update(
            {
                "entry_count": int(len(official_payload)),
                "label_counts": {
                    "1": int(label_counter.get(1, 0)),
                    "0": int(label_counter.get(0, 0)),
                    "-1": int(label_counter.get(-1, 0)),
                },
                "path_coverage": build_path_coverage(dataset_root, official_payload),
            }
        )

    feature_shapes: Dict[str, Any] = {}
    processed_label_count = 0
    local_assets = {
        "name_json_exists": resolve_local_name_json(dataset_root) is not None,
        "label_pt_exists": (dataset_root / "processed" / "label.pt").exists(),
        "feature_pt_exists": (dataset_root / "processed" / "feature.pt").exists(),
        "protac_feature_npy_exists": (dataset_root / "features" / "protac_feature.npy").exists(),
        "target_feature_npy_exists": (dataset_root / "features" / "target_feature.npy").exists(),
        "e3_feature_npy_exists": (dataset_root / "features" / "e3_feature.npy").exists(),
        "processed_graph_pt_exists": bool(list((dataset_root / "processed").glob("*.pt"))),
    }
    for key, rel_path in {
        "protac_feature": dataset_root / "features" / "protac_feature.npy",
        "target_feature": dataset_root / "features" / "target_feature.npy",
        "e3_feature": dataset_root / "features" / "e3_feature.npy",
        "feature_pt": dataset_root / "processed" / "feature.pt",
        "label_pt": dataset_root / "processed" / "label.pt",
    }.items():
        if not rel_path.exists():
            continue
        if rel_path.suffix.lower() == ".npy":
            feature_shapes[key] = describe_array_payload(np.load(rel_path, allow_pickle=True))
        else:
            payload = torch.load(rel_path, map_location="cpu", weights_only=False)
            feature_shapes[key] = describe_array_payload(payload)
            if key == "label_pt":
                if isinstance(payload, torch.Tensor):
                    processed_label_count = int(payload.shape[0])
                elif hasattr(payload, "__len__"):
                    processed_label_count = int(len(payload))

    audit_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root),
        "zip_files": [
            {
                "name": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in zip_files
        ],
        "top_level_folders": inventory["top_level_folders"],
        "files_by_suffix": inventory["files_by_suffix"],
        "local_assets": local_assets,
        "processed_label_count": processed_label_count,
        "feature_shapes": feature_shapes,
        "official_name_json": official_summary,
        "notes": [
            "Zenodo asset extract contains no JSON metadata files.",
            "Official DegradeMaster exposes data/PROTAC/name.json and this file is used as canonical benchmark metadata.",
        ],
    }
    write_json(audit_report_path(config), audit_payload)

    note_lines = [
        f"- dataset_root: {dataset_root}",
        f"- zip_files: {[item['name'] for item in audit_payload['zip_files']]}",
        f"- top_level_folders: {audit_payload['top_level_folders']}",
        f"- local_json_count: {len(audit_payload['files_by_suffix'].get('.json', []))}",
        f"- official_name_json_source: {official_summary.get('source', '')}",
        f"- official_entry_count: {official_summary.get('entry_count', 0)}",
        f"- official_label_counts: {json.dumps(official_summary.get('label_counts', {}), sort_keys=True)}",
        "- Zenodo lacks JSON metadata while official DegradeMaster exposes data/PROTAC/name.json.",
    ]
    append_note(config, f"PROTAC-8K Audit {timestamp_now()}", note_lines)

    if check_leakage and official_path is not None:
        frame, metadata, _, split_report = build_protac_name_json_frame(
            config,
            dataset_root,
            official_payload,
            official_source,
            hydrate_sequences=False,
        )
        write_split_report(config, split_report)
        leakage_payload = build_leakage_report(frame, metadata["split_policy"])
        write_leakage_report(config, leakage_payload)

    return audit_payload


def init_command(config: Dict[str, Any], _: argparse.Namespace) -> None:
    ensure_project_dirs(config)
    changelog = changelog_path(config)
    if not changelog.exists():
        changelog.write_text("# DegradeQuery Changelog\n", encoding="utf-8")
    note_file = notes_path(config)
    if not note_file.exists():
        note_file.write_text("# DegradeQuery Codex Notes\n", encoding="utf-8")
    print(f"initialized_root: {runtime_root(config)}")


def snapshot_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    ensure_project_dirs(config)
    root = runtime_root(config)
    out_dir = output_dir(config)
    previous = out_dir / "source_previous.tar.gz"
    last_good = out_dir / "source_last_good.tar.gz"
    if last_good.exists():
        if previous.exists():
            previous.unlink()
        last_good.replace(previous)
    with tarfile.open(last_good, "w:gz") as archive:
        for rel_path in SOURCE_SNAPSHOT_FILES:
            path = root / rel_path
            if path.exists():
                archive.add(path, arcname=rel_path)
    append_note(
        config,
        f"Snapshot {timestamp_now()}",
        [
            f"- tag: {safe_text(args.tag) or 'none'}",
            f"- archive: {last_good}",
            f"- source_sha256: {compute_source_hash(config)}",
            f"- source_bundle_sha256: {compute_source_bundle_sha256(config)}",
        ],
    )
    print(f"snapshot: {last_good}")


def fetch_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    ensure_project_dirs(config)
    if args.dataset != "protac8k":
        raise ValueError("fetch currently supports only --dataset protac8k")
    record_id = config["data"]["protac8k_record_id"]
    dataset_root = protac8k_root(config)
    dataset_root.mkdir(parents=True, exist_ok=True)

    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    payload = response.json()
    files = payload.get("files", [])
    if not files:
        raise RuntimeError("Zenodo record returned no files")

    note_lines = [f"- record_id: {record_id}", f"- title: {payload.get('metadata', {}).get('title', '')}"]
    for file_entry in files:
        key = file_entry["key"]
        zip_path = dataset_root / key
        if args.force or not zip_path.exists():
            download_with_progress(file_entry["links"]["self"], zip_path, f"download {key}")
        extracted = 0
        if zip_path.suffix.lower() == ".zip":
            extracted = extract_protac_zip(zip_path, dataset_root, args.force)
        note_lines.extend(
            [
                f"- file: {key}",
                f"  - size: {file_entry.get('size')}",
                f"  - checksum: {file_entry.get('checksum')}",
                f"  - local_sha256: {file_sha256(zip_path)}",
                f"  - extracted_files: {extracted}",
            ]
        )
    append_note(config, f"PROTAC-8K Fetch {timestamp_now()}", note_lines)
    print(f"fetched_dataset_root: {dataset_root}")


def audit_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.dataset != "protac8k":
        raise ValueError("audit currently supports only --dataset protac8k")
    payload = audit_protac8k(config, check_leakage=bool(args.check_leakage))
    print(f"audit: {audit_report_path(config)}")
    if args.check_leakage:
        print(f"leakage_report: {leakage_report_path(config)}")
    print(f"official_entry_count: {payload.get('official_name_json', {}).get('entry_count', 0)}")


def prep_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    ensure_project_dirs(config)
    dataset = args.dataset
    if dataset == "protac8k":
        dataset_root = protac8k_root(config)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")
        scanned = list_dataset_files(dataset_root)
        note_lines = [f"- dataset_root: {dataset_root}", f"- scanned_files: {len(scanned)}"]
        for path in scanned:
            note_lines.append(f"- file: {path.relative_to(runtime_root(config))}")
        frame, metadata, dataset_cache, split_report = build_protac8k_from_official_metadata(
            config,
            dataset_root,
            hydrate_sequences=True,
        )
        leakage_payload = build_leakage_report(frame, metadata["split_policy"])
        normalized_path = normalized_dataset_path(config, "protac8k")
        write_normalized_dataset(frame, normalized_path)
        write_split_report(config, split_report)
        write_leakage_report(config, leakage_payload)
        cache = load_feature_cache(config)
        cache.setdefault("datasets", {})
        cache["datasets"]["protac8k"] = dataset_cache
        save_feature_cache(config, cache)
        stats = summarize_standard_frame(frame, metadata["split_policy"])
        pretty_print_stats(stats)
        if not bool(frame["benchmark_ready"].all()):
            print("WARNING: dataset is not benchmark-ready. This path is smoke-test only.")
        append_note(
            config,
            f"PROTAC-8K Prep {timestamp_now()}",
            note_lines
            + [f"- metadata: {json.dumps(metadata, sort_keys=True, default=str)}"]
            + [f"- split_report: {json.dumps(split_report, sort_keys=True, default=str)}"]
            + [f"- leakage_report: {json.dumps(leakage_payload, sort_keys=True, default=str)}"]
            + [f"- stats: {json.dumps(stats, sort_keys=True, default=str)}"],
        )
        print(f"normalized_path: {normalized_path}")
        print(f"split_report: {split_report_path(config)}")
        print(f"leakage_report: {leakage_report_path(config)}")
        return

    if dataset not in {"molglue", "mgtbind"}:
        raise ValueError(f"Unsupported dataset for prep: {dataset}")
    if not args.input:
        raise ValueError(f"--input is required for dataset {dataset}")
    input_path = resolve_path(config, args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    raw = read_dataframe_from_path(input_path)
    frame, metadata = normalize_external_dataset(raw, dataset, input_path.name, config)
    normalized_path = normalized_dataset_path(config, dataset)
    write_normalized_dataset(frame, normalized_path)
    stats = summarize_standard_frame(frame, metadata["split_policy"])
    pretty_print_stats(stats)
    append_note(
        config,
        f"{dataset} Prep {timestamp_now()}",
        [
            f"- input: {input_path}",
            f"- rows_raw: {len(raw)}",
            f"- metadata: {json.dumps(metadata, sort_keys=True, default=str)}",
            f"- stats: {json.dumps(stats, sort_keys=True, default=str)}",
        ],
    )
    print(f"normalized_path: {normalized_path}")
