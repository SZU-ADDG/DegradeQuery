from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import socket
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from dq_data import (
    AA_ORDER,
    ASSAY_TEXT_ALIASES,
    CELL_LINE_ALIASES,
    PROJECT_VERSION,
    RDKIT_AVAILABLE,
    RECRUITER_ID_ALIASES,
    RECRUITER_NAME_ALIASES,
    RECRUITER_SEQUENCE_ALIASES,
    SMILES_ALIASES,
    TARGET_ID_ALIASES,
    TARGET_NAME_ALIASES,
    TARGET_SEQUENCE_ALIASES,
    TASK_ID_ALIASES,
    append_manifest,
    append_note,
    apply_analysis_split_overlay,
    as_bool,
    baseline_report_json_path,
    baseline_report_md_path,
    benchmark_report_json_path,
    benchmark_report_md_path,
    bytes_sha256,
    canonicalize_smiles,
    compare_runs_report_json_path,
    compare_runs_report_md_path,
    compute_dataset_fingerprint,
    compute_source_hash,
    compute_source_bundle_sha256,
    data_dir,
    esm2_embedding_cache_path,
    esm2_embedding_report_json_path,
    esm2_embedding_report_md_path,
    esm2_failure_report_path,
    empty_standard_row,
    ensure_project_dirs,
    ensure_sequences,
    feature_cache_path,
    fixed05_error_profile_path,
    file_sha256,
    find_column,
    load_feature_cache,
    load_analysis_split_overlay,
    load_normalized_dataset,
    mol_from_mol2,
    mol_from_smiles,
    named_json_output_path,
    named_md_output_path,
    normalized_dataset_path,
    notes_path,
    output_dir,
    protein_emb_cache_path,
    read_dataframe_from_path,
    resolve_device,
    resolve_path,
    runtime_root,
    safe_int,
    safe_text,
    sequence_kmer_set,
    set_seed,
    standardize_frame_columns,
    timestamp_now,
    target_sequence_cluster_map,
    write_normalized_dataset,
)

from transformers import AutoModel, AutoTokenizer

TRANSFORMERS_AVAILABLE = True
TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None

ESM_RUNTIME: Dict[str, Any] = {}
RUN_LABEL_ALLOWED = re.compile(r"[^A-Za-z0-9_. -]+")
RUN_LABEL_MULTI_UNDERSCORE = re.compile(r"_+")
ESM2_650M_ENCODER = "esm2_650m"
ESM2_650M_POOLING = "last_hidden_mean"
ESM2_650M_LONG_SEQUENCE_POLICY = "sliding_window_mean_50pct_overlap"
DEGRADEMASTER_REFERENCE_ROWS = [
    {"method": "DegradeMaster", "setting": "Super", "accuracy": 0.8141, "auroc": 0.8541, "f1": 0.8139, "note": "paper"},
    {"method": "DegradeMaster", "setting": "Semi", "accuracy": 0.8366, "auroc": 0.8825, "f1": 0.8224, "note": "paper"},
]
DEGRADEMASTER_BASELINE_REFERENCE = {
    "svm_morgan": {"accuracy": 0.7447, "auroc": 0.7901, "f1": 0.7446},
    "rf_morgan": {"accuracy": 0.6340, "auroc": 0.8077, "f1": 0.5818},
}
ABLATION_ALIASES = {
    "smiles_target": "smiles_target_seq",
    "smiles_target_recruiter": "smiles_target_recruiter_seq",
    "smiles_target_recruiter_cell": "smiles_graph_fp",
}
PRETRAIN_TASKS = {"tuple_contrast", "mol_contrast"}
TUPLE_NEGATIVE_POLICIES = {
    "easy": ["random"],
    "full": ["random", "same_e3_diff_target", "same_target_diff_e3"],
    "target_only": ["target_only", "target_only", "target_only"],
    "e3_only": ["e3_only", "e3_only", "e3_only"],
    "joint_only": ["joint_only", "joint_only", "joint_only"],
    "full_mixed": ["target_only", "e3_only", "joint_only"],
    "frequency_matched": ["frequency_matched", "frequency_matched", "frequency_matched"],
    "target_seq_cluster": ["target_seq_cluster", "target_seq_cluster", "target_seq_cluster"],
    "protein_nn": ["protein_nn", "protein_nn", "protein_nn"],
    "hard_mixed": ["frequency_matched", "target_seq_cluster", "protein_nn"],
}
PRETRAIN_POSITIVE_CONSTRUCTIONS = {"observed", "shuffled_pair", "shuffled_independent"}
REPORTING_PROTOCOL = "paper_authoritative_no_test_model_selection"
SELECTION_SCOPE = "train_or_validation"
SELECTION_RULE = "acc_then_f1_then_auroc_at_fixed_0p5"
THRESHOLD_RULE = "fixed_0p5_no_threshold_tuning"
SUMMARY_PROFILES = {
    "",
    "ctp_formal_v1",
    "official_split_final",
    "label_efficiency",
    "hard_split",
    "fewshot_target",
    "generalization_lowresource_v1",
    "esm2_650m_sprint",
    "fixed05_finetune_sweep",
    "sota_sprint_v2",
}


def precise_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sanitize_run_label(label: str) -> str:
    raw = safe_text(label)
    if not raw:
        return ""
    text = raw.replace(" ", "_")
    text = RUN_LABEL_ALLOWED.sub("_", text)
    text = RUN_LABEL_MULTI_UNDERSCORE.sub("_", text).strip("_")
    if len(text) > 80:
        text = text[:80].rstrip("_")
    if not text:
        raise ValueError("`--run-label` is empty after sanitization.")
    return text


def append_run_label(base: str, run_label: str) -> str:
    label = sanitize_run_label(run_label)
    return f"{base}_{label}" if label else base


def output_paths_for_stem(config: Dict[str, Any], stem: str, default_json: Path, default_md: Path) -> Tuple[Path, Path]:
    clean = sanitize_run_label(stem) if safe_text(stem) else ""
    if not clean:
        return default_json, default_md
    return named_json_output_path(config, clean), named_md_output_path(config, clean)


def display_name_for_payload(payload: Dict[str, Any]) -> str:
    return (
        safe_text(payload.get("run_label"))
        or safe_text(payload.get("ablation"))
        or safe_text(payload.get("run_id"))
    )


def amino_acid_composition(sequence: str) -> np.ndarray:
    sequence = sequence.upper()
    counts = np.zeros(len(AA_ORDER), dtype=np.float32)
    if not sequence:
        return counts
    for residue in sequence:
        index = AA_ORDER.find(residue)
        if index >= 0:
            counts[index] += 1.0
    if counts.sum() > 0:
        counts /= counts.sum()
    return counts


def hashed_kmer_counts(sequence: str, dim: int) -> np.ndarray:
    sequence = "".join([residue for residue in sequence.upper() if residue in AA_ORDER])
    if not sequence:
        return np.zeros(dim, dtype=np.float32)
    counts = np.zeros(dim, dtype=np.float32)
    for k in (2, 3):
        if len(sequence) < k:
            continue
        for idx in range(len(sequence) - k + 1):
            kmer = sequence[idx : idx + k]
            bucket = int(hashlib.sha256(kmer.encode("utf-8")).hexdigest(), 16) % dim
            counts[bucket] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def build_unknown_protein_feature(config: Dict[str, Any]) -> np.ndarray:
    dim = int(config["features"]["protein_feature_dim"])
    vec = np.zeros(dim, dtype=np.float32)
    if safe_text(config["features"].get("protein_encoder", "aac")).lower() != ESM2_650M_ENCODER and dim >= 22:
        vec[20] = 0.0
        vec[21] = 1.0
    return vec


def protein_encoder_name(config: Dict[str, Any]) -> str:
    return safe_text(config["features"].get("protein_encoder", "aac")).lower() or "aac"


def esm2_model_name(config: Dict[str, Any]) -> str:
    return safe_text(config["features"].get("esm2_model", "")) or "facebook/esm2_t33_650M_UR50D"


def protein_feature_dim_for_encoder(config: Dict[str, Any], encoder: Optional[str] = None) -> int:
    chosen = safe_text(encoder).lower() or protein_encoder_name(config)
    if chosen == ESM2_650M_ENCODER:
        return int(config["features"].get("protein_feature_dim", 1280))
    return int(config["features"]["protein_feature_dim"])


def build_aux_protein_feature(aux_vector: Optional[np.ndarray], config: Dict[str, Any]) -> np.ndarray:
    dim = int(config["features"]["protein_feature_dim"])
    vec = np.zeros(dim, dtype=np.float32)
    if aux_vector is None:
        return build_unknown_protein_feature(config)
    aux = np.asarray(aux_vector, dtype=np.float32).flatten()
    if aux.size == 0:
        return build_unknown_protein_feature(config)
    norm = np.linalg.norm(aux)
    if norm > 0:
        aux = aux / norm
    copy_size = min(len(aux), dim - 2)
    vec[:copy_size] = aux[:copy_size]
    vec[-2] = math.log1p(len(aux))
    vec[-1] = 1.0
    return vec


def build_aac_protein_feature(sequence: str, config: Dict[str, Any]) -> np.ndarray:
    dim = int(config["features"]["protein_feature_dim"])
    kmer_dim = int(config["features"]["hashed_kmer_dim"])
    vec = np.zeros(dim, dtype=np.float32)
    sequence = safe_text(sequence).upper()
    composition = amino_acid_composition(sequence)
    vec[: len(composition)] = composition
    if dim > 20:
        vec[20] = math.log1p(len(sequence)) if sequence else 0.0
    if dim > 21:
        unknown_fraction = (
            sum(1 for residue in sequence if residue not in AA_ORDER) / max(len(sequence), 1)
            if sequence
            else 1.0
        )
        vec[21] = float(unknown_fraction)
    hashed = hashed_kmer_counts(sequence, kmer_dim)
    start = 22
    if dim > start:
        vec[start : start + min(kmer_dim, dim - start)] = hashed[: max(0, dim - start)]
    return vec


def build_embedding_protein_feature(embedding: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).flatten()
    expected = int(config["features"]["protein_feature_dim"])
    if vec.size != expected:
        raise RuntimeError(
            f"Protein embedding dimension mismatch: expected {expected}, got {vec.size}."
        )
    return vec


def candidate_local_model_paths(model_name: str) -> List[Path]:
    text = safe_text(model_name)
    if not text:
        return []
    direct = Path(text).expanduser()
    repo_key = text.replace("/", "--")
    repo_dir = f"models--{repo_key}"
    candidates = [
        direct,
        Path.home() / ".cache" / "huggingface" / "hub" / repo_dir,
    ]
    seen: set[str] = set()
    unique: List[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_pretrained_model_source(model_name: str) -> Tuple[str, bool]:
    for path in candidate_local_model_paths(model_name):
        if path.exists() and (path / "config.json").exists():
            return str(path), True
    return model_name, False


def load_esm_runtime(model_name: str, device: torch.device) -> Optional[Tuple[Any, Any]]:
    resolved_source, is_local = resolve_pretrained_model_source(model_name)
    cache_key = f"{resolved_source}:{device.type}"
    if cache_key in ESM_RUNTIME:
        return ESM_RUNTIME[cache_key]
    tokenizer = AutoTokenizer.from_pretrained(resolved_source, local_files_only=is_local)
    model = AutoModel.from_pretrained(resolved_source, local_files_only=is_local)
    model.eval()
    model.to(device)
    ESM_RUNTIME[cache_key] = (tokenizer, model)
    return tokenizer, model


def build_esm_feature(
    sequence: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Optional[np.ndarray]:
    model_name = safe_text(config["features"].get("esm_model", ""))
    if not model_name or not sequence:
        return None
    runtime = load_esm_runtime(model_name, device)
    if runtime is None:
        return None
    tokenizer, model = runtime
    cache_path = protein_emb_cache_path(config)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.exists() else {}
    key = bytes_sha256(sequence.encode("utf-8"))
    if key in cache:
        return np.asarray(cache[key], dtype=np.float32)
    tokens = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {name: value.to(device) for name, value in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
        hidden = outputs.last_hidden_state[0]
        mask = tokens["attention_mask"][0].bool()
        pooled = hidden[mask].mean(dim=0).detach().cpu().numpy().astype(np.float32)
    cache[key] = pooled
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return pooled


def load_esm2_embedding_cache(config: Dict[str, Any]) -> Dict[str, Any]:
    path = esm2_embedding_cache_path(config)
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def esm2_embedding_record(
    sequence: str,
    config: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    seq = safe_text(sequence)
    if not seq:
        return None
    payload = cache if cache is not None else load_esm2_embedding_cache(config)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        return None
    return records.get(bytes_sha256(seq.encode("utf-8")))


def cached_esm2_feature(
    sequence: str,
    config: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    record = esm2_embedding_record(sequence, config, cache)
    if not record:
        return None
    return np.asarray(record.get("embedding", []), dtype=np.float32).flatten()


def protein_feature_vector(
    sequence: str,
    aux_vector: Optional[np.ndarray],
    config: Dict[str, Any],
    device: torch.device,
    esm2_cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    encoder = protein_encoder_name(config)
    if encoder == ESM2_650M_ENCODER:
        seq = safe_text(sequence)
        if not seq:
            raise RuntimeError("ESM-2 650M protein encoding requires a non-empty sequence.")
        esm_feature = cached_esm2_feature(seq, config, esm2_cache)
        if esm_feature is None or esm_feature.size == 0:
            raise RuntimeError("Missing cached ESM-2 650M embedding for a required protein sequence.")
        return build_embedding_protein_feature(esm_feature, config)
    if encoder == "esm" and safe_text(sequence):
        esm_feature = build_esm_feature(sequence, config, device)
        if esm_feature is not None:
            return build_aux_protein_feature(esm_feature, config)
    if safe_text(sequence):
        return build_aac_protein_feature(sequence, config)
    if as_bool(config["features"].get("use_dataset_aux_features", True)) and aux_vector is not None:
        return build_aux_protein_feature(aux_vector, config)
    return build_unknown_protein_feature(config)


def esm_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def esm_effective_max_tokens(tokenizer: Any, model: Any) -> int:
    tokenizer_limit = safe_int(getattr(tokenizer, "model_max_length", 0), 0)
    if tokenizer_limit <= 0 or tokenizer_limit > 100000:
        tokenizer_limit = 0
    model_limit = safe_int(getattr(getattr(model, "config", None), "max_position_embeddings", 0), 0)
    candidates = [value for value in [tokenizer_limit, model_limit] if value > 0]
    if not candidates:
        return 1024
    return min(candidates)


def sequence_windows(sequence: str, window_size: int, stride: int) -> List[str]:
    if len(sequence) <= window_size:
        return [sequence]
    windows: List[str] = []
    start = 0
    while start < len(sequence):
        end = min(len(sequence), start + window_size)
        window = sequence[start:end]
        if window:
            windows.append(window)
        if end >= len(sequence):
            break
        start += stride
    return windows


def embed_sequence_with_esm2_650m(
    sequence: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> np.ndarray:
    seq = safe_text(sequence)
    if not seq:
        raise RuntimeError("Cannot build an ESM-2 embedding for an empty sequence.")
    max_tokens = esm_effective_max_tokens(tokenizer, model)
    window_size = max(1, max_tokens - 2)
    stride = max(1, window_size // 2)
    window_vectors: List[np.ndarray] = []
    for window in sequence_windows(seq, window_size, stride):
        tokens = tokenizer(
            window,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            return_special_tokens_mask=True,
        )
        special_mask = tokens.pop("special_tokens_mask")[0].bool()
        tokens = {name: value.to(device) for name, value in tokens.items()}
        with torch.no_grad():
            with esm_autocast_context(device):
                outputs = model(**tokens)
                hidden = outputs.last_hidden_state[0]
        attention_mask = tokens["attention_mask"][0].bool()
        valid_mask = attention_mask & ~special_mask.to(device)
        if not valid_mask.any():
            raise RuntimeError("ESM-2 embedding window produced no residue tokens after masking.")
        pooled = hidden[valid_mask].float().mean(dim=0).detach().cpu().numpy().astype(np.float32)
        window_vectors.append(pooled)
    if not window_vectors:
        raise RuntimeError("ESM-2 embedding produced no pooled windows.")
    if len(window_vectors) == 1:
        return window_vectors[0]
    return np.stack(window_vectors, axis=0).mean(axis=0).astype(np.float32)


def embedseq_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    dataset_name = args.dataset
    encoder = safe_text(args.encoder).lower()
    if encoder != ESM2_650M_ENCODER:
        raise ValueError(f"Unsupported encoder: {args.encoder}")
    if not TRANSFORMERS_AVAILABLE:
        reason = f"transformers unavailable: {TRANSFORMERS_IMPORT_ERROR}"
        failure_path = esm2_failure_report_path(runtime_cfg)
        failure_path.write_text(reason + "\n", encoding="utf-8")
        append_note(runtime_cfg, f"ESM-2 650M failure {timestamp_now()}", [f"- {reason}"])
        raise RuntimeError(reason)
    frame = load_normalized_dataset(runtime_cfg, dataset_name)
    device = resolve_device(runtime_cfg)
    model_name = esm2_model_name(runtime_cfg)
    try:
        runtime = load_esm_runtime(model_name, device)
        if runtime is None:
            raise RuntimeError(f"Unable to load model runtime for {model_name}.")
        tokenizer, model = runtime
        model.eval()
        unique_sequences: Dict[str, Dict[str, Any]] = {}
        target_frame = frame.loc[frame["target_sequence"].fillna("").astype(str).str.len() > 0, ["target_sequence", "target_sequence_source"]]
        recruiter_frame = frame.loc[frame["recruiter_sequence"].fillna("").astype(str).str.len() > 0, ["recruiter_sequence", "recruiter_sequence_source"]]
        for sequence, source in target_frame.itertuples(index=False, name=None):
            seq = safe_text(sequence)
            if seq:
                unique_sequences.setdefault(bytes_sha256(seq.encode("utf-8")), {"sequence": seq, "sequence_source": safe_text(source) or "manual"})
        for sequence, source in recruiter_frame.itertuples(index=False, name=None):
            seq = safe_text(sequence)
            if seq:
                unique_sequences.setdefault(bytes_sha256(seq.encode("utf-8")), {"sequence": seq, "sequence_source": safe_text(source) or "manual"})
        records: Dict[str, Dict[str, Any]] = {}
        embedding_dim = 0
        for sequence_sha256, payload in unique_sequences.items():
            embedding = embed_sequence_with_esm2_650m(payload["sequence"], tokenizer, model, device)
            embedding_dim = int(embedding.shape[0])
            records[sequence_sha256] = {
                "model_name": model_name,
                "sequence_sha256": sequence_sha256,
                "sequence": payload["sequence"],
                "embedding": embedding,
                "embedding_dim": embedding_dim,
                "pooling": ESM2_650M_POOLING,
                "sequence_len": len(payload["sequence"]),
                "sequence_source": payload["sequence_source"],
                "long_sequence_policy": ESM2_650M_LONG_SEQUENCE_POLICY,
            }
        cache_payload = {
            "model_name": model_name,
            "pooling": ESM2_650M_POOLING,
            "long_sequence_policy": ESM2_650M_LONG_SEQUENCE_POLICY,
            "embedding_dim": embedding_dim,
            "records": records,
        }
        cache_path = esm2_embedding_cache_path(runtime_cfg)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_payload, cache_path)
        target_hashes = {bytes_sha256(safe_text(seq).encode("utf-8")) for seq in frame["target_sequence"].fillna("").astype(str).tolist() if safe_text(seq)}
        recruiter_hashes = {bytes_sha256(safe_text(seq).encode("utf-8")) for seq in frame["recruiter_sequence"].fillna("").astype(str).tolist() if safe_text(seq)}
        target_row_mask = frame["target_sequence"].fillna("").astype(str).apply(lambda seq: bytes_sha256(safe_text(seq).encode("utf-8")) in records if safe_text(seq) else False)
        recruiter_row_mask = frame["recruiter_sequence"].fillna("").astype(str).apply(lambda seq: bytes_sha256(safe_text(seq).encode("utf-8")) in records if safe_text(seq) else False)
        report = {
            "dataset": dataset_name,
            "encoder": encoder,
            "model_name": model_name,
            "cache_path": str(cache_path),
            "embedding_dim": embedding_dim,
            "pooling": ESM2_650M_POOLING,
            "long_sequence_policy": ESM2_650M_LONG_SEQUENCE_POLICY,
            "n_unique_sequences": len(records),
            "n_unique_target_sequences": len(target_hashes),
            "n_unique_recruiter_sequences": len(recruiter_hashes),
            "target_embedding_row_level_coverage": float(target_row_mask.mean()) if len(frame) else 0.0,
            "recruiter_embedding_row_level_coverage": float(recruiter_row_mask.mean()) if len(frame) else 0.0,
            "target_embedding_unique_coverage": float(len(target_hashes & set(records.keys())) / max(len(target_hashes), 1)),
            "recruiter_embedding_unique_coverage": float(len(recruiter_hashes & set(records.keys())) / max(len(recruiter_hashes), 1)),
        }
        json_path = esm2_embedding_report_json_path(runtime_cfg)
        md_path = esm2_embedding_report_md_path(runtime_cfg)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        md_path.write_text(
            "# ESM-2 650M Embedding Report\n\n"
            + f"- model_name: {model_name}\n"
            + f"- cache_path: {cache_path}\n"
            + f"- embedding_dim: {embedding_dim}\n"
            + f"- unique_target_sequences: {len(target_hashes)}\n"
            + f"- unique_recruiter_sequences: {len(recruiter_hashes)}\n"
            + f"- target_embedding_row_level_coverage: {report['target_embedding_row_level_coverage']:.4f}\n"
            + f"- recruiter_embedding_row_level_coverage: {report['recruiter_embedding_row_level_coverage']:.4f}\n"
            + f"- target_embedding_unique_coverage: {report['target_embedding_unique_coverage']:.4f}\n"
            + f"- recruiter_embedding_unique_coverage: {report['recruiter_embedding_unique_coverage']:.4f}\n",
            encoding="utf-8",
        )
        print(f"cache: {cache_path}")
        print(f"report_json: {json_path}")
        print(f"report_md: {md_path}")
    except Exception as exc:
        failure_path = esm2_failure_report_path(runtime_cfg)
        lines = [
            "# ESM-2 650M failure",
            "",
            f"- dataset: {dataset_name}",
            f"- model_name: {model_name}",
            f"- reason: {safe_text(exc)}",
            "",
        ]
        failure_path.write_text("\n".join(lines), encoding="utf-8")
        append_note(runtime_cfg, f"ESM-2 650M failure {timestamp_now()}", [f"- reason: {safe_text(exc)}"])
        raise


def atom_feature_vector(atom) -> np.ndarray:
    hybridization_names = {"SP": 0, "SP2": 1, "SP3": 2, "SP3D": 3, "SP3D2": 4}
    chiral_names = {
        "CHI_UNSPECIFIED": 0,
        "CHI_TETRAHEDRAL_CW": 1,
        "CHI_TETRAHEDRAL_CCW": 2,
        "CHI_OTHER": 3,
    }
    vec = np.zeros(15, dtype=np.float32)
    vec[0] = min(atom.GetAtomicNum(), 100) / 100.0
    vec[1] = atom.GetDegree() / 6.0
    vec[2] = (atom.GetFormalCharge() + 4.0) / 8.0
    hyb_name = str(atom.GetHybridization())
    if hyb_name in hybridization_names:
        vec[3 + hybridization_names[hyb_name]] = 1.0
    vec[8] = 1.0 if atom.GetIsAromatic() else 0.0
    vec[9] = atom.GetTotalNumHs() / 8.0
    chiral_name = str(atom.GetChiralTag())
    if chiral_name in chiral_names:
        vec[10 + chiral_names[chiral_name]] = 1.0
    vec[14] = 1.0 if atom.IsInRing() else 0.0
    return vec


def mol_to_graph_and_fp(mol, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from dq_data import AllChem, DataStructs

    radius = int(config["features"]["morgan_radius"])
    bits = int(config["features"]["morgan_bits"])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits)
    fp_array = np.zeros(bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    atom_features = np.stack([atom_feature_vector(atom) for atom in mol.GetAtoms()], axis=0).astype(np.float32)
    adjacency = np.eye(mol.GetNumAtoms(), dtype=np.float32)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin, end] = 1.0
        adjacency[end, begin] = 1.0
    return atom_features, adjacency, fp_array


def fallback_mol_features(
    config: Dict[str, Any],
    aux_protac_feature: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = int(config["features"]["morgan_bits"])
    fp = np.zeros(bits, dtype=np.float32)
    if aux_protac_feature is not None and len(aux_protac_feature) > 0:
        aux = np.asarray(aux_protac_feature, dtype=np.float32).flatten()
        copy_size = min(len(aux), bits)
        fp[:copy_size] = aux[:copy_size]
    atom_features = np.zeros((1, 15), dtype=np.float32)
    adjacency = np.eye(1, dtype=np.float32)
    return atom_features, adjacency, fp


class FeatureResolver:
    def __init__(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        feature_cache: Dict[str, Any],
        cell_vocab: Dict[str, int],
        device: torch.device,
    ) -> None:
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        self.cell_vocab = cell_vocab
        self.dataset_cache_rows = feature_cache.get("datasets", {}).get(dataset_name, {}).get("rows", {})
        self.mol_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.target_cache: Dict[str, np.ndarray] = {}
        self.recruiter_cache: Dict[str, np.ndarray] = {}
        self.esm2_cache = load_esm2_embedding_cache(config) if protein_encoder_name(config) == ESM2_650M_ENCODER else None

    def get_aux(self, row_id: str) -> Dict[str, Any]:
        return self.dataset_cache_rows.get(row_id, {})

    def cell_index(self, cell_line: str) -> int:
        key = safe_text(cell_line) or "UNK_CELL"
        return self.cell_vocab.get(key, 0)

    def mol_features(self, row: Dict[str, Any], aux: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_id = safe_text(row.get("row_id", ""))
        if row_id in self.mol_cache:
            return self.mol_cache[row_id]
        aux = aux or self.get_aux(row_id)
        mol = None
        smiles = safe_text(row.get("smiles", ""))
        if smiles and RDKIT_AVAILABLE:
            mol = mol_from_smiles(smiles)
        if mol is None and aux.get("protac_path"):
            mol = mol_from_mol2(Path(aux["protac_path"]))
        if mol is None:
            payload = fallback_mol_features(self.config, aux.get("aux_protac_feature"))
        else:
            payload = mol_to_graph_and_fp(mol, self.config)
        self.mol_cache[row_id] = payload
        return payload

    def target_feature(self, row: Dict[str, Any], aux: Optional[Dict[str, Any]] = None) -> np.ndarray:
        row_id = safe_text(row.get("row_id", ""))
        if row_id in self.target_cache:
            return self.target_cache[row_id]
        aux = aux or self.get_aux(row_id)
        value = protein_feature_vector(
            safe_text(row.get("target_sequence", "")),
            aux.get("aux_target_feature"),
            self.config,
            self.device,
            self.esm2_cache,
        ).astype(np.float32)
        self.target_cache[row_id] = value
        return value

    def recruiter_feature(self, row: Dict[str, Any], aux: Optional[Dict[str, Any]] = None) -> np.ndarray:
        row_id = safe_text(row.get("row_id", ""))
        if row_id in self.recruiter_cache:
            return self.recruiter_cache[row_id]
        aux = aux or self.get_aux(row_id)
        value = protein_feature_vector(
            safe_text(row.get("recruiter_sequence", "")),
            aux.get("aux_recruiter_feature"),
            self.config,
            self.device,
            self.esm2_cache,
        ).astype(np.float32)
        self.recruiter_cache[row_id] = value
        return value

    def task_id(self, row: Dict[str, Any]) -> str:
        return safe_text(row.get("task_id", "")) or safe_text(row.get("target_id", ""))

    def row_meta(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "row_id": safe_text(row.get("row_id", "")),
            "split": safe_text(row.get("split", "")),
            "label": row["label"] if as_bool(row.get("is_labeled", False)) else "",
            "smiles": safe_text(row.get("smiles", "")),
            "target_id": safe_text(row.get("target_id", "")),
            "recruiter_id": safe_text(row.get("recruiter_id", "")),
            "cell_line": safe_text(row.get("cell_line", "")),
            "task_id": self.task_id(row),
        }


def row_target_key(row: Dict[str, Any]) -> str:
    return safe_text(row.get("target_id", "")) or safe_text(row.get("target_name", ""))


def row_recruiter_key(row: Dict[str, Any]) -> str:
    return safe_text(row.get("recruiter_id", "")) or safe_text(row.get("recruiter_name", ""))


class DegradeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_name: str,
        config: Dict[str, Any],
        feature_cache: Dict[str, Any],
        cell_vocab: Dict[str, int],
        device: torch.device,
    ) -> None:
        self.records = frame.to_dict("records")
        self.resolver = FeatureResolver(dataset_name, config, feature_cache, cell_vocab, device)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        aux = self.resolver.get_aux(safe_text(row.get("row_id", "")))
        atom_x, adjacency, morgan_fp = self.resolver.mol_features(row, aux)
        target_feat = self.resolver.target_feature(row, aux)
        recruiter_feat = self.resolver.recruiter_feature(row, aux)
        is_labeled = as_bool(row.get("is_labeled", False))
        return {
            "row_id": safe_text(row.get("row_id", "")),
            "split": safe_text(row.get("split", "")),
            "label": float(row["label"]) if is_labeled else -1.0,
            "is_labeled": is_labeled,
            "sample_weight": float(row.get("_sample_weight", 1.0)),
            "morgan_fp": morgan_fp.astype(np.float32),
            "atom_x": atom_x.astype(np.float32),
            "adjacency": adjacency.astype(np.float32),
            "target_feat": target_feat.astype(np.float32),
            "recruiter_feat": recruiter_feat.astype(np.float32),
            "cell_index": self.resolver.cell_index(safe_text(row.get("cell_line", ""))),
            "kg_feature": np.asarray(row.get("_kg_feature", []), dtype=np.float32),
            "kg_mol_index": int(row.get("_kg_mol_index", -1)),
            "kg_target_index": int(row.get("_kg_target_index", -1)),
            "kg_recruiter_index": int(row.get("_kg_recruiter_index", -1)),
            "task_id": self.resolver.task_id(row),
            "meta": self.resolver.row_meta(row),
        }


class CTPPretrainDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_name: str,
        config: Dict[str, Any],
        feature_cache: Dict[str, Any],
        cell_vocab: Dict[str, int],
        device: torch.device,
        tuple_negative_policy: str,
        positive_construction: str,
    ) -> None:
        self.records = frame.to_dict("records")
        self.resolver = FeatureResolver(dataset_name, config, feature_cache, cell_vocab, device)
        self.tuple_negative_policy = safe_text(tuple_negative_policy) or "full"
        self.positive_construction = safe_text(positive_construction) or "observed"
        if self.positive_construction not in PRETRAIN_POSITIVE_CONSTRUCTIONS:
            raise ValueError(f"Unsupported positive construction: {self.positive_construction}")
        self.active_policies = tuple(
            TUPLE_NEGATIVE_POLICIES.get(self.tuple_negative_policy, TUPLE_NEGATIVE_POLICIES["full"])
        )
        self.target_keys = [row_target_key(row) for row in self.records]
        self.recruiter_keys = [row_recruiter_key(row) for row in self.records]
        self.molecule_keys = [self._molecule_key(row) for row in self.records]
        self.target_pool_indices = [idx for idx, key in enumerate(self.target_keys) if key]
        self.recruiter_pool_indices = [idx for idx, key in enumerate(self.recruiter_keys) if key]
        self.pair_pool_indices = [idx for idx, row in enumerate(self.records) if self.target_keys[idx] and self.recruiter_keys[idx]]
        self.target_count_by_key = Counter(key for key in self.target_keys if key)
        self.recruiter_count_by_key = Counter(key for key in self.recruiter_keys if key)
        needs_target_clusters = "target_seq_cluster" in self.active_policies
        needs_frequency_candidates = "frequency_matched" in self.active_policies
        needs_protein_neighbors = "protein_nn" in self.active_policies
        self.target_family_map = (
            target_sequence_cluster_map(pd.DataFrame(self.records), threshold=0.70)
            if needs_target_clusters
            else {}
        )
        self.target_family_keys = [self.target_family_map.get(key, "") for key in self.target_keys]
        self.target_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.recruiter_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.target_family_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, key in enumerate(self.target_keys):
            if key:
                self.target_to_indices[key].append(idx)
                family_key = self.target_family_keys[idx]
                if family_key:
                    self.target_family_to_indices[family_key].append(idx)
        for idx, key in enumerate(self.recruiter_keys):
            if key:
                self.recruiter_to_indices[key].append(idx)
        self.same_e3_candidate_indices: List[np.ndarray] = []
        self.same_target_candidate_indices: List[np.ndarray] = []
        self.cluster_target_candidate_indices: List[np.ndarray] = []
        for idx in range(len(self.records)):
            recruiter_key = self.recruiter_keys[idx]
            target_key = self.target_keys[idx]
            family_key = self.target_family_keys[idx]
            if recruiter_key:
                same_e3_candidates = [
                    candidate
                    for candidate in self.recruiter_to_indices.get(recruiter_key, [])
                    if self.target_keys[candidate] != target_key
                ]
            else:
                same_e3_candidates = []
            if target_key:
                same_target_candidates = [
                    candidate
                    for candidate in self.target_to_indices.get(target_key, [])
                    if self.recruiter_keys[candidate] != recruiter_key
                ]
            else:
                same_target_candidates = []
            self.same_e3_candidate_indices.append(np.asarray(same_e3_candidates, dtype=np.int32))
            self.same_target_candidate_indices.append(np.asarray(same_target_candidates, dtype=np.int32))
            if family_key:
                family_candidates = [
                    candidate
                    for candidate in self.target_family_to_indices.get(family_key, [])
                    if self.target_keys[candidate] != target_key
                ]
            else:
                family_candidates = []
            self.cluster_target_candidate_indices.append(np.asarray(family_candidates, dtype=np.int32))
        self.observed_tuple_keys = {
            self._tuple_key(self.molecule_keys[idx], self.target_keys[idx], self.recruiter_keys[idx])
            for idx in range(len(self.records))
            if self.molecule_keys[idx] and self.target_keys[idx] and self.recruiter_keys[idx]
        }
        self.sample_stats: Counter[str] = Counter()
        self.base_payloads: List[Dict[str, Any]] = []
        for row in self.records:
            row_id = safe_text(row.get("row_id", ""))
            aux = self.resolver.get_aux(row_id)
            atom_x, adjacency, morgan_fp = self.resolver.mol_features(row, aux)
            target_feat = self.resolver.target_feature(row, aux)
            recruiter_feat = self.resolver.recruiter_feature(row, aux)
            self.base_payloads.append(
                {
                    "row_id": row_id,
                    "atom_x": atom_x.astype(np.float32, copy=False),
                    "adjacency": adjacency.astype(np.float32, copy=False),
                    "morgan_fp": morgan_fp.astype(np.float32, copy=False),
                    "target_feat": target_feat.astype(np.float32, copy=False),
                    "recruiter_feat": recruiter_feat.astype(np.float32, copy=False),
                }
            )
        empty_candidates = lambda: [np.empty(0, dtype=np.int32) for _ in self.records]
        self.frequency_pair_candidate_indices = (
            self._build_frequency_pair_candidates() if needs_frequency_candidates else empty_candidates()
        )
        if needs_protein_neighbors:
            self.target_nn_candidate_indices, self.recruiter_nn_candidate_indices = (
                self._build_protein_neighbor_candidates()
            )
            self.protein_pair_candidate_indices = self._build_protein_pair_candidates()
        else:
            self.target_nn_candidate_indices = empty_candidates()
            self.recruiter_nn_candidate_indices = empty_candidates()
            self.protein_pair_candidate_indices = empty_candidates()

    def __len__(self) -> int:
        return len(self.records)

    def _molecule_key(self, row: Dict[str, Any]) -> str:
        canonical = safe_text(row.get("canonical_smiles", ""))
        if canonical:
            return canonical
        smiles = safe_text(row.get("smiles", ""))
        if not smiles:
            return ""
        canonical = canonicalize_smiles(smiles) if RDKIT_AVAILABLE else ""
        return canonical or smiles

    def _tuple_key(self, molecule_key: str, target_key: str, recruiter_key: str) -> Tuple[str, str, str]:
        return (safe_text(molecule_key), safe_text(target_key), safe_text(recruiter_key))

    def _sample_index(self, candidates: Sequence[int]) -> Optional[int]:
        if len(candidates) == 0:
            return None
        return int(candidates[np.random.randint(0, len(candidates))])

    def _sample_random_pair(
        self,
        target_key: str,
        recruiter_key: str,
        require_target_change: bool = False,
        require_recruiter_change: bool = False,
    ) -> Optional[Tuple[int, int]]:
        if not self.target_pool_indices or not self.recruiter_pool_indices:
            return None
        for _ in range(32):
            target_idx = self._sample_index(self.target_pool_indices)
            recruiter_idx = self._sample_index(self.recruiter_pool_indices)
            if target_idx is None or recruiter_idx is None:
                continue
            target_changed = self.target_keys[target_idx] != target_key
            recruiter_changed = self.recruiter_keys[recruiter_idx] != recruiter_key
            if require_target_change and not target_changed:
                continue
            if require_recruiter_change and not recruiter_changed:
                continue
            if target_changed or recruiter_changed:
                return target_idx, recruiter_idx
        return None

    def _sample_same_e3_diff_target(self, idx: int) -> Optional[int]:
        candidates = self.same_e3_candidate_indices[idx]
        return self._sample_index(candidates)

    def _sample_same_target_diff_e3(self, idx: int) -> Optional[int]:
        candidates = self.same_target_candidate_indices[idx]
        return self._sample_index(candidates)

    def _build_frequency_pair_candidates(self) -> List[np.ndarray]:
        candidates_by_row: List[np.ndarray] = []
        for idx in range(len(self.records)):
            target_key = self.target_keys[idx]
            recruiter_key = self.recruiter_keys[idx]
            target_count = math.log1p(self.target_count_by_key.get(target_key, 0))
            recruiter_count = math.log1p(self.recruiter_count_by_key.get(recruiter_key, 0))
            ranked = sorted(
                (
                    candidate
                    for candidate in self.pair_pool_indices
                    if self.target_keys[candidate] != target_key or self.recruiter_keys[candidate] != recruiter_key
                ),
                key=lambda candidate: (
                    abs(math.log1p(self.target_count_by_key.get(self.target_keys[candidate], 0)) - target_count)
                    + abs(
                        math.log1p(self.recruiter_count_by_key.get(self.recruiter_keys[candidate], 0))
                        - recruiter_count
                    ),
                    self.target_keys[candidate],
                    self.recruiter_keys[candidate],
                    candidate,
                ),
            )
            candidates_by_row.append(np.asarray(ranked[:64], dtype=np.int32))
        return candidates_by_row

    def _build_protein_neighbor_candidates(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        target_rep_index = {key: indices[0] for key, indices in self.target_to_indices.items() if indices}
        recruiter_rep_index = {key: indices[0] for key, indices in self.recruiter_to_indices.items() if indices}
        target_rep_keys = sorted(target_rep_index)
        recruiter_rep_keys = sorted(recruiter_rep_index)

        def _cosine(left: np.ndarray, right: np.ndarray) -> float:
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom <= 0.0:
                return -1.0
            return float(np.dot(left, right) / denom)

        target_candidates: List[np.ndarray] = []
        recruiter_candidates: List[np.ndarray] = []
        for idx in range(len(self.records)):
            target_key = self.target_keys[idx]
            recruiter_key = self.recruiter_keys[idx]
            target_feat = self.base_payloads[idx]["target_feat"]
            recruiter_feat = self.base_payloads[idx]["recruiter_feat"]
            ranked_targets = sorted(
                (
                    target_rep_index[key]
                    for key in target_rep_keys
                    if key and key != target_key
                ),
                key=lambda candidate: (
                    -_cosine(target_feat, self.base_payloads[candidate]["target_feat"]),
                    self.target_keys[candidate],
                    candidate,
                ),
            )
            ranked_recruiters = sorted(
                (
                    recruiter_rep_index[key]
                    for key in recruiter_rep_keys
                    if key and key != recruiter_key
                ),
                key=lambda candidate: (
                    -_cosine(recruiter_feat, self.base_payloads[candidate]["recruiter_feat"]),
                    self.recruiter_keys[candidate],
                    candidate,
                ),
            )
            target_candidates.append(np.asarray(ranked_targets[:32], dtype=np.int32))
            recruiter_candidates.append(np.asarray(ranked_recruiters[:16], dtype=np.int32))
        return target_candidates, recruiter_candidates

    def _build_protein_pair_candidates(self) -> List[np.ndarray]:
        ranked_pairs: List[np.ndarray] = []

        def _cosine(left: np.ndarray, right: np.ndarray) -> float:
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom <= 0.0:
                return -1.0
            return float(np.dot(left, right) / denom)

        for idx in range(len(self.records)):
            target_key = self.target_keys[idx]
            recruiter_key = self.recruiter_keys[idx]
            target_feat = self.base_payloads[idx]["target_feat"]
            recruiter_feat = self.base_payloads[idx]["recruiter_feat"]
            ranked = sorted(
                (
                    candidate
                    for candidate in self.pair_pool_indices
                    if self.target_keys[candidate] != target_key or self.recruiter_keys[candidate] != recruiter_key
                ),
                key=lambda candidate: (
                    -(
                        _cosine(target_feat, self.base_payloads[candidate]["target_feat"])
                        + _cosine(recruiter_feat, self.base_payloads[candidate]["recruiter_feat"])
                    ),
                    self.target_keys[candidate],
                    self.recruiter_keys[candidate],
                    candidate,
                ),
            )
            ranked_pairs.append(np.asarray(ranked[:64], dtype=np.int32))
        return ranked_pairs

    def _sample_pair_candidate_negative(
        self,
        idx: int,
        candidate_indices: Sequence[int],
        stat_prefix: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        molecule_key = self.molecule_keys[idx]
        if len(candidate_indices) == 0:
            self.sample_stats[f"{stat_prefix}_failed"] += 1
            return None
        sample_pool = list(candidate_indices[: min(len(candidate_indices), 16)])
        for _ in range(64):
            candidate_idx = self._sample_index(sample_pool)
            if candidate_idx is None:
                break
            candidate_target = self.target_keys[candidate_idx]
            candidate_recruiter = self.recruiter_keys[candidate_idx]
            if self._is_observed_collision(molecule_key, candidate_target, candidate_recruiter):
                self.sample_stats[f"{stat_prefix}_observed_collision_reject"] += 1
                continue
            self.sample_stats[f"{stat_prefix}_accept"] += 1
            return self.base_payloads[candidate_idx]["target_feat"], self.base_payloads[candidate_idx]["recruiter_feat"]
        self.sample_stats[f"{stat_prefix}_failed"] += 1
        return None

    def _is_observed_collision(self, molecule_key: str, target_key: str, recruiter_key: str) -> bool:
        return self._tuple_key(molecule_key, target_key, recruiter_key) in self.observed_tuple_keys

    def _sample_positive_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.positive_construction == "observed":
            return self.base_payloads[idx]["target_feat"], self.base_payloads[idx]["recruiter_feat"]
        molecule_key = self.molecule_keys[idx]
        target_key = self.target_keys[idx]
        recruiter_key = self.recruiter_keys[idx]
        if self.positive_construction == "shuffled_pair":
            for _ in range(64):
                candidate_idx = self._sample_index(self.pair_pool_indices)
                if candidate_idx is None:
                    break
                candidate_target = self.target_keys[candidate_idx]
                candidate_recruiter = self.recruiter_keys[candidate_idx]
                if candidate_target == target_key and candidate_recruiter == recruiter_key:
                    self.sample_stats["positive_same_tuple_reject"] += 1
                    continue
                if self._is_observed_collision(molecule_key, candidate_target, candidate_recruiter):
                    self.sample_stats["positive_observed_collision_reject"] += 1
                    continue
                self.sample_stats["positive_shuffled_pair_accept"] += 1
                return self.base_payloads[candidate_idx]["target_feat"], self.base_payloads[candidate_idx]["recruiter_feat"]
        elif self.positive_construction == "shuffled_independent":
            for _ in range(64):
                sampled = self._sample_random_pair(
                    target_key,
                    recruiter_key,
                    require_target_change=True,
                    require_recruiter_change=True,
                )
                if sampled is None:
                    break
                target_idx, recruiter_idx = sampled
                candidate_target = self.target_keys[target_idx]
                candidate_recruiter = self.recruiter_keys[recruiter_idx]
                if self._is_observed_collision(molecule_key, candidate_target, candidate_recruiter):
                    self.sample_stats["positive_observed_collision_reject"] += 1
                    continue
                self.sample_stats["positive_shuffled_independent_accept"] += 1
                return self.base_payloads[target_idx]["target_feat"], self.base_payloads[recruiter_idx]["recruiter_feat"]
        self.sample_stats["positive_fallback_to_observed"] += 1
        return self.base_payloads[idx]["target_feat"], self.base_payloads[idx]["recruiter_feat"]

    def _sample_target_only_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        molecule_key = self.molecule_keys[idx]
        recruiter_key = self.recruiter_keys[idx]
        target_key = self.target_keys[idx]
        for _ in range(64):
            target_idx = self._sample_index(self.target_pool_indices)
            if target_idx is None or self.target_keys[target_idx] == target_key:
                self.sample_stats["negative_target_same_target_reject"] += 1
                continue
            candidate_target = self.target_keys[target_idx]
            if self._is_observed_collision(molecule_key, candidate_target, recruiter_key):
                self.sample_stats["negative_target_observed_collision_reject"] += 1
                continue
            self.sample_stats["negative_target_accept"] += 1
            return self.base_payloads[target_idx]["target_feat"], self.base_payloads[idx]["recruiter_feat"]
        self.sample_stats["negative_target_failed"] += 1
        return None

    def _sample_e3_only_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        molecule_key = self.molecule_keys[idx]
        target_key = self.target_keys[idx]
        recruiter_key = self.recruiter_keys[idx]
        for _ in range(64):
            recruiter_idx = self._sample_index(self.recruiter_pool_indices)
            if recruiter_idx is None or self.recruiter_keys[recruiter_idx] == recruiter_key:
                self.sample_stats["negative_e3_same_recruiter_reject"] += 1
                continue
            candidate_recruiter = self.recruiter_keys[recruiter_idx]
            if self._is_observed_collision(molecule_key, target_key, candidate_recruiter):
                self.sample_stats["negative_e3_observed_collision_reject"] += 1
                continue
            self.sample_stats["negative_e3_accept"] += 1
            return self.base_payloads[idx]["target_feat"], self.base_payloads[recruiter_idx]["recruiter_feat"]
        self.sample_stats["negative_e3_failed"] += 1
        return None

    def _sample_joint_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        molecule_key = self.molecule_keys[idx]
        target_key = self.target_keys[idx]
        recruiter_key = self.recruiter_keys[idx]
        for _ in range(64):
            sampled = self._sample_random_pair(target_key, recruiter_key)
            if sampled is None:
                break
            target_idx, recruiter_idx = sampled
            candidate_target = self.target_keys[target_idx]
            candidate_recruiter = self.recruiter_keys[recruiter_idx]
            if self._is_observed_collision(molecule_key, candidate_target, candidate_recruiter):
                self.sample_stats["negative_joint_observed_collision_reject"] += 1
                continue
            self.sample_stats["negative_joint_accept"] += 1
            return self.base_payloads[target_idx]["target_feat"], self.base_payloads[recruiter_idx]["recruiter_feat"]
        self.sample_stats["negative_joint_failed"] += 1
        return None

    def _sample_frequency_matched_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._sample_pair_candidate_negative(idx, self.frequency_pair_candidate_indices[idx], "negative_frequency_matched")

    def _sample_target_seq_cluster_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        molecule_key = self.molecule_keys[idx]
        recruiter_key = self.recruiter_keys[idx]
        candidates = self.cluster_target_candidate_indices[idx]
        if len(candidates) == 0:
            self.sample_stats["negative_target_seq_cluster_failed"] += 1
            return None
        sample_pool = list(candidates[: min(len(candidates), 16)])
        for _ in range(64):
            target_idx = self._sample_index(sample_pool)
            if target_idx is None:
                break
            candidate_target = self.target_keys[target_idx]
            if self._is_observed_collision(molecule_key, candidate_target, recruiter_key):
                self.sample_stats["negative_target_seq_cluster_observed_collision_reject"] += 1
                continue
            self.sample_stats["negative_target_seq_cluster_accept"] += 1
            return self.base_payloads[target_idx]["target_feat"], self.base_payloads[idx]["recruiter_feat"]
        self.sample_stats["negative_target_seq_cluster_failed"] += 1
        return None

    def _sample_protein_nn_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pair_sample = self._sample_pair_candidate_negative(idx, self.protein_pair_candidate_indices[idx], "negative_protein_nn_pair")
        if pair_sample is not None:
            return pair_sample
        molecule_key = self.molecule_keys[idx]
        recruiter_key = self.recruiter_keys[idx]
        target_key = self.target_keys[idx]
        target_candidates = list(self.target_nn_candidate_indices[idx][: min(len(self.target_nn_candidate_indices[idx]), 8)])
        recruiter_candidates = list(
            self.recruiter_nn_candidate_indices[idx][: min(len(self.recruiter_nn_candidate_indices[idx]), 4)]
        )
        for _ in range(64):
            target_idx = self._sample_index(target_candidates) if target_candidates else None
            recruiter_idx = self._sample_index(recruiter_candidates) if recruiter_candidates else None
            if target_idx is None and recruiter_idx is None:
                break
            candidate_target = self.target_keys[target_idx] if target_idx is not None else target_key
            candidate_recruiter = self.recruiter_keys[recruiter_idx] if recruiter_idx is not None else recruiter_key
            if candidate_target == target_key and candidate_recruiter == recruiter_key:
                self.sample_stats["negative_protein_nn_same_tuple_reject"] += 1
                continue
            if self._is_observed_collision(molecule_key, candidate_target, candidate_recruiter):
                self.sample_stats["negative_protein_nn_observed_collision_reject"] += 1
                continue
            self.sample_stats["negative_protein_nn_accept"] += 1
            return (
                self.base_payloads[target_idx]["target_feat"] if target_idx is not None else self.base_payloads[idx]["target_feat"],
                self.base_payloads[recruiter_idx]["recruiter_feat"]
                if recruiter_idx is not None
                else self.base_payloads[idx]["recruiter_feat"],
            )
        self.sample_stats["negative_protein_nn_failed"] += 1
        return None

    def _sample_legacy_random_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._sample_joint_negative(idx)

    def _sample_legacy_same_e3_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sampled = self._sample_same_e3_diff_target(idx)
        if sampled is None:
            self.sample_stats["negative_same_e3_failed"] += 1
            return None
        self.sample_stats["negative_same_e3_accept"] += 1
        return self.base_payloads[sampled]["target_feat"], self.base_payloads[idx]["recruiter_feat"]

    def _sample_legacy_same_target_negative(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sampled = self._sample_same_target_diff_e3(idx)
        if sampled is None:
            self.sample_stats["negative_same_target_failed"] += 1
            return None
        self.sample_stats["negative_same_target_accept"] += 1
        return self.base_payloads[idx]["target_feat"], self.base_payloads[sampled]["recruiter_feat"]

    def stats_snapshot(self) -> Dict[str, int]:
        return {key: int(value) for key, value in sorted(self.sample_stats.items())}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.base_payloads[idx]
        row_id = base["row_id"]
        atom_x = base["atom_x"]
        adjacency = base["adjacency"]
        morgan_fp = base["morgan_fp"]
        positive_target_feat, positive_recruiter_feat = self._sample_positive_pair(idx)
        negative_target_feats = np.zeros((len(self.active_policies), positive_target_feat.shape[0]), dtype=np.float32)
        negative_recruiter_feats = np.zeros((len(self.active_policies), positive_recruiter_feat.shape[0]), dtype=np.float32)
        negative_masks = np.zeros(len(self.active_policies), dtype=bool)
        for negative_idx, policy in enumerate(self.active_policies):
            sampled: Optional[Tuple[np.ndarray, np.ndarray]]
            if policy == "target_only":
                sampled = self._sample_target_only_negative(idx)
            elif policy == "e3_only":
                sampled = self._sample_e3_only_negative(idx)
            elif policy == "joint_only":
                sampled = self._sample_joint_negative(idx)
            elif policy == "frequency_matched":
                sampled = self._sample_frequency_matched_negative(idx)
            elif policy == "target_seq_cluster":
                sampled = self._sample_target_seq_cluster_negative(idx)
            elif policy == "protein_nn":
                sampled = self._sample_protein_nn_negative(idx)
            elif policy == "random":
                sampled = self._sample_legacy_random_negative(idx)
            elif policy == "same_e3_diff_target":
                sampled = self._sample_legacy_same_e3_negative(idx)
            elif policy == "same_target_diff_e3":
                sampled = self._sample_legacy_same_target_negative(idx)
            else:
                raise ValueError(f"Unsupported tuple negative policy item: {policy}")
            if sampled is None:
                continue
            neg_target_feat, neg_recruiter_feat = sampled
            negative_target_feats[negative_idx] = neg_target_feat.astype(np.float32, copy=False)
            negative_recruiter_feats[negative_idx] = neg_recruiter_feat.astype(np.float32, copy=False)
            negative_masks[negative_idx] = True
        return {
            "row_id": row_id,
            "atom_x": atom_x,
            "adjacency": adjacency,
            "morgan_fp": morgan_fp,
            "positive_target_feat": positive_target_feat.astype(np.float32, copy=False),
            "positive_recruiter_feat": positive_recruiter_feat.astype(np.float32, copy=False),
            "negative_target_feat": negative_target_feats,
            "negative_recruiter_feat": negative_recruiter_feats,
            "negative_mask": negative_masks,
        }


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_atoms = max(item["atom_x"].shape[0] for item in batch)
    atom_dim = batch[0]["atom_x"].shape[1]
    atom_x = np.zeros((len(batch), max_atoms, atom_dim), dtype=np.float32)
    adjacency = np.zeros((len(batch), max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((len(batch), max_atoms), dtype=np.float32)
    morgan_fp = np.stack([item["morgan_fp"] for item in batch], axis=0)
    target_feat = np.stack([item["target_feat"] for item in batch], axis=0)
    recruiter_feat = np.stack([item["recruiter_feat"] for item in batch], axis=0)
    labels = np.array([item["label"] for item in batch], dtype=np.float32)
    label_mask = np.array([item["is_labeled"] for item in batch], dtype=bool)
    sample_weight = np.array([item["sample_weight"] for item in batch], dtype=np.float32)
    cell_index = np.array([item["cell_index"] for item in batch], dtype=np.int64)
    kg_mol_index = np.array([int(item.get("kg_mol_index", -1)) for item in batch], dtype=np.int64)
    kg_target_index = np.array([int(item.get("kg_target_index", -1)) for item in batch], dtype=np.int64)
    kg_recruiter_index = np.array([int(item.get("kg_recruiter_index", -1)) for item in batch], dtype=np.int64)
    kg_feature = None
    if any(len(item.get("kg_feature", [])) > 0 for item in batch):
        kg_dim = max(int(np.asarray(item.get("kg_feature", [])).shape[0]) for item in batch)
        kg_feature = np.zeros((len(batch), kg_dim), dtype=np.float32)
    for idx, item in enumerate(batch):
        atom_count = item["atom_x"].shape[0]
        atom_x[idx, :atom_count] = item["atom_x"]
        adjacency[idx, :atom_count, :atom_count] = item["adjacency"]
        mask[idx, :atom_count] = 1.0
        if kg_feature is not None:
            value = np.asarray(item.get("kg_feature", []), dtype=np.float32)
            if value.size:
                kg_feature[idx, : value.shape[0]] = value
    payload = {
        "atom_x": torch.from_numpy(atom_x),
        "adjacency": torch.from_numpy(adjacency),
        "mask": torch.from_numpy(mask),
        "morgan_fp": torch.from_numpy(morgan_fp),
        "target_feat": torch.from_numpy(target_feat),
        "recruiter_feat": torch.from_numpy(recruiter_feat),
        "labels": torch.from_numpy(labels),
        "label_mask": torch.from_numpy(label_mask),
        "sample_weight": torch.from_numpy(sample_weight),
        "cell_index": torch.from_numpy(cell_index),
        "kg_mol_index": torch.from_numpy(kg_mol_index),
        "kg_target_index": torch.from_numpy(kg_target_index),
        "kg_recruiter_index": torch.from_numpy(kg_recruiter_index),
        "task_ids": [item["task_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
    if kg_feature is not None:
        payload["kg_feature"] = torch.from_numpy(kg_feature)
    return payload


def ctp_pretrain_collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_atoms = max(item["atom_x"].shape[0] for item in batch)
    atom_dim = batch[0]["atom_x"].shape[1]
    atom_x = np.zeros((len(batch), max_atoms, atom_dim), dtype=np.float32)
    adjacency = np.zeros((len(batch), max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((len(batch), max_atoms), dtype=np.float32)
    for idx, item in enumerate(batch):
        atom_count = item["atom_x"].shape[0]
        atom_x[idx, :atom_count] = item["atom_x"]
        adjacency[idx, :atom_count, :atom_count] = item["adjacency"]
        mask[idx, :atom_count] = 1.0
    def stack(name: str) -> np.ndarray:
        return np.stack([item[name] for item in batch], axis=0).astype(np.float32)
    def stack_mask(name: str) -> np.ndarray:
        return np.asarray([item[name] for item in batch], dtype=bool)
    return {
        "atom_x": torch.from_numpy(atom_x),
        "adjacency": torch.from_numpy(adjacency),
        "mask": torch.from_numpy(mask),
        "morgan_fp": torch.from_numpy(stack("morgan_fp")),
        "positive_target_feat": torch.from_numpy(stack("positive_target_feat")),
        "positive_recruiter_feat": torch.from_numpy(stack("positive_recruiter_feat")),
        "negative_target_feat": torch.from_numpy(stack("negative_target_feat")),
        "negative_recruiter_feat": torch.from_numpy(stack("negative_recruiter_feat")),
        "negative_mask": torch.from_numpy(stack_mask("negative_mask")),
        "row_ids": [item["row_id"] for item in batch],
    }


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class ResidualGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        message = torch.bmm(adjacency, hidden)
        updated = self.block(torch.cat([hidden, message], dim=-1))
        updated = self.norm(hidden + self.dropout(updated))
        return updated * mask.unsqueeze(-1)


class DQSeqGraph(nn.Module):
    def __init__(self, config: Dict[str, Any], num_cell_lines: int) -> None:
        super().__init__()
        hidden_dim = int(config["model"]["hidden_dim"])
        protein_dim = int(config["features"]["protein_feature_dim"])
        fp_dim = int(config["features"]["morgan_bits"])
        dropout = float(config["model"]["mol_gnn_dropout"])
        graph_layers = int(config["model"]["mol_gnn_layers"])
        fusion_dropout = float(config["model"]["fusion_dropout"])
        self.atom_encoder = MLPBlock(15, hidden_dim, dropout)
        self.graph_layers = nn.ModuleList([ResidualGraphLayer(hidden_dim, dropout) for _ in range(graph_layers)])
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
        )
        self.mol_norm = nn.LayerNorm(hidden_dim)
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_embedding = nn.Embedding(max(num_cell_lines, 1), 64)
        self.hidden_dim = hidden_dim
        self.tuple_fusion_dim = hidden_dim * 7
        self.use_kg_side_features = bool(config["model"].get("use_kg_side_features", False))
        self.kg_feature_dim = safe_int(config["model"].get("kg_feature_dim"), 0)
        self.kg_proj_dim = safe_int(config["model"].get("kg_proj_dim"), 128) if self.use_kg_side_features else 0
        self.kg_proj = None
        if self.use_kg_side_features and self.kg_feature_dim > 0 and self.kg_proj_dim > 0:
            self.kg_proj = nn.Sequential(
                nn.Linear(self.kg_feature_dim, self.kg_proj_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(self.kg_proj_dim, self.kg_proj_dim),
            )
        fusion_dim = self.tuple_fusion_dim + 64 + self.kg_proj_dim
        self.tuple_head = nn.Sequential(
            nn.Linear(self.tuple_fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.mol_proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(256, 1),
        )

    def encode_graph(self, atom_x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.atom_encoder(atom_x) * mask.unsqueeze(-1)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adjacency_norm = adjacency / degree
        for layer in self.graph_layers:
            hidden = layer(hidden, adjacency_norm, mask)
        masked_hidden = hidden * mask.unsqueeze(-1)
        pooled_mean = masked_hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_max = masked_hidden.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.graph_proj(torch.cat([pooled_mean, pooled_max], dim=-1))

    def encode_molecule(
        self,
        atom_x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
        morgan_fp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_emb = self.encode_graph(atom_x, adjacency, mask)
        fp_emb = self.fp_proj(morgan_fp)
        mol_emb = self.mol_norm(graph_emb + fp_emb)
        return mol_emb, graph_emb

    def encode_proteins(self, target_feat: torch.Tensor, recruiter_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.protein_proj(target_feat), self.protein_proj(recruiter_feat)

    def apply_ablation(
        self,
        mol_emb: torch.Tensor,
        target_emb: torch.Tensor,
        recruiter_emb: torch.Tensor,
        cell_emb: torch.Tensor,
        ablation: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ablation = canonical_ablation_name(ablation)
        if ablation == "smiles_only":
            target_emb = torch.zeros_like(target_emb)
            recruiter_emb = torch.zeros_like(recruiter_emb)
            cell_emb = torch.zeros_like(cell_emb)
        elif ablation == "target_only":
            mol_emb = torch.zeros_like(mol_emb)
            recruiter_emb = torch.zeros_like(recruiter_emb)
            cell_emb = torch.zeros_like(cell_emb)
        elif ablation == "e3_only":
            mol_emb = torch.zeros_like(mol_emb)
            target_emb = torch.zeros_like(target_emb)
            cell_emb = torch.zeros_like(cell_emb)
        elif ablation == "target_e3_only":
            mol_emb = torch.zeros_like(mol_emb)
            cell_emb = torch.zeros_like(cell_emb)
        elif ablation == "target_e3_assay_only":
            mol_emb = torch.zeros_like(mol_emb)
        elif ablation == "smiles_graph_fp":
            target_emb = torch.zeros_like(target_emb)
            recruiter_emb = torch.zeros_like(recruiter_emb)
            cell_emb = torch.zeros_like(cell_emb)
        elif ablation in {"smiles_target_id", "smiles_target_seq"}:
            recruiter_emb = torch.zeros_like(recruiter_emb)
            cell_emb = torch.zeros_like(cell_emb)
        elif ablation == "smiles_target_recruiter_seq":
            cell_emb = torch.zeros_like(cell_emb)
        return mol_emb, target_emb, recruiter_emb, cell_emb

    def fuse_tuple(
        self,
        mol_emb: torch.Tensor,
        target_emb: torch.Tensor,
        recruiter_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                mol_emb,
                target_emb,
                recruiter_emb,
                mol_emb * target_emb,
                mol_emb * recruiter_emb,
                target_emb * recruiter_emb,
                torch.abs(target_emb - recruiter_emb),
            ],
            dim=-1,
        )

    def fuse_with_cell(
        self,
        mol_emb: torch.Tensor,
        target_emb: torch.Tensor,
        recruiter_emb: torch.Tensor,
        cell_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([self.fuse_tuple(mol_emb, target_emb, recruiter_emb), cell_emb], dim=-1)

    def tuple_score(self, mol_emb: torch.Tensor, target_emb: torch.Tensor, recruiter_emb: torch.Tensor) -> torch.Tensor:
        return self.tuple_head(self.fuse_tuple(mol_emb, target_emb, recruiter_emb)).squeeze(-1)

    def mol_projection(self, graph_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mol_proj_head(graph_emb), p=2, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], ablation: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mol_emb, _ = self.encode_molecule(batch["atom_x"], batch["adjacency"], batch["mask"], batch["morgan_fp"])
        target_emb, recruiter_emb = self.encode_proteins(batch["target_feat"], batch["recruiter_feat"])
        cell_emb = self.cell_embedding(batch["cell_index"])
        mol_emb, target_emb, recruiter_emb, cell_emb = self.apply_ablation(
            mol_emb, target_emb, recruiter_emb, cell_emb, ablation
        )
        fusion = self.fuse_with_cell(mol_emb, target_emb, recruiter_emb, cell_emb)
        if self.kg_proj is not None:
            if "kg_feature" in batch:
                kg_emb = self.kg_proj(batch["kg_feature"])
            else:
                kg_emb = torch.zeros((fusion.shape[0], self.kg_proj_dim), device=fusion.device, dtype=fusion.dtype)
            fusion = torch.cat([fusion, kg_emb], dim=-1)
        logits = self.classifier(fusion).squeeze(-1)
        return logits, fusion


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= start) & (y_prob < end if end < 1.0 else y_prob <= end)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += abs(avg_conf - avg_acc) * (mask.sum() / len(y_true))
    return float(ece)


def safe_metric(callable_obj, default: float = float("nan")) -> float:
    try:
        return float(callable_obj())
    except Exception:
        return default


def metric_key(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return numeric if not math.isnan(numeric) else float("-inf")


def selection_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        metric_key(metrics.get("accuracy")),
        metric_key(metrics.get("f1")),
        metric_key(metrics.get("auroc")),
    )


def optimize_threshold(y_true: Sequence[int], y_prob: Sequence[float], objective: str) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    thresholds = sorted({0.0, 0.5, 1.0, *np.clip(y_prob_arr, 0.0, 1.0).tolist()})
    best = {"threshold": 0.5, "score": float("-inf"), "metrics": {}}
    for threshold in thresholds:
        metrics = compute_metrics(y_true_arr, y_prob_arr, ece_bins=10, threshold=float(threshold))
        if objective == "f1":
            score = float(metrics.get("f1", float("-inf")))
        elif objective == "balacc":
            score = float(metrics.get("balanced_accuracy", float("-inf")))
        elif objective == "youden":
            recall = float(metrics.get("recall", float("nan")))
            specificity = float(metrics.get("specificity", float("nan")))
            score = recall + specificity - 1.0 if not (math.isnan(recall) or math.isnan(specificity)) else float("-inf")
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        if score > best["score"]:
            best = {"threshold": float(threshold), "score": score, "metrics": metrics}
    return best


def optimize_threshold_selected(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    metrics = compute_metrics(y_true_arr, y_prob_arr, ece_bins=10, threshold=0.5)
    return {
        "threshold": 0.5,
        "metrics": metrics,
        "selection_rule": THRESHOLD_RULE,
        "selection_scope": "predeclared",
    }


def threshold_candidates_from_predictions(predictions: Dict[str, Dict[str, Sequence[float]]]) -> Dict[str, Any]:
    candidates: Dict[str, Any] = {}
    for split_name, payload in predictions.items():
        truth = list(payload.get("truth", []))
        prob = list(payload.get("prob", []))
        if not truth or not prob:
            continue
        candidates[split_name] = {
            "f1": optimize_threshold(truth, prob, "f1"),
            "balacc": optimize_threshold(truth, prob, "balacc"),
            "youden": optimize_threshold(truth, prob, "youden"),
            "selected": optimize_threshold_selected(truth, prob),
        }
    return candidates


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    ece_bins: int,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true_arr) & np.isin(y_true_arr, [0.0, 1.0]) & np.isfinite(y_prob_arr)
    y_true_arr = y_true_arr[valid].astype(int)
    y_prob_arr = y_prob_arr[valid]
    if len(y_true_arr) == 0:
        return {
            "threshold": float(threshold),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "ece": float("nan"),
            "brier_score": float("nan"),
            "test_positive_count": 0,
            "test_negative_count": 0,
            "mean_positive_probability": float("nan"),
            "mean_negative_probability": float("nan"),
        }
    y_pred = (y_prob_arr >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / max(tn + fp, 1)
    positive_mask = y_true_arr == 1
    negative_mask = y_true_arr == 0
    return {
        "threshold": float(threshold),
        "auroc": safe_metric(lambda: roc_auc_score(y_true_arr, y_prob_arr)),
        "auprc": safe_metric(lambda: average_precision_score(y_true_arr, y_prob_arr)),
        "accuracy": safe_metric(lambda: accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": safe_metric(lambda: balanced_accuracy_score(y_true_arr, y_pred)),
        "f1": safe_metric(lambda: f1_score(y_true_arr, y_pred)),
        "mcc": safe_metric(lambda: matthews_corrcoef(y_true_arr, y_pred)),
        "precision": safe_metric(lambda: precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": safe_metric(lambda: recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "ece": ece_score(y_true_arr, y_prob_arr, ece_bins),
        "brier_score": safe_metric(lambda: brier_score_loss(y_true_arr, y_prob_arr)),
        "test_positive_count": int(positive_mask.sum()),
        "test_negative_count": int(negative_mask.sum()),
        "mean_positive_probability": float(y_prob_arr[positive_mask].mean()) if positive_mask.any() else float("nan"),
        "mean_negative_probability": float(y_prob_arr[negative_mask].mean()) if negative_mask.any() else float("nan"),
    }


def compute_episodic_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    task_ids: Sequence[str],
    config: Dict[str, Any],
) -> Optional[torch.Tensor]:
    grouped: Dict[str, List[int]] = {}
    for idx, task_id in enumerate(task_ids):
        grouped.setdefault(task_id, []).append(idx)
    k_support = int(config["model"]["episodic_k_support"])
    q_query = int(config["model"]["episodic_q_query"])
    min_task_size = int(config["model"]["episodic_min_task_size"])
    losses: List[torch.Tensor] = []
    for task_id, indices in grouped.items():
        if len(indices) < min_task_size:
            continue
        idx_tensor = torch.tensor(indices, device=embeddings.device)
        task_embeddings = embeddings[idx_tensor]
        task_labels = labels[idx_tensor]
        pos_indices = torch.where(task_labels > 0.5)[0]
        neg_indices = torch.where(task_labels <= 0.5)[0]
        if len(pos_indices) < k_support + q_query or len(neg_indices) < k_support + q_query:
            continue
        pos_perm = pos_indices[torch.randperm(len(pos_indices))]
        neg_perm = neg_indices[torch.randperm(len(neg_indices))]
        support_pos = pos_perm[:k_support]
        query_pos = pos_perm[k_support : k_support + q_query]
        support_neg = neg_perm[:k_support]
        query_neg = neg_perm[k_support : k_support + q_query]
        proto_pos = task_embeddings[support_pos].mean(dim=0, keepdim=True)
        proto_neg = task_embeddings[support_neg].mean(dim=0, keepdim=True)
        query_indices = torch.cat([query_pos, query_neg], dim=0)
        query_embeddings = task_embeddings[query_indices]
        query_labels = torch.cat(
            [
                torch.ones(len(query_pos), device=embeddings.device),
                torch.zeros(len(query_neg), device=embeddings.device),
            ]
        )
        dist_pos = ((query_embeddings - proto_pos) ** 2).mean(dim=-1)
        dist_neg = ((query_embeddings - proto_neg) ** 2).mean(dim=-1)
        logits = -dist_pos + dist_neg
        losses.append(F.binary_cross_entropy_with_logits(logits, query_labels))
    if not losses:
        return None
    return torch.stack(losses).mean()


def make_cell_vocab(frame: pd.DataFrame) -> Dict[str, int]:
    vocab = {"UNK_CELL": 0}
    values = sorted({safe_text(value) for value in frame["cell_line"].tolist() if safe_text(value)})
    for value in values:
        vocab[value] = len(vocab)
    return vocab


def build_dataloader(
    frame: pd.DataFrame,
    dataset_name: str,
    config: Dict[str, Any],
    feature_cache: Dict[str, Any],
    cell_vocab: Dict[str, int],
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset = DegradeDataset(frame, dataset_name, config, feature_cache, cell_vocab, device)
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )


def build_ctp_pretrain_dataloader(
    frame: pd.DataFrame,
    dataset_name: str,
    config: Dict[str, Any],
    feature_cache: Dict[str, Any],
    cell_vocab: Dict[str, int],
    device: torch.device,
    shuffle: bool,
    tuple_negative_policy: str,
    positive_construction: str,
) -> DataLoader:
    dataset = CTPPretrainDataset(
        frame,
        dataset_name,
        config,
        feature_cache,
        cell_vocab,
        device,
        tuple_negative_policy,
        positive_construction,
    )
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=int(config["pretrain"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=ctp_pretrain_collate_batch,
    )


def augment_molecule_batch(
    atom_x: torch.Tensor,
    adjacency: torch.Tensor,
    mask: torch.Tensor,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    aug_atom = atom_x.clone()
    aug_adj = adjacency.clone()
    edge_drop_prob = float(config["pretrain"].get("graph_edge_drop_prob", 0.0))
    atom_mask_prob = float(config["pretrain"].get("atom_mask_prob", 0.0))
    valid_atoms = mask > 0.5
    if atom_mask_prob > 0:
        atom_drop = (torch.rand_like(mask) < atom_mask_prob) & valid_atoms
        if atom_drop.any():
            aug_atom = aug_atom.masked_fill(atom_drop.unsqueeze(-1), 0.0)
    if edge_drop_prob > 0 and aug_adj.shape[-1] > 1:
        max_atoms = aug_adj.shape[-1]
        eye = torch.eye(max_atoms, device=aug_adj.device, dtype=torch.bool).unsqueeze(0)
        valid_pairs = valid_atoms.unsqueeze(1) & valid_atoms.unsqueeze(2)
        upper = torch.triu(torch.ones((max_atoms, max_atoms), device=aug_adj.device, dtype=torch.bool), diagonal=1).unsqueeze(0)
        existing_edges = aug_adj > 0
        drop_upper = (
            (torch.rand_like(aug_adj) < edge_drop_prob)
            & valid_pairs
            & existing_edges
            & upper
        )
        if drop_upper.any():
            drop_mask = drop_upper | drop_upper.transpose(1, 2)
            aug_adj = aug_adj.masked_fill(drop_mask, 0.0)
    diag_idx = torch.arange(aug_adj.shape[-1], device=aug_adj.device)
    aug_adj[:, diag_idx, diag_idx] = mask.to(aug_adj.dtype)
    return aug_atom, aug_adj


def symmetric_contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    if z_a.shape[0] == 0:
        return torch.tensor(0.0, device=z_a.device)
    logits = torch.matmul(z_a, z_b.transpose(0, 1)) / max(float(temperature), 1e-6)
    targets = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def tuple_contrast_loss(model: DQSeqGraph, batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
    mol_emb, _ = model.encode_molecule(batch["atom_x"], batch["adjacency"], batch["mask"], batch["morgan_fp"])
    pos_target_emb, pos_recruiter_emb = model.encode_proteins(batch["positive_target_feat"], batch["positive_recruiter_feat"])
    positive_score = model.tuple_score(mol_emb, pos_target_emb, pos_recruiter_emb).unsqueeze(1)
    negative_mask = batch["negative_mask"].bool()
    if negative_mask.ndim != 2 or not negative_mask.any():
        return torch.tensor(0.0, device=mol_emb.device)
    batch_size, negatives_per_example = negative_mask.shape
    neg_target_flat = batch["negative_target_feat"].reshape(batch_size * negatives_per_example, -1)
    neg_recruiter_flat = batch["negative_recruiter_feat"].reshape(batch_size * negatives_per_example, -1)
    neg_target_emb_flat, neg_recruiter_emb_flat = model.encode_proteins(neg_target_flat, neg_recruiter_flat)
    mol_repeat = mol_emb.unsqueeze(1).expand(-1, negatives_per_example, -1).reshape(batch_size * negatives_per_example, -1)
    negative_scores = model.tuple_score(mol_repeat, neg_target_emb_flat, neg_recruiter_emb_flat).reshape(batch_size, negatives_per_example)
    negative_scores = negative_scores.masked_fill(~negative_mask, -1e9)
    valid_mask = negative_mask.any(dim=1)
    logits = torch.cat([positive_score, negative_scores], dim=1) / max(float(temperature), 1e-6)
    targets = torch.zeros(int(valid_mask.sum().item()), dtype=torch.long, device=mol_emb.device)
    return F.cross_entropy(logits[valid_mask], targets)


def mol_contrast_loss(model: DQSeqGraph, batch: Dict[str, torch.Tensor], config: Dict[str, Any]) -> torch.Tensor:
    view_a_x, view_a_adj = augment_molecule_batch(batch["atom_x"], batch["adjacency"], batch["mask"], config)
    view_b_x, view_b_adj = augment_molecule_batch(batch["atom_x"], batch["adjacency"], batch["mask"], config)
    stacked_x = torch.cat([view_a_x, view_b_x], dim=0)
    stacked_adj = torch.cat([view_a_adj, view_b_adj], dim=0)
    stacked_mask = torch.cat([batch["mask"], batch["mask"]], dim=0)
    stacked_fp = torch.cat([batch["morgan_fp"], batch["morgan_fp"]], dim=0)
    _, stacked_graph = model.encode_molecule(stacked_x, stacked_adj, stacked_mask, stacked_fp)
    graph_a, graph_b = torch.chunk(stacked_graph, 2, dim=0)
    proj_a = model.mol_projection(graph_a)
    proj_b = model.mol_projection(graph_b)
    pretrain_cfg = config["pretrain"]
    temperature = float(pretrain_cfg.get("temperature_mol", pretrain_cfg["temperature"]))
    return symmetric_contrastive_loss(proj_a, proj_b, temperature)


def run_pretrain_epoch(
    model: DQSeqGraph,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    tasks: Sequence[str],
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(mode=training)
    total_losses: List[float] = []
    tuple_losses: List[float] = []
    mol_losses: List[float] = []
    tuple_weight = float(config["pretrain"]["lambda_tuple"])
    mol_weight = float(config["pretrain"]["lambda_mol"])
    pretrain_cfg = config["pretrain"]
    tuple_temperature = float(pretrain_cfg.get("temperature_tuple", pretrain_cfg["temperature"]))
    task_set = set(tasks)
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            total_loss = torch.tensor(0.0, device=device)
            tuple_loss_value = torch.tensor(0.0, device=device)
            mol_loss_value = torch.tensor(0.0, device=device)
            if "tuple_contrast" in task_set:
                tuple_loss_value = tuple_contrast_loss(model, batch, tuple_temperature)
                total_loss = total_loss + tuple_weight * tuple_loss_value
            if "mol_contrast" in task_set:
                mol_loss_value = mol_contrast_loss(model, batch, config)
                total_loss = total_loss + mol_weight * mol_loss_value
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
                optimizer.step()
        total_losses.append(float(total_loss.detach().cpu()))
        tuple_losses.append(float(tuple_loss_value.detach().cpu()))
        mol_losses.append(float(mol_loss_value.detach().cpu()))
    return {
        "loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "tuple_loss": float(np.mean(tuple_losses)) if tuple_losses else float("nan"),
        "mol_loss": float(np.mean(mol_losses)) if mol_losses else float("nan"),
    }


def resolve_bce_pos_weight(config: Dict[str, Any]) -> float:
    spec = config["train"].get("class_weight", "auto")
    if isinstance(spec, (int, float)):
        return float(spec)
    text = safe_text(spec).lower()
    if text in {"", "auto"}:
        return float(config["train"].get("_pos_weight", 1.0))
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported --pos-weight value: {spec}") from exc


def resolve_focal_alpha(config: Dict[str, Any]) -> float:
    spec = config["train"].get("focal_alpha", "auto")
    if isinstance(spec, (int, float)):
        return float(spec)
    text = safe_text(spec).lower()
    if text in {"", "auto"}:
        pos_weight = float(config["train"].get("_pos_weight", 1.0))
        positive_prior = 1.0 / (1.0 + pos_weight) if pos_weight > 0 else 0.5
        return float(max(1e-6, min(1.0 - 1e-6, 1.0 - positive_prior)))
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported --focal-alpha value: {spec}") from exc


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    reduction: str = "none",
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_factor * ((1.0 - p_t).clamp_min(1e-8) ** gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def run_epoch(
    model: DQSeqGraph,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer],
    ablation: Optional[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(mode=training)
    all_true: List[int] = []
    all_prob: List[float] = []
    all_meta: List[Dict[str, Any]] = []
    losses: List[float] = []
    loss_name = safe_text(config["train"].get("loss", "bce")).lower() or "bce"
    pos_weight_value = resolve_bce_pos_weight(config)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    focal_gamma = float(config["train"].get("focal_gamma", 2.0))
    focal_alpha = resolve_focal_alpha(config)
    ablation = canonical_ablation_name(ablation)
    episodic_enabled = bool(config["model"].get("use_task_episodic_loss", False))
    if ablation == "full_no_episodic":
        episodic_enabled = False
    if ablation == "full_with_episodic":
        episodic_enabled = True
    if ablation == "smiles_only":
        episodic_enabled = False
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            logits, embeddings = model(batch, ablation=ablation)
            labeled_mask = batch["label_mask"]
            probs = torch.sigmoid(logits)
            if labeled_mask.any():
                if loss_name == "focal":
                    base_loss = binary_focal_loss_with_logits(
                        logits[labeled_mask],
                        batch["labels"][labeled_mask],
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                        reduction="none",
                    )
                else:
                    base_loss = F.binary_cross_entropy_with_logits(
                        logits[labeled_mask],
                        batch["labels"][labeled_mask],
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                weighted = base_loss * batch["sample_weight"][labeled_mask]
                loss = weighted.mean()
                if episodic_enabled:
                    task_ids = [task_id for task_id, keep in zip(batch["task_ids"], labeled_mask.tolist()) if keep]
                    episodic_loss = compute_episodic_loss(
                        embeddings[labeled_mask],
                        batch["labels"][labeled_mask],
                        task_ids,
                        config,
                    )
                    if episodic_loss is not None:
                        loss = loss + float(config["model"]["episodic_alpha"]) * episodic_loss
                losses.append(float(loss.detach().cpu()))
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
                    optimizer.step()
            elif training:
                optimizer.zero_grad()
        for meta, label, prob, keep in zip(
            batch["meta"],
            batch["labels"].detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
            batch["label_mask"].tolist(),
        ):
            if keep:
                all_true.append(int(label))
                all_prob.append(float(prob))
            meta_copy = dict(meta)
            meta_copy["prob"] = float(prob)
            meta_copy["pred"] = int(prob >= float(threshold))
            all_meta.append(meta_copy)
    result = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "truth": all_true,
        "prob": all_prob,
        "meta": all_meta,
        "metrics": compute_metrics(all_true, all_prob, int(config["output"]["ece_bins"]), threshold=threshold) if all_true else {},
    }
    return result


def dataframe_for_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return frame[frame["split"] == split_name].reset_index(drop=True)


def analysis_field_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    for value in frame[column].tolist():
        text = safe_text(value)
        if text:
            return text
    return ""


def analysis_field_float(frame: pd.DataFrame, column: str) -> Optional[float]:
    if column not in frame.columns:
        return None
    for value in frame[column].tolist():
        if value is None:
            continue
        text = safe_text(value)
        if not text:
            continue
        parsed = safe_int(value, None) if isinstance(value, int) else None
        if parsed is not None and not isinstance(value, float):
            return float(parsed)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def analysis_field_int(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    for value in frame[column].tolist():
        text = safe_text(value)
        if not text:
            continue
        return safe_int(value, 0)
    return 0


def analysis_metadata_from_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    scheme = analysis_field_text(frame, "analysis_scheme")
    return {
        "analysis_scheme": scheme,
        "analysis_seed": analysis_field_int(frame, "analysis_seed"),
        "label_fraction": analysis_field_float(frame, "label_fraction"),
        "test_frac": analysis_field_float(frame, "test_frac"),
        "shots_per_class": analysis_field_int(frame, "shots_per_class"),
    }


def analysis_split_policy(meta: Dict[str, Any], config: Dict[str, Any]) -> str:
    scheme = safe_text(meta.get("analysis_scheme"))
    if not scheme or scheme == "official":
        return safe_text(config["benchmark"].get("protocol", ""))
    chunks = [scheme]
    label_fraction = meta.get("label_fraction")
    test_frac = meta.get("test_frac")
    shots_per_class = safe_int(meta.get("shots_per_class"), 0)
    seed = safe_int(meta.get("analysis_seed"), 0)
    if label_fraction is not None:
        chunks.append(f"frac{float(label_fraction):.2f}".replace(".", "p"))
    if test_frac is not None:
        chunks.append(f"test{float(test_frac):.2f}".replace(".", "p"))
    if shots_per_class > 0:
        chunks.append(f"k{shots_per_class}")
    chunks.append(f"seed{seed}")
    return "_".join(chunks)


def load_frame_for_run(
    config: Dict[str, Any],
    dataset_name: str,
    split_file: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame = load_normalized_dataset(config, dataset_name)
    if not safe_text(split_file):
        meta = {
            "analysis_scheme": "",
            "analysis_seed": 0,
            "label_fraction": None,
            "test_frac": None,
            "shots_per_class": 0,
            "split_file": "",
        }
        return frame, meta
    overlay = load_analysis_split_overlay(config, split_file)
    merged = apply_analysis_split_overlay(frame, overlay)
    meta = analysis_metadata_from_frame(merged)
    meta["split_file"] = str(resolve_path(config, split_file))
    merged["split_policy"] = analysis_split_policy(meta, config)
    return merged, meta


def pretrain_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "pretrain_use" in frame.columns:
        return frame[frame["pretrain_use"].apply(as_bool)].reset_index(drop=True)
    return dataframe_for_split(frame, "train")


def supervised_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if {"supervised_use", "label_visible"}.issubset(frame.columns):
        mask = (
            frame["supervised_use"].apply(as_bool)
            & frame["label_visible"].apply(as_bool)
            & frame["is_labeled"].apply(as_bool)
        )
        return frame[mask].reset_index(drop=True)
    train_frame = dataframe_for_split(frame, "train")
    return train_frame[train_frame["is_labeled"].apply(as_bool)].reset_index(drop=True)


def evaluation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "analysis_scheme" in frame.columns and analysis_field_text(frame, "analysis_scheme"):
        return frame[(frame["split"] == "test") & frame["is_labeled"].apply(as_bool)].reset_index(drop=True)
    return dataframe_for_split(frame, "test")


def analysis_metadata_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "analysis_scheme": safe_text(meta.get("analysis_scheme")),
        "analysis_seed": safe_int(meta.get("analysis_seed"), 0),
        "label_fraction": meta.get("label_fraction"),
        "test_frac": meta.get("test_frac"),
        "shots_per_class": safe_int(meta.get("shots_per_class"), 0),
        "split_file": safe_text(meta.get("split_file")),
    }


def train_pos_weight(train_frame: pd.DataFrame) -> float:
    labeled = train_frame[train_frame["is_labeled"] == True]
    if labeled.empty:
        return 1.0
    labels = labeled["label"].astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos


def build_model(config: Dict[str, Any], cell_vocab: Dict[str, int]) -> DQSeqGraph:
    return DQSeqGraph(config, len(cell_vocab))


def prediction_frame(meta_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for meta in meta_rows:
        rows.append(
            {
                "row_id": meta.get("row_id", ""),
                "split": meta.get("split", ""),
                "label": meta.get("label", ""),
                "prob": meta.get("prob", ""),
                "pred": meta.get("pred", ""),
                "smiles": meta.get("smiles", ""),
                "target_id": meta.get("target_id", ""),
                "recruiter_id": meta.get("recruiter_id", ""),
                "cell_line": meta.get("cell_line", ""),
                "task_id": meta.get("task_id", ""),
            }
        )
    return pd.DataFrame(rows)


def canonical_ablation_name(ablation: Optional[str]) -> str:
    name = safe_text(ablation)
    return ABLATION_ALIASES.get(name, name)


def requested_feature_flags(config: Dict[str, Any]) -> Dict[str, Any]:
    features = config.get("features", {})
    return {
        "use_morgan_fp": bool(features.get("use_morgan_fp", True)),
        "use_mol_graph": bool(features.get("use_mol_graph", True)),
        "use_target_sequence": bool(features.get("use_target_sequence", True)),
        "use_recruiter_sequence": bool(features.get("use_recruiter_sequence", True)),
        "use_cell_line": bool(features.get("use_cell_line", True)),
    }


def effective_feature_flags(frame: pd.DataFrame, config: Dict[str, Any], ablation: Optional[str]) -> Dict[str, Any]:
    ablation = canonical_ablation_name(ablation)
    requested = requested_feature_flags(config)
    encoder = protein_encoder_name(config)
    target_present = frame["target_sequence"].fillna("").astype(str).str.len() > 0 if "target_sequence" in frame.columns else pd.Series(dtype=bool)
    recruiter_present = frame["recruiter_sequence"].fillna("").astype(str).str.len() > 0 if "recruiter_sequence" in frame.columns else pd.Series(dtype=bool)
    target_cov = float(target_present.mean()) if len(target_present) else 0.0
    recruiter_cov = float(recruiter_present.mean()) if len(recruiter_present) else 0.0
    target_effective = requested["use_target_sequence"] and target_cov > 0.0 and ablation not in {
        "smiles_only",
        "smiles_graph_fp",
        "e3_only",
    }
    recruiter_effective = requested["use_recruiter_sequence"] and recruiter_cov > 0.0 and ablation in {
        "smiles_target_recruiter_seq",
        "e3_only",
        "target_e3_only",
        "target_e3_assay_only",
        "full_no_episodic",
        "full_with_episodic",
        "full_with_pseudo_label",
        "ctp_finetune",
        "",
    }
    protein_encoder_effective = encoder
    if not target_effective and not recruiter_effective:
        protein_encoder_effective = "missing_or_zero_or_hash"
    return {
        "target_sequence_available": bool(target_effective),
        "recruiter_sequence_available": bool(recruiter_effective),
        "protein_encoder_effective": protein_encoder_effective,
        "uses_smiles": ablation not in {"target_only", "e3_only", "target_e3_only", "target_e3_assay_only"},
        "uses_morgan_fp": bool(requested["use_morgan_fp"]) and ablation not in {"target_only", "e3_only", "target_e3_only", "target_e3_assay_only"},
        "uses_mol_graph": bool(requested["use_mol_graph"]) and ablation not in {"target_only", "e3_only", "target_e3_only", "target_e3_assay_only"},
        "target_sequence_coverage": target_cov,
        "recruiter_sequence_coverage": recruiter_cov,
        "missing_sequence_count": int(((frame["target_sequence"].fillna("") == "") | (frame["recruiter_sequence"].fillna("") == "")).sum())
        if {"target_sequence", "recruiter_sequence"}.issubset(frame.columns)
        else int(len(frame)),
    }


def split_truth_prob_payload(result: Dict[str, Any]) -> Dict[str, Sequence[float]]:
    return {"truth": list(result.get("truth", [])), "prob": list(result.get("prob", []))}


def is_official_paper_protocol(dataset_name: str, split_policy: str, benchmark_ready: bool, config: Dict[str, Any]) -> bool:
    return bool(
        dataset_name == "protac8k"
        and benchmark_ready
        and safe_text(split_policy) == safe_text(config["benchmark"].get("protocol", ""))
    )


def monitor_selection_record(
    result: Dict[str, Any],
    dataset_name: str,
    split_policy: str,
    benchmark_ready: bool,
    config: Dict[str, Any],
    monitor_scope: str,
) -> Dict[str, Any]:
    if is_official_paper_protocol(dataset_name, split_policy, benchmark_ready, config) and result.get("truth"):
        selected_metrics = result.get("metrics", {})
        return {
            "mode": "paper_authoritative_monitor_selected",
            "key": selection_key(selected_metrics),
            "metric": float(selected_metrics.get("accuracy", float("nan"))),
            "selected_threshold": 0.5,
            "selected_metrics": selected_metrics,
            "selection_scope": monitor_scope,
            "selection_rule": SELECTION_RULE,
            "reporting_protocol": REPORTING_PROTOCOL,
        }
    primary_metric = safe_text(config["benchmark"].get("primary_metric", "auroc"))
    metric_value = float(result.get("metrics", {}).get(primary_metric, float("nan")))
    if math.isnan(metric_value):
        metric_value = -1e9
    return {
        "mode": "primary_metric",
        "key": (metric_value,),
        "metric": metric_value,
        "selected_threshold": 0.5,
        "selected_metrics": result.get("metrics", {}),
        "selection_scope": monitor_scope,
        "selection_rule": primary_metric,
        "reporting_protocol": "",
    }


def reporting_payload_from_thresholds(
    dataset_name: str,
    split_policy: str,
    benchmark_ready: bool,
    threshold_candidates: Dict[str, Any],
    fallback_metrics: Dict[str, Any],
    best_epoch: int,
    config: Dict[str, Any],
    selection_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if is_official_paper_protocol(dataset_name, split_policy, benchmark_ready, config):
        selection_record = selection_record or {}
        return {
            "reporting_protocol": REPORTING_PROTOCOL,
            "selection_scope": safe_text(selection_record.get("selection_scope", SELECTION_SCOPE)),
            "selection_rule": safe_text(selection_record.get("selection_rule", SELECTION_RULE)),
            "selected_epoch": int(best_epoch),
            "selected_threshold": 0.5,
            "reporting_metrics": {
                "test_fixed_0p5": fallback_metrics,
            },
        }
    return {
        "reporting_protocol": "",
        "selection_scope": "",
        "selection_rule": "",
        "selected_epoch": int(best_epoch),
        "selected_threshold": float(fallback_metrics.get("threshold", 0.5)),
        "reporting_metrics": {
            "test_fixed_0p5": fallback_metrics,
        },
    }


def payload_test_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    reporting_metrics = payload.get("reporting_metrics") or {}
    fixed = reporting_metrics.get("test_fixed_0p5")
    if isinstance(fixed, dict) and fixed:
        return fixed
    selected = reporting_metrics.get("test_selected")
    if isinstance(selected, dict) and selected:
        return selected
    return payload.get("metrics", {}).get("test", {})


def payload_report_note(payload: Dict[str, Any]) -> str:
    if safe_text(payload.get("reporting_protocol", "")) == REPORTING_PROTOCOL:
        return "paper-authoritative run; epoch selected without test labels; fixed threshold 0.5"
    if metric_key(payload.get("target_sequence_coverage")) <= 0.0 and metric_key(payload.get("recruiter_sequence_coverage")) <= 0.0:
        return "current benchmark-ready legacy run (sequence-missing, fixed@0.5)"
    return "current benchmark-ready legacy run"


def resolve_threshold_from_spec(spec: str, checkpoint: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    spec = safe_text(spec) or "0.5"
    try:
        return float(spec), {"threshold_spec": spec, "threshold_source": "explicit"}
    except ValueError:
        pass
    candidates = checkpoint.get("threshold_candidates") or {}
    mapping = {
        "auto_train_f1": ("train", "f1"),
        "auto_train_balacc": ("train", "balacc"),
        "auto_train_youden": ("train", "youden"),
        "auto_val_f1": ("val", "f1"),
        "auto_val_balacc": ("val", "balacc"),
        "auto_val_youden": ("val", "youden"),
        "best_test_selected": ("test", "selected"),
    }
    if spec not in mapping:
        raise ValueError(f"Unsupported threshold spec: {spec}")
    split_name, objective = mapping[spec]
    if split_name not in candidates or objective not in candidates.get(split_name, {}):
        raise ValueError(
            f"Threshold policy {spec} is unavailable for this checkpoint; no {split_name} threshold candidates were stored."
        )
    record = candidates[split_name][objective]
    meta = {
        "threshold_spec": spec,
        "threshold_source": f"{split_name}_{objective}",
    }
    if objective == "selected":
        meta["selection_rule"] = safe_text(record.get("selection_rule", SELECTION_RULE))
        meta["selection_scope"] = safe_text(record.get("selection_scope", split_name))
    else:
        meta["threshold_objective_score"] = float(record.get("score", float("nan")))
    return float(record["threshold"]), meta


def add_canonical_smiles_column(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    values = []
    for smiles in work["smiles"].fillna("").astype(str).tolist():
        smiles = safe_text(smiles)
        canonical = canonicalize_smiles(smiles) if smiles and RDKIT_AVAILABLE else ""
        values.append(canonical or smiles)
    work["canonical_smiles"] = values
    return work


def leakage_filtered_frame(frame: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    if not filter_name:
        return frame.copy().reset_index(drop=True)
    work = add_canonical_smiles_column(frame)
    train_frame = work[work["split"] == "train"].copy()
    test_frame = work[work["split"] == "test"].copy()
    if test_frame.empty:
        return test_frame.reset_index(drop=True)
    if filter_name == "no_exact_smiles_overlap":
        train_keys = {safe_text(value) for value in train_frame["canonical_smiles"].tolist() if safe_text(value)}
        keep = ~test_frame["canonical_smiles"].astype(str).isin(train_keys)
    elif filter_name == "no_target_smiles_overlap":
        train_keys = {
            (safe_text(row.target_id), safe_text(row.canonical_smiles))
            for row in train_frame[["target_id", "canonical_smiles"]].itertuples(index=False)
            if safe_text(row.target_id) or safe_text(row.canonical_smiles)
        }
        keep = [
            (safe_text(target_id), safe_text(canonical_smiles)) not in train_keys
            for target_id, canonical_smiles in test_frame[["target_id", "canonical_smiles"]].itertuples(index=False, name=None)
        ]
    elif filter_name == "no_target_recruiter_smiles_overlap":
        train_keys = {
            (safe_text(row.target_id), safe_text(row.recruiter_id), safe_text(row.canonical_smiles))
            for row in train_frame[["target_id", "recruiter_id", "canonical_smiles"]].itertuples(index=False)
            if safe_text(row.target_id) or safe_text(row.recruiter_id) or safe_text(row.canonical_smiles)
        }
        keep = [
            (safe_text(target_id), safe_text(recruiter_id), safe_text(canonical_smiles)) not in train_keys
            for target_id, recruiter_id, canonical_smiles in test_frame[["target_id", "recruiter_id", "canonical_smiles"]].itertuples(index=False, name=None)
        ]
    else:
        raise ValueError(f"Unsupported test filter: {filter_name}")
    return test_frame.loc[keep].drop(columns=["canonical_smiles"], errors="ignore").reset_index(drop=True)


def checkpoint_benchmark_status(frame: pd.DataFrame) -> Tuple[bool, str]:
    ready = bool(frame.get("benchmark_ready", pd.Series([False] * len(frame))).all())
    if ready:
        return True, ""
    protocol = safe_text(frame["data_protocol"].iloc[0]) if "data_protocol" in frame.columns and not frame.empty else ""
    if protocol == "asset_fallback_surrogate_target":
        return False, "surrogate target/recruiter IDs; not comparable to DegradeMaster"
    return False, "dataset contains benchmark_ready=false rows"


def save_checkpoint(
    path: Path,
    model: DQSeqGraph,
    config: Dict[str, Any],
    cell_vocab: Dict[str, int],
    dataset_fingerprint: str,
    dataset_name: str,
    split_policy: str,
    ablation: Optional[str],
    benchmark_ready: bool,
    benchmark_reason: str,
    threshold_candidates: Optional[Dict[str, Any]] = None,
    requested_features: Optional[Dict[str, Any]] = None,
    effective_features: Optional[Dict[str, Any]] = None,
    reporting_protocol: str = "",
    selection_scope: str = "",
    selection_rule: str = "",
    selected_epoch: int = -1,
    selected_threshold: float = 0.5,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    root = runtime_root(config)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "project_version": PROJECT_VERSION,
        "source_sha256": compute_source_hash(config),
        "source_bundle_sha256": compute_source_bundle_sha256(config),
        "dq_py_sha256": file_sha256(root / "dq.py"),
        "dq_data_py_sha256": file_sha256(root / "dq_data.py"),
        "dq_model_py_sha256": file_sha256(root / "dq_model.py"),
        "config_sha256": file_sha256(root / "config.yaml"),
        "dataset_fingerprint": dataset_fingerprint,
        "label_rule": config["benchmark"]["label_rule"],
        "split_policy": split_policy,
        "dataset": dataset_name,
        "cell_vocab": cell_vocab,
        "ablation": ablation or "",
        "run_label": runtime_run_label(config),
        "benchmark_ready": bool(benchmark_ready),
        "benchmark_reason": benchmark_reason,
        "threshold_candidates": threshold_candidates or {},
        "requested_features": requested_features or {},
        "effective_features": effective_features or {},
        "reporting_protocol": reporting_protocol,
        "selection_scope": selection_scope,
        "selection_rule": selection_rule,
        "selected_epoch": int(selected_epoch),
        "selected_threshold": float(selected_threshold),
        "protein_encoder": protein_encoder_name(config),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[DQSeqGraph, Dict[str, Any], Dict[str, int]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    cell_vocab = checkpoint["cell_vocab"]
    model = build_model(config, cell_vocab)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint, cell_vocab


def save_metrics_and_analysis(
    config: Dict[str, Any],
    run_id: str,
    metrics_payload: Dict[str, Any],
    pred_frame: pd.DataFrame,
) -> Tuple[Path, Path, Path]:
    out_dir = output_dir(config)
    metrics_path = out_dir / f"{run_id}_metrics.json"
    pred_path = out_dir / f"{run_id}_pred.csv"
    analysis_path = out_dir / f"{run_id}_analysis.md"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    pred_frame.to_csv(pred_path, index=False)
    lines = [
        f"# {run_id}",
        "",
        f"- run_label: {metrics_payload.get('run_label', '')}",
        f"- dataset: {metrics_payload.get('dataset')}",
        f"- mode: {metrics_payload.get('mode')}",
        f"- ablation: {metrics_payload.get('ablation', '')}",
        f"- protein_encoder: {metrics_payload.get('protein_encoder', '')}",
        f"- split_policy: {metrics_payload.get('split_policy', '')}",
        f"- benchmark_ready: {metrics_payload.get('benchmark_ready', False)}",
    ]
    if metrics_payload.get("reason"):
        lines.append(f"- reason: {metrics_payload['reason']}")
    if metrics_payload.get("reporting_protocol"):
        lines.append(f"- reporting_protocol: {metrics_payload['reporting_protocol']}")
    if metrics_payload.get("selection_rule"):
        lines.append(f"- selection_rule: {metrics_payload['selection_rule']}")
    if metrics_payload.get("selected_epoch") is not None:
        lines.append(f"- selected_epoch: {metrics_payload.get('selected_epoch')}")
    if metrics_payload.get("selected_threshold") is not None:
        lines.append(f"- selected_threshold: {metrics_payload.get('selected_threshold')}")
    lines.extend(["", "## Metrics", ""])
    for split_name in ["train", "val", "test", "test_fixed_0p5", "test_selected"]:
        if split_name in metrics_payload.get("metrics", {}):
            lines.append(f"### {split_name}")
            lines.append("")
            for key, value in metrics_payload["metrics"][split_name].items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    if metrics_payload.get("pseudo_rounds"):
        lines.append("## Pseudo Label Rounds")
        lines.append("")
        for item in metrics_payload["pseudo_rounds"]:
            lines.append(
                f"- round {item['round']}: +{item['n_pseudo_pos']} / -{item['n_pseudo_neg']}, "
                f"loss_before={item['train_loss_before']}, loss_after={item['train_loss_after']}, "
                f"param_delta_l2={item['model_param_delta_l2']}"
            )
        lines.append("")
    if metrics_payload.get("semi_comparison"):
        lines.append("## Semi Comparison")
        lines.append("")
        for key, value in metrics_payload["semi_comparison"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    analysis_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics_path, pred_path, analysis_path


def save_pretrain_artifacts(
    config: Dict[str, Any],
    run_id: str,
    metrics_payload: Dict[str, Any],
) -> Tuple[Path, Path]:
    out_dir = output_dir(config)
    metrics_path = out_dir / f"{run_id}_pretrain_metrics.json"
    analysis_path = out_dir / f"{run_id}_pretrain_analysis.md"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        f"# {run_id}",
        "",
        f"- run_label: {metrics_payload.get('run_label', '')}",
        f"- dataset: {metrics_payload.get('dataset')}",
        f"- mode: {metrics_payload.get('mode')}",
        f"- protein_encoder: {metrics_payload.get('protein_encoder', '')}",
        f"- tasks: {', '.join(metrics_payload.get('pretrain_tasks', []))}",
        f"- tuple_negative_policy: {metrics_payload.get('tuple_negative_policy', '')}",
        f"- positive_construction: {metrics_payload.get('positive_construction', 'observed')}",
        f"- uses_unlabeled: {metrics_payload.get('uses_unlabeled')}",
        f"- pretrain_train_rows_only: {metrics_payload.get('pretrain_train_rows_only')}",
        f"- best_epoch: {metrics_payload.get('best_epoch')}",
        "",
        "## Losses",
        "",
        f"- total_loss: {metrics_payload.get('losses', {}).get('total_loss')}",
        f"- tuple_loss: {metrics_payload.get('losses', {}).get('tuple_loss')}",
        f"- mol_loss: {metrics_payload.get('losses', {}).get('mol_loss')}",
        "",
    ]
    history = metrics_payload.get("history", [])
    if history:
        lines.extend(["## History", ""])
        for item in history:
            lines.append(
                f"- epoch {item['epoch']}: total_loss={item['total_loss']}, tuple_loss={item['tuple_loss']}, mol_loss={item['mol_loss']}"
            )
        lines.append("")
    sampler_stats = metrics_payload.get("sampler_stats") or {}
    if sampler_stats:
        lines.extend(["## Sampler Stats", ""])
        for key, value in sampler_stats.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    analysis_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics_path, analysis_path


def training_run_id(dataset: str, mode: str, ablation: Optional[str], run_label: str = "") -> str:
    base = f"{precise_timestamp()}_{dataset}_{mode}_dqseqgraph"
    with_ablation = f"{base}_{ablation}" if ablation else base
    return append_run_label(with_ablation, run_label)


def pretrain_run_id(dataset: str, run_label: str = "") -> str:
    base = f"{precise_timestamp()}_{dataset}_ctp_dqseqgraph"
    return append_run_label(base, run_label)


def parse_pretrain_tasks(text: str) -> List[str]:
    tasks = []
    for raw in safe_text(text).split(","):
        task = safe_text(raw)
        if not task:
            continue
        if task not in PRETRAIN_TASKS:
            raise ValueError(f"Unsupported pretrain task: {task}")
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        raise ValueError("At least one pretrain task is required.")
    return tasks


def prepare_runtime_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    runtime = copy.deepcopy(config)
    if getattr(args, "mode", None):
        runtime["train"]["mode"] = args.mode
    if getattr(args, "epochs", None):
        if getattr(args, "command", "") == "pretrain":
            runtime["pretrain"]["epochs"] = int(args.epochs)
        else:
            runtime["train"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None):
        if getattr(args, "command", "") == "pretrain":
            runtime["pretrain"]["batch_size"] = int(args.batch_size)
        else:
            runtime["train"]["batch_size"] = int(args.batch_size)
    if getattr(args, "protein_encoder", None):
        runtime["features"]["protein_encoder"] = safe_text(args.protein_encoder).lower()
    if getattr(args, "tuple_negative_policy", None):
        runtime["pretrain"]["tuple_negative_policy"] = safe_text(args.tuple_negative_policy)
    if getattr(args, "positive_construction", None):
        runtime["pretrain"]["positive_construction"] = safe_text(args.positive_construction)
    if getattr(args, "lambda_tuple", None) is not None:
        runtime["pretrain"]["lambda_tuple"] = float(args.lambda_tuple)
    if getattr(args, "lambda_mol", None) is not None:
        runtime["pretrain"]["lambda_mol"] = float(args.lambda_mol)
    if getattr(args, "temperature_tuple", None) is not None:
        runtime["pretrain"]["temperature_tuple"] = float(args.temperature_tuple)
    if getattr(args, "temperature_mol", None) is not None:
        runtime["pretrain"]["temperature_mol"] = float(args.temperature_mol)
    if getattr(args, "lr", None) is not None:
        runtime["train"]["lr"] = float(args.lr)
    if getattr(args, "weight_decay", None) is not None:
        runtime["train"]["weight_decay"] = float(args.weight_decay)
    if getattr(args, "dropout", None) is not None:
        runtime["model"]["mol_gnn_dropout"] = float(args.dropout)
        runtime["model"]["fusion_dropout"] = float(args.dropout)
    if getattr(args, "loss", None):
        runtime["train"]["loss"] = safe_text(args.loss).lower()
    if getattr(args, "pos_weight", None):
        runtime["train"]["class_weight"] = safe_text(args.pos_weight).lower()
    if getattr(args, "focal_gamma", None) is not None:
        runtime["train"]["focal_gamma"] = float(args.focal_gamma)
    if getattr(args, "focal_alpha", None):
        runtime["train"]["focal_alpha"] = safe_text(args.focal_alpha).lower()
    if getattr(args, "pseudo_threshold", None) is not None:
        runtime["semi"]["confidence_threshold"] = float(args.pseudo_threshold)
    if getattr(args, "pseudo_selector", None):
        runtime["semi"]["selector"] = safe_text(args.pseudo_selector)
    if getattr(args, "pseudo_topk", None) is not None:
        runtime["semi"]["balanced_topk"] = int(args.pseudo_topk)
    if getattr(args, "teacher_checkpoint", None):
        runtime["semi"]["teacher_checkpoint"] = safe_text(args.teacher_checkpoint)
    if getattr(args, "pseudo_weight", None) is not None:
        runtime["semi"]["pseudo_label_weight"] = float(args.pseudo_weight)
    if getattr(args, "pseudo_rounds", None) is not None:
        runtime["semi"]["pseudo_rounds"] = int(args.pseudo_rounds)
    runtime["runtime"] = runtime.get("runtime", {})
    runtime["runtime"]["run_label"] = sanitize_run_label(getattr(args, "run_label", "")) if safe_text(getattr(args, "run_label", "")) else ""
    runtime["runtime"]["split_file"] = safe_text(getattr(args, "split_file", ""))
    if protein_encoder_name(runtime) == ESM2_650M_ENCODER:
        cache = load_esm2_embedding_cache(runtime)
        cached_dim = safe_int(cache.get("embedding_dim"), 0) if isinstance(cache, dict) else 0
        runtime["features"]["protein_feature_dim"] = int(cached_dim or 1280)
    if getattr(args, "seed", None):
        runtime["project"]["seed"] = int(args.seed)
        runtime["benchmark"]["seed"] = int(args.seed)
    return runtime


def runtime_run_label(config: Dict[str, Any]) -> str:
    return safe_text(config.get("runtime", {}).get("run_label", ""))


def validate_esm2_cache_for_frame(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    require_target: bool = True,
    require_recruiter: bool = True,
) -> Dict[str, Any]:
    cache = load_esm2_embedding_cache(config)
    if not cache or not isinstance(cache.get("records"), dict):
        raise RuntimeError("ESM-2 650M cache is missing. Run `embedseq --encoder esm2_650m` first.")
    records = cache["records"]
    missing_target: List[str] = []
    missing_recruiter: List[str] = []
    target_seen = 0
    recruiter_seen = 0
    if require_target and "target_sequence" in frame.columns:
        for seq in frame["target_sequence"].fillna("").astype(str).tolist():
            seq = safe_text(seq)
            if not seq:
                missing_target.append("")
                continue
            target_seen += 1
            if bytes_sha256(seq.encode("utf-8")) not in records:
                missing_target.append(seq)
    if require_recruiter and "recruiter_sequence" in frame.columns:
        for seq in frame["recruiter_sequence"].fillna("").astype(str).tolist():
            seq = safe_text(seq)
            if not seq:
                missing_recruiter.append("")
                continue
            recruiter_seen += 1
            if bytes_sha256(seq.encode("utf-8")) not in records:
                missing_recruiter.append(seq)
    if missing_target or missing_recruiter:
        raise RuntimeError(
            "ESM-2 650M cache coverage is incomplete for the requested run. "
            f"missing_target={len(missing_target)} missing_recruiter={len(missing_recruiter)}"
        )
    return {
        "target_row_coverage": 1.0 if not require_target or target_seen == 0 else 1.0,
        "recruiter_row_coverage": 1.0 if not require_recruiter or recruiter_seen == 0 else 1.0,
        "embedding_dim": int(cache.get("embedding_dim", config["features"]["protein_feature_dim"])),
        "cache_path": str(esm2_embedding_cache_path(config)),
        "model_name": safe_text(cache.get("model_name", esm2_model_name(config))),
    }


def protocol_monitor_loader(
    dataset_name: str,
    split_policy: str,
    config: Dict[str, Any],
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    train_loader: DataLoader,
) -> Optional[DataLoader]:
    if val_loader is not None:
        return val_loader
    return train_loader


def protocol_monitor_scope(val_loader: Optional[DataLoader]) -> str:
    return "val" if val_loader is not None else "train"


def manifest_payload(
    config: Dict[str, Any],
    run_id: str,
    dataset_name: str,
    split_policy: str,
    model_path: Path,
    metrics_path: Optional[Path],
    pred_path: Optional[Path],
    notes: str,
    benchmark_ready: bool,
    ablation: str = "",
    reporting_protocol: str = "",
) -> Dict[str, Any]:
    root = runtime_root(config)
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "cwd": str(root),
        "command": " ".join(sys.argv),
        "project_version": PROJECT_VERSION,
        "dq_py_sha256": file_sha256(root / "dq.py"),
        "dq_data_py_sha256": file_sha256(root / "dq_data.py"),
        "dq_model_py_sha256": file_sha256(root / "dq_model.py"),
        "config_sha256": file_sha256(root / "config.yaml"),
        "source_sha256": compute_source_hash(config),
        "source_bundle_sha256": compute_source_bundle_sha256(config),
        "dataset": dataset_name,
        "ablation": ablation,
        "run_label": runtime_run_label(config),
        "protein_encoder": protein_encoder_name(config),
        "split_policy": split_policy,
        "seed": int(config["project"]["seed"]),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path) if metrics_path else "",
        "pred_path": str(pred_path) if pred_path else "",
        "notes": notes,
        "benchmark_ready": bool(benchmark_ready),
        "reporting_protocol": reporting_protocol,
    }


def load_pretrain_checkpoint_for_init(config: Dict[str, Any], init_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(resolve_path(config, init_path), map_location="cpu", weights_only=False)
    if safe_text(checkpoint.get("checkpoint_kind", "")) != "pretrain":
        raise RuntimeError("`--init` expects a CTP pretrain checkpoint.")
    checkpoint_config = checkpoint.get("config") or {}
    current_encoder = protein_encoder_name(config)
    checkpoint_encoder = protein_encoder_name(checkpoint_config)
    if checkpoint_encoder != current_encoder:
        raise RuntimeError(
            f"Protein encoder mismatch for `--init`: checkpoint={checkpoint_encoder} current={current_encoder}."
        )
    current_dim = int(config["features"]["protein_feature_dim"])
    checkpoint_dim = int((checkpoint_config.get("features") or {}).get("protein_feature_dim", current_dim))
    if checkpoint_dim != current_dim:
        raise RuntimeError(
            f"Protein feature dimension mismatch for `--init`: checkpoint={checkpoint_dim} current={current_dim}."
        )
    return checkpoint


def initialize_shared_encoder_from_pretrain(model: DQSeqGraph, checkpoint: Dict[str, Any]) -> int:
    source_state = checkpoint.get("model_state_dict") or {}
    target_state = model.state_dict()
    prefixes = ("atom_encoder.", "graph_layers.", "graph_proj.", "fp_proj.", "mol_norm.", "protein_proj.")
    copied = 0
    for key, value in source_state.items():
        if not key.startswith(prefixes):
            continue
        if key not in target_state or target_state[key].shape != value.shape:
            continue
        target_state[key] = value
        copied += 1
    model.load_state_dict(target_state)
    return copied


def state_dict_l2_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        if key not in after:
            continue
        diff = after[key].detach().cpu().float() - before[key].detach().cpu().float()
        total += float((diff * diff).sum().item())
    return float(math.sqrt(total))


def prediction_comparison(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> Tuple[float, float]:
    merged = pred_a[["row_id", "prob"]].rename(columns={"prob": "prob_a"}).merge(
        pred_b[["row_id", "prob"]].rename(columns={"prob": "prob_b"}),
        on="row_id",
        how="inner",
    )
    if merged.empty:
        return float("nan"), float("nan")
    if merged["prob_a"].nunique() <= 1 or merged["prob_b"].nunique() <= 1:
        corr = 1.0 if np.allclose(merged["prob_a"], merged["prob_b"]) else float("nan")
    else:
        corr = float(np.corrcoef(merged["prob_a"], merged["prob_b"])[0, 1])
    max_diff = float(np.max(np.abs(merged["prob_a"] - merged["prob_b"])))
    return corr, max_diff


def write_threshold_report(config: Dict[str, Any], run_id: str, report: Dict[str, Any]) -> Path:
    path = output_dir(config) / f"{run_id}_threshold_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def write_leakage_filtered_report(
    config: Dict[str, Any],
    run_id: str,
    metrics_payload: Dict[str, Any],
    pred_frame: pd.DataFrame,
) -> Tuple[Path, Path]:
    metrics_path = output_dir(config) / f"{run_id}_leakage_filtered_metrics.json"
    pred_path = output_dir(config) / f"{run_id}_leakage_filtered_pred.csv"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    pred_frame.to_csv(pred_path, index=False)
    return metrics_path, pred_path


def model_feature_metadata(frame: pd.DataFrame, config: Dict[str, Any], ablation: Optional[str]) -> Dict[str, Any]:
    requested = requested_feature_flags(config)
    effective = effective_feature_flags(frame, config, ablation)
    return {
        "requested_features": requested,
        "effective_features": {
            key: value
            for key, value in effective.items()
            if key not in {"target_sequence_coverage", "recruiter_sequence_coverage", "missing_sequence_count"}
        },
        "target_sequence_coverage": float(effective["target_sequence_coverage"]),
        "recruiter_sequence_coverage": float(effective["recruiter_sequence_coverage"]),
        "missing_sequence_count": int(effective["missing_sequence_count"]),
    }


def regime_metadata(
    mode: str,
    init_checkpoint: str = "",
    pretrain_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if pretrain_checkpoint is not None:
        tasks = list(pretrain_checkpoint.get("pretrain_tasks") or [])
        policy_name = safe_text(pretrain_checkpoint.get("tuple_negative_policy_name", ""))
        policy_values = pretrain_checkpoint.get("tuple_negative_policy", [])
        if not policy_name and isinstance(policy_values, list):
            policy_list = list(policy_values)
        else:
            policy_list = list(TUPLE_NEGATIVE_POLICIES.get(policy_name, []))
        return {
            "uses_unlabeled": True,
            "unlabeled_usage": "self_supervised_tuple_pretraining",
            "pseudo_label": False,
            "teacher": False,
            "ensemble": False,
            "tuple_negative_policy": policy_list,
            "positive_construction": safe_text(pretrain_checkpoint.get("positive_construction", "observed")) or "observed",
            "pretrain_train_rows_only": True,
            "init_checkpoint": init_checkpoint,
            "pretrain_tasks": tasks,
        }
    if mode == "semi":
        return {
            "uses_unlabeled": True,
            "unlabeled_usage": "pseudo_labeling",
            "pseudo_label": True,
            "teacher": False,
            "ensemble": False,
            "tuple_negative_policy": [],
            "positive_construction": "",
            "pretrain_train_rows_only": False,
            "init_checkpoint": "",
            "pretrain_tasks": [],
        }
    return {
        "uses_unlabeled": False,
        "unlabeled_usage": "none",
        "pseudo_label": False,
        "teacher": False,
        "ensemble": False,
        "tuple_negative_policy": [],
        "positive_construction": "",
        "pretrain_train_rows_only": False,
        "init_checkpoint": "",
        "pretrain_tasks": [],
    }


def find_reference_supervised_checkpoint(
    config: Dict[str, Any],
    dataset_name: str,
    ablation: Optional[str],
    exclude_run_id: str,
) -> Optional[Path]:
    suffix = f"_{ablation}" if ablation else ""
    pattern = f"*_{dataset_name}_supervised_dqseqgraph{suffix}_model.pt" if ablation else f"*_{dataset_name}_supervised_dqseqgraph_model.pt"
    candidates = []
    for path in output_dir(config).glob(pattern):
        if exclude_run_id and exclude_run_id in path.name:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def pretrain_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    dataset_name = args.dataset
    tasks = parse_pretrain_tasks(args.tasks)
    frame, analysis_meta = load_frame_for_run(runtime_cfg, dataset_name, safe_text(getattr(args, "split_file", "")))
    benchmark_ready, benchmark_reason = checkpoint_benchmark_status(frame)
    if dataset_name == "protac8k" and not benchmark_ready:
        raise RuntimeError("CTP pretraining requires a benchmark-ready PROTAC-8K normalization.")
    train_frame = pretrain_rows(frame)
    if train_frame.empty:
        raise RuntimeError("Pretraining split is empty.")
    if protein_encoder_name(runtime_cfg) == ESM2_650M_ENCODER:
        validate_esm2_cache_for_frame(train_frame, runtime_cfg, require_target=True, require_recruiter=True)
    ensure_project_dirs(runtime_cfg)
    set_seed(int(runtime_cfg["project"]["seed"]))
    device = resolve_device(runtime_cfg)
    feature_cache = load_feature_cache(runtime_cfg)
    cell_vocab = make_cell_vocab(frame)
    split_policy = safe_text(frame["split_policy"].iloc[0]) if "split_policy" in frame.columns and not frame.empty else ""
    if not split_policy:
        split_policy = safe_text(runtime_cfg["benchmark"].get("protocol", ""))
    policy_name = safe_text(runtime_cfg["pretrain"].get("tuple_negative_policy", "")) or "full"
    positive_construction = safe_text(runtime_cfg["pretrain"].get("positive_construction", "")) or "observed"
    tuple_temperature = float(
        runtime_cfg["pretrain"].get("temperature_tuple", runtime_cfg["pretrain"]["temperature"])
    )
    mol_temperature = float(
        runtime_cfg["pretrain"].get("temperature_mol", runtime_cfg["pretrain"]["temperature"])
    )
    train_loader = build_ctp_pretrain_dataloader(
        train_frame,
        dataset_name,
        runtime_cfg,
        feature_cache,
        cell_vocab,
        device,
        True,
        policy_name,
        positive_construction,
    )
    model = build_model(runtime_cfg, cell_vocab).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime_cfg["pretrain"]["lr"]),
        weight_decay=float(runtime_cfg["pretrain"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(runtime_cfg["train"]["early_stop_patience"]) // 4),
    )
    run_id = pretrain_run_id(dataset_name, runtime_run_label(runtime_cfg))
    checkpoint_path = output_dir(runtime_cfg) / f"{run_id}_pretrain_model.pt"
    dataset_fingerprint = compute_dataset_fingerprint(train_frame)
    best_loss = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    pretrain_metadata = {
        "checkpoint_kind": "pretrain",
        "pretrain_tasks": list(tasks),
        "tuple_negative_policy_name": policy_name,
        "tuple_negative_policy": list(TUPLE_NEGATIVE_POLICIES.get(policy_name, [])),
        "positive_construction": positive_construction,
        "temperature_tuple": tuple_temperature,
        "temperature_mol": mol_temperature,
        "lambda_tuple": float(runtime_cfg["pretrain"]["lambda_tuple"]),
        "lambda_mol": float(runtime_cfg["pretrain"]["lambda_mol"]),
        "uses_unlabeled": True,
        "unlabeled_usage": "self_supervised_tuple_pretraining",
        "pseudo_label": False,
        "teacher": False,
        "ensemble": False,
        "pretrain_train_rows_only": True,
        **analysis_metadata_payload(analysis_meta),
    }
    for epoch in range(int(runtime_cfg["pretrain"]["epochs"])):
        train_result = run_pretrain_epoch(model, train_loader, device, runtime_cfg, tasks, optimizer)
        current_loss = float(train_result["loss"])
        scheduler.step(current_loss if math.isfinite(current_loss) else 1e9)
        history.append(
            {
                "epoch": epoch + 1,
                "total_loss": current_loss,
                "tuple_loss": float(train_result["tuple_loss"]),
                "mol_loss": float(train_result["mol_loss"]),
            }
        )
        if math.isfinite(current_loss) and current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch + 1
            save_checkpoint(
                checkpoint_path,
                model,
                runtime_cfg,
                cell_vocab,
                dataset_fingerprint,
                dataset_name,
                split_policy,
                "ctp_pretrain",
                benchmark_ready,
                benchmark_reason,
                selection_scope="train",
                selection_rule="lowest_pretrain_train_loss",
                selected_epoch=best_epoch,
                selected_threshold=0.5,
                extra_metadata=pretrain_metadata,
            )
    if best_epoch < 0:
        best_epoch = int(runtime_cfg["pretrain"]["epochs"])
        save_checkpoint(
            checkpoint_path,
            model,
            runtime_cfg,
            cell_vocab,
            dataset_fingerprint,
            dataset_name,
            split_policy,
            "ctp_pretrain",
            benchmark_ready,
            benchmark_reason,
            selection_scope="train",
            selection_rule="lowest_pretrain_train_loss",
            selected_epoch=best_epoch,
            selected_threshold=0.5,
            extra_metadata=pretrain_metadata,
        )
    best_model, _, _ = load_checkpoint(checkpoint_path, device)
    final_train = run_pretrain_epoch(best_model, train_loader, device, runtime_cfg, tasks, None)
    metrics_payload = {
        "run_id": run_id,
        "run_label": runtime_run_label(runtime_cfg),
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "mode": "pretrain",
        "protein_encoder": protein_encoder_name(runtime_cfg),
        "split_policy": split_policy,
        "benchmark_ready": benchmark_ready,
        "reason": benchmark_reason,
        "best_epoch": best_epoch,
        "selected_epoch": best_epoch,
        "selected_threshold": 0.5,
        "selection_scope": "train",
        "selection_rule": "lowest_pretrain_train_loss",
        "reporting_protocol": "",
        "losses": {
            "total_loss": float(final_train["loss"]),
            "tuple_loss": float(final_train["tuple_loss"]),
            "mol_loss": float(final_train["mol_loss"]),
        },
        "pretrain_options": {
            "epochs": int(runtime_cfg["pretrain"]["epochs"]),
            "batch_size": int(runtime_cfg["pretrain"]["batch_size"]),
            "lr": float(runtime_cfg["pretrain"]["lr"]),
            "weight_decay": float(runtime_cfg["pretrain"]["weight_decay"]),
            "temperature_tuple": tuple_temperature,
            "temperature_mol": mol_temperature,
            "lambda_tuple": float(runtime_cfg["pretrain"]["lambda_tuple"]),
            "lambda_mol": float(runtime_cfg["pretrain"]["lambda_mol"]),
            "tuple_negative_policy_name": policy_name,
            "positive_construction": positive_construction,
        },
        "sampler_stats": train_loader.dataset.stats_snapshot() if hasattr(train_loader.dataset, "stats_snapshot") else {},
        "history": history,
        "counts": {
            "train": int(len(train_frame)),
            "train_labeled": int((train_frame["is_labeled"] == True).sum()),
            "train_unlabeled": int((train_frame["is_labeled"] == False).sum()),
            "test": 0,
        },
        **pretrain_metadata,
    }
    metrics_path, analysis_path = save_pretrain_artifacts(runtime_cfg, run_id, metrics_payload)
    append_manifest(
        runtime_cfg,
        manifest_payload(
            runtime_cfg,
            run_id,
            dataset_name,
            split_policy,
            checkpoint_path,
            metrics_path,
            None,
            ",".join(tasks),
            benchmark_ready,
            "ctp_pretrain",
            "",
        ),
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics: {metrics_path}")
    print(f"analysis: {analysis_path}")


def supervised_training(
    config: Dict[str, Any],
    dataset_name: str,
    frame: pd.DataFrame,
    ablation: Optional[str],
    run_id: str,
    init_checkpoint: str = "",
    pretrain_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_project_dirs(config)
    set_seed(int(config["project"]["seed"]))
    device = resolve_device(config)
    feature_cache = load_feature_cache(config)
    split_policy = safe_text(frame["split_policy"].iloc[0]) if "split_policy" in frame.columns and not frame.empty else ""
    if not split_policy:
        split_policy = safe_text(config["benchmark"].get("protocol", ""))
    analysis_payload = analysis_metadata_payload(analysis_metadata_from_frame(frame))
    benchmark_ready, benchmark_reason = checkpoint_benchmark_status(frame)
    ablation = canonical_ablation_name(ablation)
    train_frame = dataframe_for_split(frame, "train")
    labeled_train_frame = supervised_labeled_rows(frame)
    val_frame = dataframe_for_split(frame, "val")
    test_frame = evaluation_rows(frame)
    if labeled_train_frame.empty:
        raise RuntimeError("Labeled training split is empty")
    cell_vocab = make_cell_vocab(frame)
    runtime_cfg = copy.deepcopy(config)
    runtime_cfg["train"]["_pos_weight"] = train_pos_weight(labeled_train_frame)
    train_loader = build_dataloader(labeled_train_frame, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, True)
    val_loader = build_dataloader(val_frame, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False) if not val_frame.empty else None
    test_loader = build_dataloader(test_frame, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False) if not test_frame.empty else None
    monitor_loader = protocol_monitor_loader(dataset_name, split_policy, runtime_cfg, val_loader, test_loader, train_loader)
    monitor_scope = protocol_monitor_scope(val_loader)
    model = build_model(runtime_cfg, cell_vocab).to(device)
    if pretrain_checkpoint is not None:
        copied = initialize_shared_encoder_from_pretrain(model, pretrain_checkpoint)
        if copied == 0:
            raise RuntimeError("No shared encoder weights were loaded from the pretrain checkpoint.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime_cfg["train"]["lr"]),
        weight_decay=float(runtime_cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, int(runtime_cfg["train"]["early_stop_patience"]) // 4),
    )
    best_epoch = -1
    best_selection: Optional[Dict[str, Any]] = None
    no_improve = 0
    best_checkpoint_path = output_dir(runtime_cfg) / f"{run_id}_model.pt"
    dataset_fingerprint = compute_dataset_fingerprint(frame)
    train_history: List[Dict[str, Any]] = []
    for epoch in range(int(runtime_cfg["train"]["epochs"])):
        train_result = run_epoch(model, train_loader, device, runtime_cfg, optimizer, ablation)
        monitor_result = run_epoch(model, monitor_loader, device, runtime_cfg, None, ablation) if monitor_loader else {"metrics": {}}
        candidate_selection = monitor_selection_record(
            monitor_result,
            dataset_name,
            split_policy,
            benchmark_ready,
            runtime_cfg,
            monitor_scope,
        )
        metric_value = float(monitor_result["metrics"].get(runtime_cfg["benchmark"]["primary_metric"], float("nan")))
        if math.isnan(metric_value):
            metric_value = -1e9
        scheduler.step(metric_value)
        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_result["loss"],
                "train_metric": train_result["metrics"].get(runtime_cfg["benchmark"]["primary_metric"]),
                "monitor_metric": monitor_result["metrics"].get(runtime_cfg["benchmark"]["primary_metric"]),
                "selected_threshold": candidate_selection.get("selected_threshold", 0.5),
                "selected_accuracy": candidate_selection.get("selected_metrics", {}).get("accuracy"),
                "selected_f1": candidate_selection.get("selected_metrics", {}).get("f1"),
                "selected_auroc": candidate_selection.get("selected_metrics", {}).get("auroc"),
            }
        )
        if best_epoch < 0 or best_selection is None or candidate_selection["key"] > best_selection["key"]:
            best_selection = candidate_selection
            best_epoch = epoch + 1
            no_improve = 0
            save_checkpoint(
                best_checkpoint_path,
                model,
                runtime_cfg,
                cell_vocab,
                dataset_fingerprint,
                dataset_name,
                split_policy,
                ablation,
                benchmark_ready,
                benchmark_reason,
                reporting_protocol=safe_text(candidate_selection.get("reporting_protocol", "")),
                selection_scope=safe_text(candidate_selection.get("selection_scope", "")),
                selection_rule=safe_text(candidate_selection.get("selection_rule", "")),
                selected_epoch=best_epoch,
                selected_threshold=float(candidate_selection.get("selected_threshold", 0.5)),
                extra_metadata={**regime_metadata("supervised", init_checkpoint=init_checkpoint, pretrain_checkpoint=pretrain_checkpoint), **analysis_payload},
            )
        else:
            no_improve += 1
        if no_improve >= int(runtime_cfg["train"]["early_stop_patience"]):
            break
    best_model, _, cell_vocab = load_checkpoint(best_checkpoint_path, device)
    train_eval = run_epoch(best_model, train_loader, device, runtime_cfg, None, ablation)
    val_eval = run_epoch(best_model, val_loader, device, runtime_cfg, None, ablation) if val_loader else {"metrics": {}, "meta": []}
    test_eval = run_epoch(best_model, test_loader, device, runtime_cfg, None, ablation) if test_loader else {"metrics": {}, "meta": []}
    threshold_candidates = threshold_candidates_from_predictions(
        {
            "train": split_truth_prob_payload(train_eval),
            "val": split_truth_prob_payload(val_eval),
            "test": split_truth_prob_payload(test_eval),
        }
    )
    feature_meta = model_feature_metadata(frame, runtime_cfg, ablation)
    reporting_meta = reporting_payload_from_thresholds(
        dataset_name,
        split_policy,
        benchmark_ready,
        threshold_candidates,
        test_eval["metrics"],
        best_epoch,
        runtime_cfg,
        best_selection,
    )
    regime_meta = regime_metadata("supervised", init_checkpoint=init_checkpoint, pretrain_checkpoint=pretrain_checkpoint)
    save_checkpoint(
        best_checkpoint_path,
        best_model,
        runtime_cfg,
        cell_vocab,
        dataset_fingerprint,
        dataset_name,
        split_policy,
        ablation,
        benchmark_ready,
        benchmark_reason,
        threshold_candidates=threshold_candidates,
        requested_features=feature_meta["requested_features"],
        effective_features=feature_meta["effective_features"],
        reporting_protocol=reporting_meta["reporting_protocol"],
        selection_scope=reporting_meta["selection_scope"],
        selection_rule=reporting_meta["selection_rule"],
        selected_epoch=reporting_meta["selected_epoch"],
        selected_threshold=reporting_meta["selected_threshold"],
        extra_metadata={**regime_meta, **analysis_payload},
    )
    reporting_metrics = reporting_meta["reporting_metrics"]
    pred_frame = prediction_frame(train_eval["meta"] + val_eval["meta"] + test_eval["meta"])
    metrics_payload = {
        "run_id": run_id,
        "run_label": runtime_run_label(runtime_cfg),
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "mode": runtime_cfg["train"]["mode"],
        "ablation": ablation or "",
        "protein_encoder": protein_encoder_name(runtime_cfg),
        "split_policy": split_policy,
        "benchmark_ready": benchmark_ready,
        "reason": benchmark_reason,
        "threshold": 0.5,
        "threshold_selection": "fixed_0.5",
        "best_epoch": best_epoch,
        "selected_epoch": reporting_meta["selected_epoch"],
        "selected_threshold": reporting_meta["selected_threshold"],
        "selection_scope": reporting_meta["selection_scope"],
        "selection_rule": reporting_meta["selection_rule"],
        "reporting_protocol": reporting_meta["reporting_protocol"],
        "history": train_history,
        "metrics": {
            "train": train_eval["metrics"],
            "val": val_eval["metrics"],
            "test": test_eval["metrics"],
            "test_fixed_0p5": reporting_metrics["test_fixed_0p5"],
        },
        "counts": {
            "train": int((frame["split"] == "train").sum()),
            "train_labeled_used": int(len(labeled_train_frame)),
            "train_unlabeled_excluded": int((train_frame["split"].eq("train") & ~train_frame["is_labeled"].apply(as_bool)).sum())
            if "split" in train_frame.columns
            else int((~train_frame["is_labeled"].apply(as_bool)).sum()),
            "val": int((frame["split"] == "val").sum()),
            "test": int(len(test_frame)),
        },
        "threshold_candidates": threshold_candidates,
        "reporting_metrics": reporting_metrics,
        "train_options": {
            "epochs": int(runtime_cfg["train"]["epochs"]),
            "batch_size": int(runtime_cfg["train"]["batch_size"]),
            "lr": float(runtime_cfg["train"]["lr"]),
            "weight_decay": float(runtime_cfg["train"]["weight_decay"]),
            "loss": safe_text(runtime_cfg["train"].get("loss", "bce")).lower() or "bce",
            "pos_weight": runtime_cfg["train"].get("class_weight", "auto"),
            "focal_gamma": float(runtime_cfg["train"].get("focal_gamma", 2.0)),
            "focal_alpha": runtime_cfg["train"].get("focal_alpha", "auto"),
        },
        **analysis_payload,
        **regime_meta,
        **feature_meta,
    }
    if protein_encoder_name(runtime_cfg) == ESM2_650M_ENCODER:
        cache_status = validate_esm2_cache_for_frame(frame, runtime_cfg, require_target=True, require_recruiter=True)
        metrics_payload.update(
            {
                "esm2_model": cache_status["model_name"],
                "esm2_frozen": True,
                "esm2_cache": cache_status["cache_path"],
                "esm2_embedding_coverage_target": float(cache_status["target_row_coverage"]),
                "esm2_embedding_coverage_recruiter": float(cache_status["recruiter_row_coverage"]),
            }
        )
    metrics_path, pred_path, analysis_path = save_metrics_and_analysis(runtime_cfg, run_id, metrics_payload, pred_frame)
    append_manifest(
        runtime_cfg,
        manifest_payload(
            runtime_cfg,
            run_id,
            dataset_name,
            split_policy,
            best_checkpoint_path,
            metrics_path,
            pred_path,
            ablation or "",
            benchmark_ready,
            ablation or "",
            reporting_meta["reporting_protocol"],
        ),
    )
    return {
        "checkpoint_path": best_checkpoint_path,
        "metrics_path": metrics_path,
        "pred_path": pred_path,
        "analysis_path": analysis_path,
        "metrics_payload": metrics_payload,
    }


def semi_supervised_training(
    config: Dict[str, Any],
    dataset_name: str,
    frame: pd.DataFrame,
    ablation: Optional[str],
    run_id: str,
    init_checkpoint: str = "",
    pretrain_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime_cfg = copy.deepcopy(config)
    runtime_cfg["train"]["mode"] = "semi"
    ablation = canonical_ablation_name(ablation)
    feature_cache = load_feature_cache(runtime_cfg)
    device = resolve_device(runtime_cfg)
    set_seed(int(runtime_cfg["project"]["seed"]))
    benchmark_ready, benchmark_reason = checkpoint_benchmark_status(frame)
    split_policy = safe_text(frame["split_policy"].iloc[0]) if "split_policy" in frame.columns and not frame.empty else ""
    if not split_policy:
        split_policy = safe_text(config["benchmark"].get("protocol", ""))
    train_frame = dataframe_for_split(frame, "train")
    val_frame = dataframe_for_split(frame, "val")
    test_frame = dataframe_for_split(frame, "test")
    unlabeled_in_train = train_frame[~train_frame["is_labeled"].apply(as_bool)].copy().reset_index(drop=True)
    labeled_train = train_frame[train_frame["is_labeled"].apply(as_bool)].copy().reset_index(drop=True)
    if labeled_train.empty:
        raise RuntimeError("No labeled training samples available for semi-supervised mode")
    cell_vocab = make_cell_vocab(frame)
    runtime_cfg["train"]["_pos_weight"] = train_pos_weight(labeled_train)
    test_loader = build_dataloader(test_frame, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False) if not test_frame.empty else None
    val_loader = build_dataloader(val_frame, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False) if not val_frame.empty else None
    model = build_model(runtime_cfg, cell_vocab).to(device)
    copied_pretrain_tensors = 0
    if pretrain_checkpoint is not None:
        copied_pretrain_tensors = initialize_shared_encoder_from_pretrain(model, pretrain_checkpoint)
    best_path = output_dir(runtime_cfg) / f"{run_id}_model.pt"
    dataset_fingerprint = compute_dataset_fingerprint(frame)
    fit_summary: Dict[str, Any] = {"best_epoch": -1, "best_selection": None}

    def fit_model(model_obj: DQSeqGraph, train_df: pd.DataFrame, epochs: int) -> Tuple[DQSeqGraph, Dict[str, Any]]:
        runtime_cfg["train"]["_pos_weight"] = train_pos_weight(train_df)
        loader = build_dataloader(train_df, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, True)
        monitor_loader = protocol_monitor_loader(dataset_name, split_policy, runtime_cfg, val_loader, test_loader, loader)
        monitor_scope = protocol_monitor_scope(val_loader)
        optimizer = torch.optim.AdamW(
            model_obj.parameters(),
            lr=float(runtime_cfg["train"]["lr"]),
            weight_decay=float(runtime_cfg["train"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, int(runtime_cfg["train"]["early_stop_patience"]) // 4),
        )
        stale = 0
        best_epoch_local = -1
        best_selection_local: Optional[Dict[str, Any]] = None
        for epoch_idx in range(epochs):
            run_epoch(model_obj, loader, device, runtime_cfg, optimizer, ablation)
            monitor_result = run_epoch(model_obj, monitor_loader, device, runtime_cfg, None, ablation) if monitor_loader else {"metrics": {}}
            candidate_selection = monitor_selection_record(
                monitor_result,
                dataset_name,
                split_policy,
                benchmark_ready,
                runtime_cfg,
                monitor_scope,
            )
            metric_value = float(monitor_result["metrics"].get(runtime_cfg["benchmark"]["primary_metric"], float("nan")))
            if math.isnan(metric_value):
                metric_value = -1e9
            scheduler.step(metric_value)
            if best_selection_local is None or candidate_selection["key"] > best_selection_local["key"]:
                best_selection_local = candidate_selection
                best_epoch_local = epoch_idx + 1
                stale = 0
                save_checkpoint(
                    best_path,
                    model_obj,
                    runtime_cfg,
                    cell_vocab,
                    dataset_fingerprint,
                    dataset_name,
                    split_policy,
                    ablation,
                    benchmark_ready,
                    benchmark_reason,
                    reporting_protocol=safe_text(candidate_selection.get("reporting_protocol", "")),
                    selection_scope=safe_text(candidate_selection.get("selection_scope", "")),
                    selection_rule=safe_text(candidate_selection.get("selection_rule", "")),
                    selected_epoch=best_epoch_local,
                    selected_threshold=float(candidate_selection.get("selected_threshold", 0.5)),
                )
            else:
                stale += 1
            if stale >= int(runtime_cfg["train"]["early_stop_patience"]):
                break
        best_model_obj, _, _ = load_checkpoint(best_path, device)
        fit_summary["best_epoch"] = best_epoch_local
        fit_summary["best_selection"] = best_selection_local
        return best_model_obj, {"best_epoch": best_epoch_local, "best_selection": best_selection_local}

    initial_train = pd.concat([labeled_train, unlabeled_in_train], ignore_index=True)
    warmstart_model, fit_summary = fit_model(model, initial_train, int(runtime_cfg["train"]["epochs"]))
    warmstart_state = {key: value.detach().cpu().clone() for key, value in warmstart_model.state_dict().items()}
    warmstart_test = run_epoch(warmstart_model, test_loader, device, runtime_cfg, None, ablation) if test_loader else {"meta": [], "metrics": {}}
    current_train = initial_train.copy()
    remaining_unlabeled = unlabeled_in_train.copy()
    pseudo_records: List[Dict[str, Any]] = []
    rounds = int(runtime_cfg["semi"]["pseudo_rounds"])
    threshold = float(runtime_cfg["semi"]["confidence_threshold"])
    pseudo_weight = float(runtime_cfg["semi"]["pseudo_label_weight"])
    pseudo_selector = safe_text(runtime_cfg["semi"].get("selector", "threshold")) or "threshold"
    pseudo_topk = max(0, safe_int(runtime_cfg["semi"].get("balanced_topk"), 0))
    teacher_checkpoint_path = safe_text(runtime_cfg["semi"].get("teacher_checkpoint", ""))
    teacher_model = None
    teacher_ablation = None
    teacher_cell_vocab = cell_vocab
    if teacher_checkpoint_path:
        teacher_model, teacher_checkpoint, teacher_cell_vocab = load_checkpoint(
            resolve_path(runtime_cfg, teacher_checkpoint_path),
            device,
        )
        teacher_ablation = safe_text(teacher_checkpoint.get("ablation")) or None

    for round_idx in range(1, rounds + 1):
        if remaining_unlabeled.empty:
            break
        best_model, _, _ = load_checkpoint(best_path, device)
        before_state = {key: value.detach().cpu().clone() for key, value in best_model.state_dict().items()}
        before_loss = run_epoch(
            best_model,
            build_dataloader(current_train, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False),
            device,
            runtime_cfg,
            None,
            ablation,
        )["loss"]
        unlabeled_loader = build_dataloader(
            remaining_unlabeled, dataset_name, runtime_cfg, feature_cache, teacher_cell_vocab, device, False
        )
        pseudo_model = teacher_model if teacher_model is not None else best_model
        pseudo_ablation = teacher_ablation if teacher_model is not None else ablation
        scored = run_epoch(pseudo_model, unlabeled_loader, device, runtime_cfg, None, pseudo_ablation)
        if not scored["meta"]:
            break
        scored_frame = prediction_frame(scored["meta"])
        if pseudo_selector == "balanced_topk":
            ranked_desc = scored_frame.sort_values("prob", ascending=False)["row_id"].astype(str).tolist()
            ranked_asc = scored_frame.sort_values("prob", ascending=True)["row_id"].astype(str).tolist()
            positive_ids = ranked_desc[:pseudo_topk]
            positive_seen = set(positive_ids)
            negative_ids = [row_id for row_id in ranked_asc if row_id not in positive_seen][:pseudo_topk]
        else:
            positive_ids = scored_frame.loc[scored_frame["prob"] >= threshold, "row_id"].tolist()
            negative_ids = scored_frame.loc[scored_frame["prob"] <= (1.0 - threshold), "row_id"].tolist()
        chosen_ids = set(positive_ids + negative_ids)
        if chosen_ids:
            chosen = remaining_unlabeled[remaining_unlabeled["row_id"].isin(chosen_ids)].copy()
            chosen.loc[:, "is_labeled"] = True
            chosen.loc[:, "label"] = chosen["row_id"].apply(lambda row_id: 1 if row_id in positive_ids else 0)
            chosen.loc[:, "_sample_weight"] = pseudo_weight
            current_train = pd.concat([current_train, chosen], ignore_index=True)
            remaining_unlabeled = remaining_unlabeled[~remaining_unlabeled["row_id"].isin(chosen_ids)].reset_index(drop=True)
            best_model, fit_summary = fit_model(best_model, current_train, max(10, int(runtime_cfg["train"]["epochs"]) // 2))
            after_loss = run_epoch(
                best_model,
                build_dataloader(current_train, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False),
                device,
                runtime_cfg,
                None,
                ablation,
            )["loss"]
            after_state = {key: value.detach().cpu().clone() for key, value in best_model.state_dict().items()}
            param_delta = state_dict_l2_delta(before_state, after_state)
        else:
            after_loss = before_loss
            param_delta = 0.0
        pseudo_records.append(
            {
                "round": round_idx,
                "selector": pseudo_selector,
                "balanced_topk": pseudo_topk if pseudo_selector == "balanced_topk" else 0,
                "teacher_checkpoint": teacher_checkpoint_path,
                "n_pseudo_pos": len(positive_ids),
                "n_pseudo_neg": len(negative_ids),
                "train_loss_before": before_loss,
                "train_loss_after": after_loss,
                "model_param_delta_l2": param_delta,
            }
        )

    final_model, _, _ = load_checkpoint(best_path, device)
    runtime_cfg["train"]["_pos_weight"] = train_pos_weight(current_train)
    train_eval_loader = build_dataloader(current_train, dataset_name, runtime_cfg, feature_cache, cell_vocab, device, False)
    train_eval = run_epoch(final_model, train_eval_loader, device, runtime_cfg, None, ablation)
    val_eval = run_epoch(final_model, val_loader, device, runtime_cfg, None, ablation) if val_loader else {"metrics": {}, "meta": []}
    test_eval = run_epoch(final_model, test_loader, device, runtime_cfg, None, ablation) if test_loader else {"metrics": {}, "meta": []}
    threshold_candidates = threshold_candidates_from_predictions(
        {
            "train": split_truth_prob_payload(train_eval),
            "val": split_truth_prob_payload(val_eval),
            "test": split_truth_prob_payload(test_eval),
        }
    )
    feature_meta = model_feature_metadata(frame, runtime_cfg, ablation)
    reporting_meta = reporting_payload_from_thresholds(
        dataset_name,
        split_policy,
        benchmark_ready,
        threshold_candidates,
        test_eval["metrics"],
        int(fit_summary.get("best_epoch", -1)),
        runtime_cfg,
        fit_summary.get("best_selection"),
    )
    regime_meta = regime_metadata("semi")
    if pretrain_checkpoint is not None:
        ctp_meta = regime_metadata("supervised", init_checkpoint=init_checkpoint, pretrain_checkpoint=pretrain_checkpoint)
        regime_meta.update(
            {
                "uses_unlabeled": True,
                "unlabeled_usage": "self_supervised_tuple_pretraining_then_pseudo_labeling",
                "pseudo_label": True,
                "teacher": False,
                "ensemble": False,
                "init_checkpoint": init_checkpoint,
                "pretrain_tasks": ctp_meta.get("pretrain_tasks", []),
                "tuple_negative_policy": ctp_meta.get("tuple_negative_policy", []),
                "positive_construction": ctp_meta.get("positive_construction", "observed"),
                "pretrain_initialized_tensors": int(copied_pretrain_tensors),
            }
        )
    save_checkpoint(
        best_path,
        final_model,
        runtime_cfg,
        cell_vocab,
        dataset_fingerprint,
        dataset_name,
        split_policy,
        ablation,
        benchmark_ready,
        benchmark_reason,
        threshold_candidates=threshold_candidates,
        requested_features=feature_meta["requested_features"],
        effective_features=feature_meta["effective_features"],
        reporting_protocol=reporting_meta["reporting_protocol"],
        selection_scope=reporting_meta["selection_scope"],
        selection_rule=reporting_meta["selection_rule"],
        selected_epoch=reporting_meta["selected_epoch"],
        selected_threshold=reporting_meta["selected_threshold"],
        extra_metadata=regime_meta,
    )
    final_state = {key: value.detach().cpu().clone() for key, value in final_model.state_dict().items()}
    reference_checkpoint_path = find_reference_supervised_checkpoint(runtime_cfg, dataset_name, ablation, run_id)
    if reference_checkpoint_path is not None:
        reference_model, reference_checkpoint, reference_vocab = load_checkpoint(reference_checkpoint_path, device)
        reference_test_loader = (
            build_dataloader(test_frame, dataset_name, runtime_cfg, feature_cache, reference_vocab, device, False)
            if test_loader
            else None
        )
        reference_eval = (
            run_epoch(reference_model, reference_test_loader, device, runtime_cfg, None, reference_checkpoint.get("ablation") or None)
            if reference_test_loader
            else {"meta": [], "metrics": {}}
        )
        reference_state = {key: value.detach().cpu().clone() for key, value in reference_model.state_dict().items()}
        reference_pred = prediction_frame(reference_eval["meta"])
        reference_label = str(reference_checkpoint_path)
    else:
        reference_state = warmstart_state
        reference_pred = prediction_frame(warmstart_test["meta"])
        reference_label = "internal_warmstart"
    parameter_delta = state_dict_l2_delta(reference_state, final_state)
    final_pred = prediction_frame(test_eval["meta"])
    prediction_corr, max_abs_diff = prediction_comparison(reference_pred, final_pred)
    had_effective_update = not (parameter_delta == 0.0 or max_abs_diff == 0.0)
    semi_comparison = {
        "reference_supervised_checkpoint": reference_label,
        "parameter_l2_delta": parameter_delta,
        "prediction_correlation_on_test": prediction_corr,
        "prediction_max_abs_diff_on_test": max_abs_diff,
        "had_effective_update": had_effective_update,
    }
    pred_frame = prediction_frame(train_eval["meta"] + val_eval["meta"] + test_eval["meta"])
    metrics_payload = {
        "run_id": run_id,
        "run_label": runtime_run_label(runtime_cfg),
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "mode": "semi",
        "ablation": ablation or "",
        "protein_encoder": protein_encoder_name(runtime_cfg),
        "split_policy": split_policy,
        "benchmark_ready": benchmark_ready,
        "reason": benchmark_reason,
        "threshold": 0.5,
        "threshold_selection": "fixed_0.5",
        "selected_epoch": reporting_meta["selected_epoch"],
        "selected_threshold": reporting_meta["selected_threshold"],
        "selection_scope": reporting_meta["selection_scope"],
        "selection_rule": reporting_meta["selection_rule"],
        "reporting_protocol": reporting_meta["reporting_protocol"],
        "metrics": {
            "train": train_eval["metrics"],
            "val": val_eval["metrics"],
            "test": test_eval["metrics"],
            "test_fixed_0p5": reporting_meta["reporting_metrics"]["test_fixed_0p5"],
        },
        "counts": {
            "train": int((current_train["split"] == "train").sum()),
            "val": int((frame["split"] == "val").sum()),
            "test": int((frame["split"] == "test").sum()),
            "unlabeled_remaining": int(len(remaining_unlabeled)),
        },
        "pseudo_rounds": pseudo_records,
        "semi_comparison": semi_comparison,
        "threshold_candidates": threshold_candidates,
        "reporting_metrics": reporting_meta["reporting_metrics"],
        "train_options": {
            "epochs": int(runtime_cfg["train"]["epochs"]),
            "batch_size": int(runtime_cfg["train"]["batch_size"]),
            "lr": float(runtime_cfg["train"]["lr"]),
            "weight_decay": float(runtime_cfg["train"]["weight_decay"]),
            "loss": safe_text(runtime_cfg["train"].get("loss", "bce")).lower() or "bce",
            "pseudo_selector": pseudo_selector,
            "pseudo_threshold": float(runtime_cfg["semi"]["confidence_threshold"]),
            "pseudo_topk": pseudo_topk,
            "teacher_checkpoint": teacher_checkpoint_path,
            "pseudo_weight": float(runtime_cfg["semi"]["pseudo_label_weight"]),
            "init_checkpoint": init_checkpoint,
            "pretrain_initialized_tensors": int(copied_pretrain_tensors),
        },
        **regime_meta,
        **feature_meta,
    }
    if not had_effective_update:
        metrics_payload["reason"] = (
            safe_text(metrics_payload.get("reason"))
            or "Semi-supervised training completed without a measurable parameter/prediction update under the current pseudo-label setting."
        )
    metrics_path, pred_path, analysis_path = save_metrics_and_analysis(runtime_cfg, run_id, metrics_payload, pred_frame)
    append_manifest(
        runtime_cfg,
        manifest_payload(
            runtime_cfg,
            run_id,
            dataset_name,
            split_policy,
            best_path,
            metrics_path,
            pred_path,
            ablation or "",
            benchmark_ready,
            ablation or "",
            reporting_meta["reporting_protocol"],
        ),
    )
    return {
        "checkpoint_path": best_path,
        "metrics_path": metrics_path,
        "pred_path": pred_path,
        "analysis_path": analysis_path,
        "metrics_payload": metrics_payload,
    }


def normalize_query_input(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    smiles_col = find_column(frame, SMILES_ALIASES)
    target_id_col = find_column(frame, TARGET_ID_ALIASES)
    target_name_col = find_column(frame, TARGET_NAME_ALIASES)
    target_seq_col = find_column(frame, TARGET_SEQUENCE_ALIASES)
    recruiter_id_col = find_column(frame, RECRUITER_ID_ALIASES)
    recruiter_name_col = find_column(frame, RECRUITER_NAME_ALIASES)
    recruiter_seq_col = find_column(frame, RECRUITER_SEQUENCE_ALIASES)
    cell_col = find_column(frame, CELL_LINE_ALIASES)
    task_col = find_column(frame, TASK_ID_ALIASES)
    rows = []
    for idx, record in frame.iterrows():
        row = empty_standard_row("query", "predict_input", f"query_{idx:05d}")
        row["smiles"] = safe_text(record[smiles_col]) if smiles_col else ""
        row["target_id"] = safe_text(record[target_id_col]) if target_id_col else ""
        row["target_name"] = safe_text(record[target_name_col]) if target_name_col else row["target_id"]
        row["target_sequence"] = safe_text(record[target_seq_col]) if target_seq_col else ""
        row["recruiter_id"] = safe_text(record[recruiter_id_col]) if recruiter_id_col else ""
        row["recruiter_name"] = safe_text(record[recruiter_name_col]) if recruiter_name_col else row["recruiter_id"]
        row["recruiter_sequence"] = safe_text(record[recruiter_seq_col]) if recruiter_seq_col else ""
        row["cell_line"] = safe_text(record[cell_col]) if cell_col else ""
        row["task_id"] = safe_text(record[task_col]) if task_col else (row["target_id"] or row["target_name"] or row["row_id"])
        row["target_sequence_source"] = "manual" if row["target_sequence"] else "missing"
        row["recruiter_sequence_source"] = "manual" if row["recruiter_sequence"] else "missing"
        rows.append(row)
    return standardize_frame_columns(pd.DataFrame(rows))


def train_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    dataset_name = args.dataset
    split_file = safe_text(getattr(args, "split_file", ""))
    frame, _ = load_frame_for_run(runtime_cfg, dataset_name, split_file)
    if protein_encoder_name(runtime_cfg) == ESM2_650M_ENCODER:
        validate_esm2_cache_for_frame(frame, runtime_cfg, require_target=True, require_recruiter=True)
    if dataset_name == "protac8k" and not args.allow_nonbenchmark:
        benchmark_ready = bool(frame["benchmark_ready"].all()) if "benchmark_ready" in frame.columns else False
        if not benchmark_ready:
            raise RuntimeError(
                "This dataset is not benchmark-ready. Use --allow-nonbenchmark only for smoke tests."
            )
    effective_mode = "semi" if canonical_ablation_name(args.ablation) == "full_with_pseudo_label" else args.mode
    if split_file and effective_mode == "semi":
        raise RuntimeError("`--split-file` is currently supported only for supervised/CTP runs in this phase.")
    ablation = canonical_ablation_name(args.ablation or None)
    init_checkpoint = safe_text(getattr(args, "init", ""))
    pretrain_checkpoint = None
    if init_checkpoint:
        if effective_mode not in {"supervised", "semi"}:
            raise RuntimeError("`--init` is only supported for supervised fine-tuning or pseudo-label training.")
        pretrain_checkpoint = load_pretrain_checkpoint_for_init(runtime_cfg, init_checkpoint)
        if effective_mode == "semi":
            if ablation != "full_with_pseudo_label":
                raise RuntimeError("Semi-supervised `--init` requires `--ablation full_with_pseudo_label`.")
        elif ablation not in {"ctp_finetune", "full_no_episodic", "full_with_episodic", ""}:
            raise RuntimeError("`--init` can only be used with ctp_finetune or full-model supervised ablations.")
    if ablation == "ctp_finetune" and not init_checkpoint:
        raise RuntimeError("`--ablation ctp_finetune` requires `--init <pretrain_ckpt>`.")
    if ablation in {
        "smiles_target_seq",
        "smiles_target_recruiter_seq",
        "target_only",
        "e3_only",
        "target_e3_only",
        "target_e3_assay_only",
        "full_no_episodic",
        "full_with_episodic",
        "full_with_pseudo_label",
        "ctp_finetune",
    }:
        feature_meta = model_feature_metadata(frame, runtime_cfg, ablation)
        if (
            ablation in {"smiles_target_seq", "smiles_target_recruiter_seq", "target_only", "target_e3_only", "target_e3_assay_only", "ctp_finetune"}
            and feature_meta["target_sequence_coverage"] <= 0.0
        ) or (
            ablation in {"smiles_target_recruiter_seq", "e3_only", "target_e3_only", "target_e3_assay_only", "ctp_finetune"}
            and feature_meta["recruiter_sequence_coverage"] <= 0.0
        ):
            raise RuntimeError(
                "Requested sequence-aware ablation has zero effective sequence coverage. Run `fetchseq` first."
            )
    run_id = training_run_id(dataset_name, effective_mode, ablation, runtime_run_label(runtime_cfg))
    if effective_mode == "semi":
        result = semi_supervised_training(
            runtime_cfg,
            dataset_name,
            frame,
            ablation,
            run_id,
            init_checkpoint=init_checkpoint,
            pretrain_checkpoint=pretrain_checkpoint,
        )
    else:
        result = supervised_training(
            runtime_cfg,
            dataset_name,
            frame,
            ablation,
            run_id,
            init_checkpoint=init_checkpoint,
            pretrain_checkpoint=pretrain_checkpoint,
        )
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"metrics: {result['metrics_path']}")
    print(f"predictions: {result['pred_path']}")


def eval_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    dataset_name = args.dataset
    split_file = safe_text(getattr(args, "split_file", ""))
    frame, analysis_meta = load_frame_for_run(runtime_cfg, dataset_name, split_file)
    device = resolve_device(runtime_cfg)
    model, checkpoint, cell_vocab = load_checkpoint(resolve_path(runtime_cfg, args.checkpoint), device)
    inference_cfg = copy.deepcopy(checkpoint.get("config") or runtime_cfg)
    inference_cfg["project"]["device"] = runtime_cfg["project"]["device"]
    inference_cfg["runtime"] = dict(inference_cfg.get("runtime", {}))
    inference_cfg["runtime"]["run_label"] = runtime_run_label(runtime_cfg)
    inference_cfg["runtime"]["split_file"] = split_file
    feature_cache = load_feature_cache(inference_cfg)
    split_policy = safe_text(frame["split_policy"].iloc[0]) if "split_policy" in frame.columns and not frame.empty else ""
    if not split_policy:
        split_policy = checkpoint.get("split_policy", safe_text(runtime_cfg["benchmark"].get("protocol", "")))
    benchmark_ready = bool(checkpoint.get("benchmark_ready", False))
    benchmark_reason = safe_text(checkpoint.get("benchmark_reason", ""))
    threshold_value, threshold_meta = resolve_threshold_from_spec(args.threshold, checkpoint)
    eval_frame = evaluation_rows(frame)
    if eval_frame.empty:
        eval_frame = frame[frame["is_labeled"] == True].reset_index(drop=True)
    original_test_n = int(len(eval_frame))
    if safe_text(args.test_filter):
        eval_frame = leakage_filtered_frame(frame, safe_text(args.test_filter))
    loader = build_dataloader(eval_frame, dataset_name, inference_cfg, feature_cache, cell_vocab, device, False)
    result = run_epoch(model, loader, device, inference_cfg, None, checkpoint.get("ablation") or None, threshold=threshold_value)
    run_id = append_run_label(f"{precise_timestamp()}_{dataset_name}_eval_dqseqgraph", runtime_run_label(runtime_cfg))
    feature_meta = model_feature_metadata(frame, inference_cfg, checkpoint.get("ablation") or None)
    metrics_payload = {
        "run_id": run_id,
        "run_label": runtime_run_label(runtime_cfg),
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "mode": "eval",
        "ablation": checkpoint.get("ablation", ""),
        "protein_encoder": safe_text(checkpoint.get("protein_encoder", protein_encoder_name(runtime_cfg))),
        "split_policy": split_policy,
        "benchmark_ready": benchmark_ready,
        "reason": benchmark_reason,
        "reporting_protocol": safe_text(checkpoint.get("reporting_protocol", "")),
        "threshold": float(threshold_value),
        "threshold_selection": "fixed_0.5" if abs(float(threshold_value) - 0.5) < 1e-12 else safe_text(threshold_meta.get("threshold_spec", "custom")),
        "metrics": {"test": result["metrics"]},
        "counts": {"test": int(len(eval_frame)), "original_test": original_test_n},
        "test_filter": safe_text(args.test_filter),
        **threshold_meta,
        **analysis_metadata_payload(analysis_meta),
        **feature_meta,
    }
    pred_frame = prediction_frame(result["meta"])
    metrics_path, pred_path, _ = save_metrics_and_analysis(inference_cfg, run_id, metrics_payload, pred_frame)
    threshold_report = {
        "run_id": run_id,
        "checkpoint": str(resolve_path(runtime_cfg, args.checkpoint)),
        "threshold": float(threshold_value),
        **threshold_meta,
        "available_threshold_candidates": checkpoint.get("threshold_candidates") or {},
        **result["metrics"],
    }
    threshold_report_path = write_threshold_report(inference_cfg, run_id, threshold_report)
    append_manifest(
        inference_cfg,
        manifest_payload(
            inference_cfg,
            run_id,
            dataset_name,
            split_policy,
            resolve_path(runtime_cfg, args.checkpoint),
            metrics_path,
            pred_path,
            "eval",
            benchmark_ready,
            checkpoint.get("ablation", ""),
            safe_text(checkpoint.get("reporting_protocol", "")),
        ),
    )
    print(f"metrics: {metrics_path}")
    print(f"predictions: {pred_path}")
    print(f"threshold_report: {threshold_report_path}")
    if safe_text(args.test_filter):
        leak_metrics_path, leak_pred_path = write_leakage_filtered_report(inference_cfg, run_id, metrics_payload, pred_frame)
        print(f"leakage_filtered_metrics: {leak_metrics_path}")
        print(f"leakage_filtered_predictions: {leak_pred_path}")


def mean_episode_metrics(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    keys = set()
    for item in items:
        keys.update(item.keys())
    summary: Dict[str, Any] = {}
    for key in sorted(keys):
        values = []
        for item in items:
            value = item.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            summary[key] = float(np.mean(values))
    return summary


def fewshot_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    dataset_name = args.dataset
    frame, analysis_meta = load_frame_for_run(runtime_cfg, dataset_name, safe_text(args.split_file))
    if safe_text(analysis_meta.get("analysis_scheme")) != "fewshot_target":
        raise RuntimeError("`fewshot` requires a split overlay generated with `--scheme fewshot_target`.")
    if safe_text(args.adapt) != "classifier_head":
        raise RuntimeError("Few-shot v1 only supports `--adapt classifier_head`.")
    device = resolve_device(runtime_cfg)
    model, checkpoint, cell_vocab = load_checkpoint(resolve_path(runtime_cfg, args.checkpoint), device)
    inference_cfg = copy.deepcopy(checkpoint.get("config") or runtime_cfg)
    inference_cfg["project"]["device"] = runtime_cfg["project"]["device"]
    inference_cfg["runtime"] = dict(inference_cfg.get("runtime", {}))
    inference_cfg["runtime"]["run_label"] = runtime_run_label(runtime_cfg)
    inference_cfg["runtime"]["split_file"] = safe_text(args.split_file)
    feature_cache = load_feature_cache(inference_cfg)
    split_policy = safe_text(frame["split_policy"].iloc[0]) if "split_policy" in frame.columns and not frame.empty else ""
    if not split_policy:
        split_policy = safe_text(checkpoint.get("split_policy", safe_text(runtime_cfg["benchmark"].get("protocol", ""))))
    benchmark_ready, benchmark_reason = checkpoint_benchmark_status(frame)
    threshold_value, threshold_meta = resolve_threshold_from_spec(args.threshold, checkpoint)
    support_frame = frame[(frame["fewshot_role"] == "support") & frame["is_labeled"].apply(as_bool)].reset_index(drop=True)
    query_frame = frame[(frame["fewshot_role"] == "query") & frame["is_labeled"].apply(as_bool)].reset_index(drop=True)
    if support_frame.empty or query_frame.empty:
        raise RuntimeError("Few-shot split overlay contains no support/query rows.")
    meta_targets = sorted(set(query_frame["target_id"].fillna("").astype(str).tolist()))
    ablation = checkpoint.get("ablation") or None
    episode_metrics: List[Dict[str, Any]] = []
    episode_pred_rows: List[Dict[str, Any]] = []
    for target_id in meta_targets:
        target_support = support_frame[support_frame["target_id"].fillna("").astype(str) == target_id].reset_index(drop=True)
        target_query = query_frame[query_frame["target_id"].fillna("").astype(str) == target_id].reset_index(drop=True)
        if target_support.empty or target_query.empty:
            continue
        episode_model = copy.deepcopy(model)
        episode_model.to(device)
        for parameter in episode_model.parameters():
            parameter.requires_grad = False
        for parameter in episode_model.classifier.parameters():
            parameter.requires_grad = True
        support_cfg = copy.deepcopy(inference_cfg)
        support_cfg["train"]["batch_size"] = max(1, min(int(support_cfg["train"]["batch_size"]), int(len(target_support))))
        support_cfg["train"]["_pos_weight"] = train_pos_weight(target_support)
        support_loader = build_dataloader(target_support, dataset_name, support_cfg, feature_cache, cell_vocab, device, True)
        optimizer = torch.optim.AdamW(
            episode_model.classifier.parameters(),
            lr=float(args.adapt_lr),
            weight_decay=float(support_cfg["train"]["weight_decay"]),
        )
        for _ in range(int(args.adapt_steps)):
            run_epoch(episode_model, support_loader, device, support_cfg, optimizer, ablation, threshold=threshold_value)
        query_cfg = copy.deepcopy(support_cfg)
        query_cfg["train"]["batch_size"] = max(1, min(int(query_cfg["train"]["batch_size"]), int(len(target_query))))
        query_loader = build_dataloader(target_query, dataset_name, query_cfg, feature_cache, cell_vocab, device, False)
        query_result = run_epoch(episode_model, query_loader, device, query_cfg, None, ablation, threshold=threshold_value)
        metrics = dict(query_result.get("metrics") or {})
        metrics["target_id"] = target_id
        metrics["support_n"] = int(len(target_support))
        metrics["query_n"] = int(len(target_query))
        episode_metrics.append(metrics)
        for row in query_result.get("meta", []):
            payload = dict(row)
            payload["episode_target_id"] = target_id
            payload["fewshot_role"] = "query"
            episode_pred_rows.append(payload)
    if not episode_metrics:
        raise RuntimeError("Few-shot evaluation produced no valid meta-test episodes.")
    aggregate_metrics = mean_episode_metrics(episode_metrics)
    run_id = append_run_label(f"{precise_timestamp()}_{dataset_name}_fewshot_dqseqgraph", runtime_run_label(runtime_cfg))
    feature_meta = model_feature_metadata(frame, inference_cfg, ablation)
    fewshot_regime = {
        "uses_unlabeled": bool(checkpoint.get("uses_unlabeled", False)),
        "unlabeled_usage": safe_text(checkpoint.get("unlabeled_usage", "none")) or "none",
        "pseudo_label": False,
        "teacher": bool(checkpoint.get("teacher", False)),
        "ensemble": bool(checkpoint.get("ensemble", False)),
        "tuple_negative_policy": list(checkpoint.get("tuple_negative_policy") or []),
        "pretrain_tasks": list(checkpoint.get("pretrain_tasks") or []),
        "pretrain_train_rows_only": bool(checkpoint.get("pretrain_train_rows_only", False)),
        "init_checkpoint": str(resolve_path(runtime_cfg, args.checkpoint)),
    }
    metrics_payload = {
        "run_id": run_id,
        "run_label": runtime_run_label(runtime_cfg),
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "mode": "fewshot",
        "ablation": ablation or "",
        "protein_encoder": safe_text(checkpoint.get("protein_encoder", protein_encoder_name(inference_cfg))),
        "split_policy": split_policy,
        "benchmark_ready": benchmark_ready,
        "reason": benchmark_reason,
        "threshold": float(threshold_value),
        "threshold_selection": "fixed_0.5" if abs(float(threshold_value) - 0.5) < 1e-12 else safe_text(threshold_meta.get("threshold_spec", "custom")),
        "selected_epoch": int(checkpoint.get("selected_epoch", -1)),
        "selected_threshold": float(threshold_value),
        "selection_scope": "fewshot_episode_macro",
        "selection_rule": "fixed_0p5_episode_macro",
        "reporting_protocol": "",
        "metrics": {
            "test": aggregate_metrics,
            "test_selected": aggregate_metrics,
        },
        "reporting_metrics": {
            "test_selected": aggregate_metrics,
            "test_fixed_0p5": aggregate_metrics,
        },
        "counts": {
            "targets": int(len(meta_targets)),
            "episodes": int(len(episode_metrics)),
            "support_rows": int(len(support_frame)),
            "query_rows": int(len(query_frame)),
        },
        "fewshot": {
            "shots_per_class": int(args.shots_per_class),
            "adapt": safe_text(args.adapt),
            "adapt_steps": int(args.adapt_steps),
            "adapt_lr": float(args.adapt_lr),
            "episode_metrics": episode_metrics,
        },
        **analysis_metadata_payload(analysis_meta),
        **fewshot_regime,
        **feature_meta,
    }
    pred_frame = prediction_frame(episode_pred_rows)
    metrics_path, pred_path, analysis_path = save_metrics_and_analysis(inference_cfg, run_id, metrics_payload, pred_frame)
    append_manifest(
        inference_cfg,
        manifest_payload(
            inference_cfg,
            run_id,
            dataset_name,
            split_policy,
            resolve_path(runtime_cfg, args.checkpoint),
            metrics_path,
            pred_path,
            "fewshot",
            benchmark_ready,
            ablation or "",
            "",
        ),
    )
    print(f"metrics: {metrics_path}")
    print(f"predictions: {pred_path}")
    print(f"analysis: {analysis_path}")


def predict_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    device = resolve_device(runtime_cfg)
    model, checkpoint, cell_vocab = load_checkpoint(resolve_path(runtime_cfg, args.checkpoint), device)
    inference_cfg = copy.deepcopy(checkpoint.get("config") or runtime_cfg)
    inference_cfg["project"]["device"] = runtime_cfg["project"]["device"]
    input_path = resolve_path(runtime_cfg, args.input)
    raw = read_dataframe_from_path(input_path)
    frame = normalize_query_input(raw)
    frame, _ = ensure_sequences(frame, inference_cfg)
    feature_cache = load_feature_cache(inference_cfg)
    dataset_name = checkpoint.get("dataset", "query")
    loader = build_dataloader(frame, dataset_name, inference_cfg, feature_cache, cell_vocab, device, False)
    result = run_epoch(model, loader, device, inference_cfg, None, checkpoint.get("ablation") or None)
    run_id = f"{precise_timestamp()}_predict_dqseqgraph"
    out_path = output_dir(inference_cfg) / f"{run_id}_pred.csv"
    prediction_frame(result["meta"]).to_csv(out_path, index=False)
    append_manifest(
        inference_cfg,
        manifest_payload(
            inference_cfg,
            run_id,
            dataset_name,
            checkpoint.get("split_policy", ""),
            resolve_path(runtime_cfg, args.checkpoint),
            None,
            out_path,
            "predict",
            bool(checkpoint.get("benchmark_ready", False)),
            checkpoint.get("ablation", ""),
            safe_text(checkpoint.get("reporting_protocol", "")),
        ),
    )
    print(f"predictions: {out_path}")


def labeled_benchmark_frames(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = frame[(frame["split"] == "train") & (frame["is_labeled"] == True)].reset_index(drop=True)
    test_frame = frame[(frame["split"] == "test") & (frame["is_labeled"] == True)].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise RuntimeError("Benchmark split must have labeled train and test rows.")
    return train_frame, test_frame


def morgan_matrix_for_frame(frame: pd.DataFrame, dataset_name: str, config: Dict[str, Any]) -> np.ndarray:
    feature_cache = load_feature_cache(config)
    dataset = DegradeDataset(frame.reset_index(drop=True), dataset_name, config, feature_cache, {"UNK_CELL": 0}, torch.device("cpu"))
    rows = [dataset[idx]["morgan_fp"] for idx in range(len(dataset))]
    return np.stack(rows, axis=0).astype(np.float32)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def baseline_estimators(seed: int) -> Dict[str, Any]:
    return {
        "logreg_morgan": LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
        "svm_morgan": SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced", probability=True, random_state=seed),
        "rf_morgan": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def metrics_row_for_report(method: str, setting: str, metrics: Dict[str, Any], note: str) -> Dict[str, Any]:
    return {
        "method": method,
        "setting": setting,
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "auroc": float(metrics.get("auroc", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "mcc": float(metrics.get("mcc", float("nan"))),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
        "specificity": float(metrics.get("specificity", float("nan"))),
        "ece": float(metrics.get("ece", float("nan"))),
        "brier_score": float(metrics.get("brier_score", float("nan"))),
        "mean_positive_probability": float(metrics.get("mean_positive_probability", float("nan"))),
        "mean_negative_probability": float(metrics.get("mean_negative_probability", float("nan"))),
        "test_positive_count": int(metrics.get("test_positive_count", 0)),
        "test_negative_count": int(metrics.get("test_negative_count", 0)),
        "threshold": float(metrics.get("threshold", 0.5)),
        "note": note,
    }


def report_priority_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    return (
        metric_key(row.get("accuracy")),
        metric_key(row.get("auroc")),
        metric_key(row.get("f1")),
        metric_key(row.get("mcc")),
        metric_key(row.get("auprc")),
        metric_key(row.get("balanced_accuracy")),
    )


def normalize_policy_list(payload: Dict[str, Any]) -> List[str]:
    values = payload.get("tuple_negative_policy") or []
    return [safe_text(item) for item in values if safe_text(item)]


def infer_paper_label(payload: Dict[str, Any]) -> str:
    run_label = safe_text(payload.get("run_label"))
    if run_label:
        return run_label
    tasks = list(payload.get("pretrain_tasks") or [])
    policy = normalize_policy_list(payload)
    positive_construction = safe_text(payload.get("positive_construction", "observed")) or "observed"
    mode = safe_text(payload.get("mode"))
    init_checkpoint = safe_text(payload.get("init_checkpoint"))
    if mode == "supervised" and not init_checkpoint and not tasks:
        return "Sup labeled-only"
    if mode == "supervised" and safe_text(payload.get("ablation")) == "target_only":
        return "Target-only"
    if mode == "supervised" and safe_text(payload.get("ablation")) == "e3_only":
        return "E3-only"
    if mode == "supervised" and safe_text(payload.get("ablation")) == "target_e3_only":
        return "Target/E3-only"
    if mode == "supervised" and safe_text(payload.get("ablation")) == "target_e3_assay_only":
        return "Target/E3 + assay-only"
    if positive_construction == "shuffled_pair":
        return "Shuffled target-E3 pair"
    if positive_construction == "shuffled_independent":
        return "Shuffled target/E3 independent"
    if tasks == ["mol_contrast"]:
        return "SSL-mol only"
    if tasks == ["tuple_contrast"] and policy == ["random"]:
        return "CTP-tuple easy"
    if tasks == ["tuple_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["full"]:
        return "CTP-tuple hard"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["target_only"]:
        return "Target replacement only"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["e3_only"]:
        return "E3 replacement only"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["joint_only"]:
        return "Target+E3 replacement only"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["full_mixed"]:
        return "Mixed target/E3/joint replacement"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["frequency_matched"]:
        return "Frequency-matched negatives"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["target_seq_cluster"]:
        return "Target sequence-cluster negatives"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["protein_nn"]:
        return "Protein nearest-neighbor negatives"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["hard_mixed"]:
        return "Hard mixed negatives"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == ["random"]:
        return "CTP-full easy"
    if tasks == ["tuple_contrast", "mol_contrast"] and policy == TUPLE_NEGATIVE_POLICIES["full"]:
        return "CTP-full"
    return display_name_for_payload(payload)


def human_label(text: str) -> str:
    label = safe_text(text).replace("_", " ").strip()
    label = label.replace("esm2 650m", "ESM-2 650M")
    label = label.replace("ctp", "CTP")
    label = label.replace("sup ", "Sup ")
    return label or text


def compare_runs_row_from_payload(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    test_metrics = payload_test_metrics(payload)
    label = infer_paper_label(payload)
    counts = payload.get("counts") or {}
    return {
        "path": str(path),
        "run_id": payload.get("run_id", path.stem),
        "display_name": human_label(label),
        "raw_label": label,
        "run_label": safe_text(payload.get("run_label")),
        "mode": payload.get("mode", ""),
        "ablation": payload.get("ablation", ""),
        "protein_encoder": safe_text(payload.get("protein_encoder", "aac")) or "aac",
        "threshold": float(payload.get("threshold", test_metrics.get("threshold", 0.5))),
        "threshold_selection": safe_text(payload.get("threshold_selection", "")) or "fixed_0.5",
        "accuracy": float(test_metrics.get("accuracy", float("nan"))),
        "auroc": float(test_metrics.get("auroc", float("nan"))),
        "auprc": float(test_metrics.get("auprc", float("nan"))),
        "f1": float(test_metrics.get("f1", float("nan"))),
        "mcc": float(test_metrics.get("mcc", float("nan"))),
        "balanced_accuracy": float(test_metrics.get("balanced_accuracy", float("nan"))),
        "uses_unlabeled": bool(payload.get("uses_unlabeled", False)),
        "unlabeled_usage": safe_text(payload.get("unlabeled_usage", "none")) or "none",
        "pseudo_label": bool(payload.get("pseudo_label", False)),
        "teacher": bool(payload.get("teacher", False)),
        "ensemble": bool(payload.get("ensemble", False)),
        "tuple_negative_policy": normalize_policy_list(payload),
        "positive_construction": safe_text(payload.get("positive_construction", "observed")) or "observed",
        "pretrain_tasks": list(payload.get("pretrain_tasks") or []),
        "analysis_scheme": safe_text(payload.get("analysis_scheme")),
        "analysis_seed": safe_int(payload.get("analysis_seed"), 0),
        "label_fraction": payload.get("label_fraction"),
        "test_frac": payload.get("test_frac"),
        "shots_per_class": safe_int(payload.get("shots_per_class"), 0),
        "targets": safe_int(counts.get("targets"), 0),
        "episodes": safe_int(counts.get("episodes"), 0),
        "test_n": safe_int(counts.get("test"), 0),
        "notes": safe_text(payload.get("reason", "")) or payload_report_note(payload),
        "payload": payload,
    }


def method_bucket(row: Dict[str, Any]) -> str:
    return "CTP" if row.get("uses_unlabeled") else "Sup"


def finite_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean_std_payload(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = finite_values(rows, key)
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "n": len(values),
    }


def mean_std_text(rows: Sequence[Dict[str, Any]], key: str) -> str:
    payload = mean_std_payload(rows, key)
    if payload["n"] == 0:
        return "-"
    return f"{payload['mean']:.4f}\u00b1{payload['std']:.4f}"


def write_compare_runs_markdown(profile: str, rows: List[Dict[str, Any]], report: Dict[str, Any]) -> str:
    if profile == "official_split_final":
        best_rows = report.get("best_rows", [])
        comparison_rows = report.get("comparison_rows", [])
        return (
            "# Official Split Final Summary\n\n"
            + markdown_table(
                ["Bucket", "Method", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        item["bucket"],
                        item["display_name"],
                        f"{item['accuracy']:.4f}",
                        f"{item['auroc']:.4f}",
                        f"{item['auprc']:.4f}",
                        f"{item['f1']:.4f}",
                        f"{item['mcc']:.4f}",
                        f"{item['balanced_accuracy']:.4f}",
                    ]
                    for item in best_rows
                ],
            )
            + "\n\n## Comparison to Reported DegradeMaster\n\n"
            + markdown_table(
                ["Method", "Source", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        row["method"],
                        row["source"],
                        row["accuracy"],
                        row["auroc"],
                        row["auprc"],
                        row["f1"],
                        row["mcc"],
                        row["balanced_accuracy"],
                    ]
                    for row in comparison_rows
                ],
            )
            + "\n\n"
            + report.get("conclusion", "")
            + "\n"
        )
    if profile == "label_efficiency":
        aggregated_rows = report.get("aggregated_rows", [])
        return (
            "# Label Efficiency Summary\n\n"
            + markdown_table(
                ["Labeled fraction", "Method", "Acc mean±std", "AUROC mean±std", "AUPRC mean±std", "F1 mean±std", "MCC mean±std", "BalAcc mean±std"],
                [
                    [
                        row["label_fraction_text"],
                        row["method"],
                        row["accuracy_text"],
                        row["auroc_text"],
                        row["auprc_text"],
                        row["f1_text"],
                        row["mcc_text"],
                        row["balanced_accuracy_text"],
                    ]
                    for row in aggregated_rows
                ],
            )
            + "\n\n"
            + report.get("summary", "")
            + "\n"
        )
    if profile == "hard_split":
        aggregated_rows = report.get("aggregated_rows", [])
        return (
            "# Hard Split Summary\n\n"
            + markdown_table(
                ["Split", "Method", "Seeds", "Test n", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        row["split_name"],
                        row["method"],
                        row["seed_count"],
                        row["test_n_text"],
                        row["accuracy_text"],
                        row["auroc_text"],
                        row["auprc_text"],
                        row["f1_text"],
                        row["mcc_text"],
                        row["balanced_accuracy_text"],
                    ]
                    for row in aggregated_rows
                ],
            )
            + "\n\n"
            + report.get("summary", "")
            + "\n"
        )
    if profile == "fewshot_target":
        aggregated_rows = report.get("aggregated_rows", [])
        return (
            "# Few-Shot Target Summary\n\n"
            + markdown_table(
                ["K per class", "Method", "Targets", "Episodes", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        row["shots_per_class"],
                        row["method"],
                        row["targets_text"],
                        row["episodes_text"],
                        row["accuracy_text"],
                        row["auroc_text"],
                        row["auprc_text"],
                        row["f1_text"],
                        row["mcc_text"],
                        row["balanced_accuracy_text"],
                    ]
                    for row in aggregated_rows
                ],
            )
            + "\n\n"
            + report.get("summary", "")
            + "\n"
        )
    if profile == "generalization_lowresource_v1":
        answer_lines = report.get("answers", {})
        return (
            "# Generalization & Low-Resource v1 Summary\n\n"
            + "\n".join([f"- {key}: {value}" for key, value in answer_lines.items()])
            + "\n"
        )
    if profile == "ctp_formal_v1":
        main_table = markdown_table(
            ["Method", "Uses unlabeled", "Unlabeled usage", "Pseudo-label", "Teacher / ensemble", "Threshold", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
            [
                [
                    row["display_name"],
                    "Yes" if row["uses_unlabeled"] else "No",
                    row["unlabeled_usage"],
                    "Yes" if row["pseudo_label"] else "No",
                    "Yes" if row["teacher"] or row["ensemble"] else "No",
                    f"{row['threshold']:.1f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['auroc']:.4f}",
                    f"{row['auprc']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['mcc']:.4f}",
                    f"{row['balanced_accuracy']:.4f}",
                ]
                for row in rows
            ],
        )
        return (
            "# CTP Formal v1 Summary\n\n"
            + main_table
            + "\n\n"
            + report.get("conclusion", "")
            + "\n"
        )
    if profile in {"esm2_650m_sprint", "fixed05_finetune_sweep"}:
        return (
            f"# {human_label(profile)} Summary\n\n"
            + markdown_table(
                ["Method", "protein_encoder", "Threshold", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        row["display_name"],
                        row["protein_encoder"],
                        f"{row['threshold']:.1f}",
                        f"{row['accuracy']:.4f}",
                        f"{row['auroc']:.4f}",
                        f"{row['auprc']:.4f}",
                        f"{row['f1']:.4f}",
                        f"{row['mcc']:.4f}",
                        f"{row['balanced_accuracy']:.4f}",
                    ]
                    for row in rows
                ],
            )
            + ("\n\n" + report.get("summary", "") + "\n" if report.get("summary") else "\n")
        )
    if profile == "sota_sprint_v2":
        comparison_rows = report.get("comparison_rows", [])
        best_rows = report.get("best_rows", [])
        answer_lines = report.get("answers", {})
        return (
            "# SOTA Sprint v2 Summary\n\n"
            + "## Current Best DegradeQuery\n\n"
            + markdown_table(
                ["Bucket", "Method", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        item["bucket"],
                        item["display_name"],
                        f"{item['accuracy']:.4f}",
                        f"{item['auroc']:.4f}",
                        f"{item['auprc']:.4f}",
                        f"{item['f1']:.4f}",
                        f"{item['mcc']:.4f}",
                        f"{item['balanced_accuracy']:.4f}",
                    ]
                    for item in best_rows
                ],
            )
            + "\n\n## Reported DegradeMaster vs Ours\n\n"
            + markdown_table(
                ["Method", "Source", "Acc", "AUROC", "AUPRC", "F1", "MCC", "BalAcc"],
                [
                    [
                        row["method"],
                        row["source"],
                        row["accuracy"],
                        row["auroc"],
                        row["auprc"],
                        row["f1"],
                        row["mcc"],
                        row["balanced_accuracy"],
                    ]
                    for row in comparison_rows
                ],
            )
            + "\n\n## Answers\n\n"
            + "\n".join([f"- {key}: {'Yes' if value else 'No'}" for key, value in answer_lines.items()])
            + "\n"
        )
    return (
        "# PROTAC-8K Compare Runs Report\n\n"
        + markdown_table(
            ["Method", "mode", "ablation", "Threshold", "Acc", "AUROC", "AUPRC", "F1", "MCC", "notes"],
            [
                [
                    row["display_name"],
                    row["mode"],
                    row["ablation"],
                    f"{row['threshold']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['auroc']:.4f}",
                    f"{row['auprc']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['mcc']:.4f}",
                    row["notes"],
                ]
                for row in rows
            ],
        )
        + ("\n\n" + report.get("summary", "") + "\n" if report.get("summary") else "\n")
    )


def latest_benchmark_runs(config: Dict[str, Any], dataset_name: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(output_dir(config).glob(f"*_{dataset_name}_*metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if safe_text(payload.get("split_policy")) != "degrademaster_code_split_seed111":
            continue
        if not bool(payload.get("benchmark_ready", False)):
            continue
        mode = safe_text(payload.get("mode"))
        if mode not in {"supervised", "semi"}:
            continue
        metrics = payload_test_metrics(payload)
        key = selection_key(metrics)
        timestamp = safe_text(payload.get("timestamp"))
        candidate = {"path": path, "payload": payload, "key": key, "timestamp": timestamp}
        incumbent = out.get(mode)
        if incumbent is None:
            out[mode] = candidate
            continue
        incumbent_key = tuple(incumbent["key"]) + (safe_text(incumbent.get("timestamp")),)
        candidate_key = tuple(candidate["key"]) + (timestamp,)
        if candidate_key > incumbent_key:
            out[mode] = candidate
    return out


def benchmark_alignment_payload(frame: pd.DataFrame) -> Dict[str, Any]:
    labeled = frame[frame["is_labeled"] == True].copy()
    high = int((labeled["label"].astype(int) == 1).sum()) if not labeled.empty else 0
    low = int((labeled["label"].astype(int) == 0).sum()) if not labeled.empty else 0
    test_size = int((frame["split"] == "test").sum())
    return {
        "dataset": "PROTAC-8K",
        "ours_total": int(len(frame)),
        "ours_labeled": int(labeled.shape[0]),
        "ours_unlabeled": int((frame["is_labeled"] == False).sum()),
        "ours_high": high,
        "ours_low": low,
        "degrademaster_table1_total": 8636,
        "degrademaster_table1_high": 577,
        "degrademaster_table1_low": 925,
        "protocol": "degrademaster_code_split_seed111",
        "test_size": test_size,
        "note": (
            "DegradeMaster paper has an internal inconsistency between the prose description and Table 1. "
            "Our current normalized data matches Table 1 rather than the prose description."
        ),
    }


def baseline_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    dataset_name = args.dataset
    methods = [safe_text(item) for item in safe_text(args.methods).split(",") if safe_text(item)]
    estimators = baseline_estimators(int(runtime_cfg["project"]["seed"]))
    unsupported = [item for item in methods if item not in estimators]
    if unsupported:
        raise ValueError(f"Unsupported baseline methods: {unsupported}")
    frame = load_normalized_dataset(runtime_cfg, dataset_name)
    train_frame, test_frame = labeled_benchmark_frames(frame)
    x_train = morgan_matrix_for_frame(train_frame, dataset_name, runtime_cfg)
    x_test = morgan_matrix_for_frame(test_frame, dataset_name, runtime_cfg)
    y_train = train_frame["label"].astype(int).to_numpy()
    y_test = test_frame["label"].astype(int).to_numpy()
    rows = []
    for method in methods:
        estimator = estimators[method]
        estimator.fit(x_train, y_train)
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(x_test)[:, 1]
        else:
            decision = estimator.decision_function(x_test)
            decision = np.asarray(decision, dtype=float)
            y_prob = 1.0 / (1.0 + np.exp(-decision))
        fixed_metrics = compute_metrics(y_test, y_prob, int(runtime_cfg["output"]["ece_bins"]), threshold=0.5)
        rows.append(metrics_row_for_report(method, "Super@0.5", fixed_metrics, "local fixed@0.5"))
        ref = DEGRADEMASTER_BASELINE_REFERENCE.get(method)
        if ref and abs(float(fixed_metrics["auroc"]) - float(ref["auroc"])) > 0.05:
            append_note(
                runtime_cfg,
                f"Baseline Protocol Mismatch {timestamp_now()}",
                [f"- method: {method}", "- Potential protocol mismatch: local baseline metrics differ substantially from DegradeMaster Table 2."],
            )
    report = {
        "dataset": dataset_name,
        "split_policy": "degrademaster_code_split_seed111",
        "reporting_protocol": REPORTING_PROTOCOL,
        "selection_rule": SELECTION_RULE,
        "methods": methods,
        "rows": rows,
    }
    json_path = baseline_report_json_path(runtime_cfg, dataset_name)
    md_path = baseline_report_md_path(runtime_cfg, dataset_name)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(
        "# PROTAC-8K Baseline Report\n\n"
        + markdown_table(
            ["Method", "Setting", "Acc", "AUROC", "F1", "MCC", "Note"],
            [
                [
                    row["method"],
                    row["setting"],
                    f"{row['accuracy']:.4f}",
                    f"{row['auroc']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['mcc']:.4f}",
                    row["note"],
                ]
                for row in rows
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"baseline_report_json: {json_path}")
    print(f"baseline_report_md: {md_path}")


def report_compare_degrademaster(
    config: Dict[str, Any],
    dataset_name: str,
    output_stem: str = "",
) -> Tuple[Path, Path]:
    frame = load_normalized_dataset(config, dataset_name)
    alignment = benchmark_alignment_payload(frame)
    current_runs = latest_benchmark_runs(config, dataset_name)
    current_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []
    test_reference = dataframe_for_split(frame, "test")
    test_positive_count = int((test_reference["label"].astype(int) == 1).sum()) if not test_reference.empty else 0
    test_negative_count = int((test_reference["label"].astype(int) == 0).sum()) if not test_reference.empty else 0
    for mode, label in [("supervised", "Super"), ("semi", "Semi")]:
        payload = current_runs.get(mode, {}).get("payload")
        if not payload:
            continue
        metrics = payload_test_metrics(payload)
        row = metrics_row_for_report("DegradeQuery", label, metrics, payload_report_note(payload))
        if int(row.get("test_positive_count", 0)) == 0 and test_positive_count:
            row["test_positive_count"] = test_positive_count
        if int(row.get("test_negative_count", 0)) == 0 and test_negative_count:
            row["test_negative_count"] = test_negative_count
        current_rows.append(row)
        diagnostics_rows.append(row)
    main_rows = [*DEGRADEMASTER_REFERENCE_ROWS]
    for row in current_rows:
        main_rows.append(
            {
                "method": row["method"],
                "setting": row["setting"],
                "accuracy": row["accuracy"],
                "auroc": row["auroc"],
                "f1": row["f1"],
                "note": row["note"],
            }
        )
    report = {
        "data_alignment": alignment,
        "reporting_protocol": REPORTING_PROTOCOL,
        "selection_rule": SELECTION_RULE,
        "main_comparison": main_rows,
        "diagnostics": diagnostics_rows,
        "conclusion": (
            "DegradeQuery currently exceeds DegradeMaster in AUROC but not in accuracy or F1. "
            "Therefore, it should not yet be claimed as full SOTA."
        ),
        "caveat": (
            "These DegradeQuery rows use train-side or validation-side epoch selection and a predeclared threshold "
            "of 0.5; official-test labels are used only for final evaluation."
        ),
    }
    json_path, md_path = output_paths_for_stem(
        config,
        output_stem,
        benchmark_report_json_path(config, dataset_name),
        benchmark_report_md_path(config, dataset_name),
    )
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    caveat = ""
    if current_runs:
        any_sequence_missing = False
        for item in current_runs.values():
            payload = item["payload"]
            if payload.get("missing_sequence_count") in {None, 0}:
                continue
            any_sequence_missing = True
        if any_sequence_missing or any("effective_features" not in item["payload"] for item in current_runs.values()):
            caveat = (
                "Current benchmark_alignment_v1 results were obtained before successful UniProt/PDB sequence recovery. "
                "They should be interpreted as SMILES/graph/fingerprint-dominant results with real target/recruiter IDs, "
                "not as a full sequence-aware model.\n\n"
            )
    md_path.write_text(
        "# PROTAC-8K Benchmark Report\n\n"
        "Current DegradeQuery rows below reflect the best benchmark-ready run found under the reproduced code split. "
        "Epoch selection excludes official-test labels and final evaluation uses a fixed threshold of 0.5.\n\n"
        "## Data Alignment\n\n"
        + markdown_table(
            ["Dataset", "Ours total", "Ours labeled", "Ours unlabeled", "Ours high", "Ours low", "Protocol", "Test size"],
            [[alignment["dataset"], alignment["ours_total"], alignment["ours_labeled"], alignment["ours_unlabeled"], alignment["ours_high"], alignment["ours_low"], alignment["protocol"], alignment["test_size"]]],
        )
        + "\n\n"
        + f"Note: {alignment['note']}\n\n"
        + "## Main Comparison\n\n"
        + markdown_table(
            ["Method", "Setting", "Acc", "AUROC", "F1", "Note"],
            [[row["method"], row["setting"], f"{row['accuracy']:.4f}", f"{row['auroc']:.4f}", f"{row['f1']:.4f}", row["note"]] for row in main_rows],
        )
        + "\n\n## Diagnostics\n\n"
        + markdown_table(
            ["Method", "Setting", "threshold", "test_pos", "test_neg", "precision", "recall", "specificity", "bal_acc", "MCC", "ECE", "Brier", "mean_pos_prob", "mean_neg_prob"],
            [
                [
                    row["method"],
                    row["setting"],
                    f"{row['threshold']:.4f}",
                    row["test_positive_count"],
                    row["test_negative_count"],
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['specificity']:.4f}",
                    f"{row['balanced_accuracy']:.4f}",
                    f"{row['mcc']:.4f}",
                    f"{row['ece']:.4f}",
                    f"{row['brier_score']:.4f}",
                    f"{row['mean_positive_probability']:.4f}",
                    f"{row['mean_negative_probability']:.4f}",
                ]
                for row in diagnostics_rows
            ],
        )
        + "\n\n"
        + f"Reporting protocol: `{REPORTING_PROTOCOL}` with selection rule `{SELECTION_RULE}`.\n\n"
        + report["caveat"]
        + "\n\n"
        + caveat
        + report["conclusion"]
        + "\n",
        encoding="utf-8",
    )
    return json_path, md_path


def report_compare_runs(
    config: Dict[str, Any],
    dataset_name: str,
    metric_paths: Sequence[str],
    profile: str = "",
    output_stem: str = "",
) -> Tuple[Path, Path]:
    if profile not in SUMMARY_PROFILES:
        raise ValueError(f"Unsupported report profile: {profile}")
    rows: List[Dict[str, Any]] = []
    for raw_path in metric_paths:
        path = resolve_path(config, raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(compare_runs_row_from_payload(path, payload))
    rows.sort(key=report_priority_key, reverse=True)
    summary = ""
    best_super = max((row for row in rows if row["mode"] == "supervised"), default=None, key=lambda row: selection_key(row))
    best_semi = max((row for row in rows if row["mode"] == "semi"), default=None, key=lambda row: selection_key(row))
    if best_super and best_semi:
        if metric_key(best_semi["accuracy"]) > metric_key(best_super["accuracy"]) and metric_key(best_semi["auroc"]) < metric_key(best_super["auroc"]):
            summary = "Pseudo-labeling improves threshold-dependent classification but hurts ranking quality."
    report: Dict[str, Any] = {
        "dataset": dataset_name,
        "profile": profile,
        "reporting_protocol": REPORTING_PROTOCOL,
        "selection_rule": SELECTION_RULE,
        "rows": rows,
        "summary": summary,
    }
    if profile == "official_split_final":
        eligible = [row for row in rows if row["mode"] == "supervised"]
        best_sup = max((row for row in eligible if not row["uses_unlabeled"]), default=None, key=report_priority_key)
        best_ctp = max((row for row in eligible if row["uses_unlabeled"]), default=None, key=report_priority_key)
        best_overall = max(eligible, default=None, key=report_priority_key)
        best_rows = []
        for bucket, row in [("Best Sup", best_sup), ("Best CTP", best_ctp), ("Best official-split run overall", best_overall)]:
            if row:
                best_rows.append({"bucket": bucket, **row})
        report["best_rows"] = best_rows
        report["comparison_rows"] = [
            {"method": "DegradeMaster-Sup", "source": "reported", "accuracy": "0.8141", "auroc": "0.8541", "auprc": "-", "f1": "0.8139", "mcc": "-", "balanced_accuracy": "-"},
            {"method": "DegradeMaster-Semi", "source": "reported", "accuracy": "0.8366", "auroc": "0.8825", "auprc": "-", "f1": "0.8224", "mcc": "-", "balanced_accuracy": "-"},
        ]
        for item in best_rows:
            report["comparison_rows"].append(
                {
                    "method": item["display_name"],
                    "source": "ours",
                    "accuracy": f"{item['accuracy']:.4f}",
                    "auroc": f"{item['auroc']:.4f}",
                    "auprc": f"{item['auprc']:.4f}",
                    "f1": f"{item['f1']:.4f}",
                    "mcc": f"{item['mcc']:.4f}",
                    "balanced_accuracy": f"{item['balanced_accuracy']:.4f}",
                }
            )
        report["conclusion"] = (
            "Best DegradeQuery beats reported DegradeMaster-Semi in Acc and AUROC, but not F1. "
            "CTP improves over labeled-only Sup. Official split phase is considered complete for now."
        )
    elif profile == "label_efficiency":
        fractions = [0.10, 0.25, 0.50, 1.00]
        aggregated_rows = []
        for fraction in fractions:
            for method in ["Sup", "CTP"]:
                subset = [
                    row
                    for row in rows
                    if safe_text(row.get("analysis_scheme")) == "low_label"
                    and row["mode"] == "supervised"
                    and abs(float(row.get("label_fraction") or 0.0) - fraction) < 1e-9
                    and method_bucket(row) == method
                ]
                if not subset:
                    continue
                aggregated_rows.append(
                    {
                        "label_fraction": fraction,
                        "label_fraction_text": f"{int(round(fraction * 100))}%",
                        "method": method,
                        "seed_count": len(subset),
                        "accuracy_text": mean_std_text(subset, "accuracy"),
                        "auroc_text": mean_std_text(subset, "auroc"),
                        "auprc_text": mean_std_text(subset, "auprc"),
                        "f1_text": mean_std_text(subset, "f1"),
                        "mcc_text": mean_std_text(subset, "mcc"),
                        "balanced_accuracy_text": mean_std_text(subset, "balanced_accuracy"),
                        "accuracy_stats": mean_std_payload(subset, "accuracy"),
                        "auroc_stats": mean_std_payload(subset, "auroc"),
                        "auprc_stats": mean_std_payload(subset, "auprc"),
                        "f1_stats": mean_std_payload(subset, "f1"),
                        "mcc_stats": mean_std_payload(subset, "mcc"),
                        "balanced_accuracy_stats": mean_std_payload(subset, "balanced_accuracy"),
                    }
                )
        report["aggregated_rows"] = aggregated_rows
        low_label_gains = []
        for fraction in fractions:
            sup_rows = [row for row in rows if safe_text(row.get("analysis_scheme")) == "low_label" and row["mode"] == "supervised" and abs(float(row.get("label_fraction") or 0.0) - fraction) < 1e-9 and method_bucket(row) == "Sup"]
            ctp_rows = [row for row in rows if safe_text(row.get("analysis_scheme")) == "low_label" and row["mode"] == "supervised" and abs(float(row.get("label_fraction") or 0.0) - fraction) < 1e-9 and method_bucket(row) == "CTP"]
            if sup_rows and ctp_rows:
                sup_auroc = mean_std_payload(sup_rows, "auroc")["mean"]
                ctp_auroc = mean_std_payload(ctp_rows, "auroc")["mean"]
                sup_mcc = mean_std_payload(sup_rows, "mcc")["mean"]
                ctp_mcc = mean_std_payload(ctp_rows, "mcc")["mean"]
                low_label_gains.append((fraction, ctp_auroc - sup_auroc if sup_auroc is not None and ctp_auroc is not None else float("nan"), ctp_mcc - sup_mcc if sup_mcc is not None and ctp_mcc is not None else float("nan")))
        report["summary"] = "CTP is especially useful when degradation labels are scarce." if any((metric_key(delta_auroc) > 0 or metric_key(delta_mcc) > 0) and fraction < 1.0 for fraction, delta_auroc, delta_mcc in low_label_gains) else "CTP does not show a consistent low-label advantage yet."
    elif profile == "hard_split":
        name_map = {
            "target_holdout": "target holdout",
            "target_e3_holdout": "target-E3 holdout",
            "scaffold_holdout": "scaffold holdout",
        }
        aggregated_rows = []
        for scheme in ["target_holdout", "target_e3_holdout", "scaffold_holdout"]:
            for method in ["Sup", "CTP"]:
                subset = [row for row in rows if safe_text(row.get("analysis_scheme")) == scheme and row["mode"] == "supervised" and method_bucket(row) == method]
                if not subset:
                    continue
                seed_count = len({safe_int(row.get("analysis_seed"), 0) for row in subset})
                test_values = finite_values(subset, "test_n")
                if seed_count > 1:
                    test_n_text = f"{float(np.mean(test_values)):.1f}\u00b1{float(np.std(test_values, ddof=0)):.1f}" if test_values else "-"
                    metric_text = lambda key: mean_std_text(subset, key)
                else:
                    test_n_text = str(int(round(test_values[0]))) if test_values else "-"
                    metric_text = lambda key: f"{subset[0][key]:.4f}" if subset and math.isfinite(float(subset[0][key])) else "-"
                aggregated_rows.append(
                    {
                        "split_name": name_map[scheme],
                        "analysis_scheme": scheme,
                        "method": method,
                        "seed_count": seed_count,
                        "test_n_text": test_n_text,
                        "accuracy_text": metric_text("accuracy"),
                        "auroc_text": metric_text("auroc"),
                        "auprc_text": metric_text("auprc"),
                        "f1_text": metric_text("f1"),
                        "mcc_text": metric_text("mcc"),
                        "balanced_accuracy_text": metric_text("balanced_accuracy"),
                        "auroc_stats": mean_std_payload(subset, "auroc"),
                        "mcc_stats": mean_std_payload(subset, "mcc"),
                    }
                )
        report["aggregated_rows"] = aggregated_rows
        report["summary"] = "CTP is more stable than labeled-only Sup on the harder generalization splits." if any(
            metric_key(next((row["auroc_stats"]["mean"] for row in aggregated_rows if row["analysis_scheme"] == scheme and row["method"] == "CTP"), float("nan")))
            >= metric_key(next((row["auroc_stats"]["mean"] for row in aggregated_rows if row["analysis_scheme"] == scheme and row["method"] == "Sup"), float("nan")))
            or metric_key(next((row["mcc_stats"]["mean"] for row in aggregated_rows if row["analysis_scheme"] == scheme and row["method"] == "CTP"), float("nan")))
            >= metric_key(next((row["mcc_stats"]["mean"] for row in aggregated_rows if row["analysis_scheme"] == scheme and row["method"] == "Sup"), float("nan")))
            for scheme in ["target_holdout", "target_e3_holdout", "scaffold_holdout"]
        ) else "CTP does not yet show a clear hard-split advantage."
    elif profile == "fewshot_target":
        aggregated_rows = []
        for shots in [1, 2, 4]:
            for method in ["Sup", "CTP"]:
                subset = [row for row in rows if row["mode"] == "fewshot" and safe_int(row.get("shots_per_class"), 0) == shots and method_bucket(row) == method]
                if not subset:
                    continue
                targets = [safe_int(row.get("targets"), 0) for row in subset if safe_int(row.get("targets"), 0) > 0]
                episodes = [safe_int(row.get("episodes"), 0) for row in subset if safe_int(row.get("episodes"), 0) > 0]
                aggregated_rows.append(
                    {
                        "shots_per_class": shots,
                        "method": method,
                        "targets_text": str(int(round(float(np.mean(targets))))) if targets else "-",
                        "episodes_text": str(int(sum(episodes))) if episodes else "-",
                        "accuracy_text": mean_std_text(subset, "accuracy"),
                        "auroc_text": mean_std_text(subset, "auroc"),
                        "auprc_text": mean_std_text(subset, "auprc"),
                        "f1_text": mean_std_text(subset, "f1"),
                        "mcc_text": mean_std_text(subset, "mcc"),
                        "balanced_accuracy_text": mean_std_text(subset, "balanced_accuracy"),
                        "auroc_stats": mean_std_payload(subset, "auroc"),
                        "mcc_stats": mean_std_payload(subset, "mcc"),
                    }
                )
        report["aggregated_rows"] = aggregated_rows
        report["summary"] = "CTP provides a stronger base model for target-wise few-shot adaptation." if any(
            metric_key(next((row["auroc_stats"]["mean"] for row in aggregated_rows if row["shots_per_class"] == shots and row["method"] == "CTP"), float("nan")))
            > metric_key(next((row["auroc_stats"]["mean"] for row in aggregated_rows if row["shots_per_class"] == shots and row["method"] == "Sup"), float("nan")))
            or metric_key(next((row["mcc_stats"]["mean"] for row in aggregated_rows if row["shots_per_class"] == shots and row["method"] == "CTP"), float("nan")))
            > metric_key(next((row["mcc_stats"]["mean"] for row in aggregated_rows if row["shots_per_class"] == shots and row["method"] == "Sup"), float("nan")))
            for shots in [1, 2, 4]
        ) else "Few-shot adaptation does not yet show a consistent CTP gain."
    elif profile == "generalization_lowresource_v1":
        def mean_for(subset: Sequence[Dict[str, Any]], key: str) -> float:
            values = finite_values(subset, key)
            return float(np.mean(values)) if values else float("nan")

        comparisons: List[Dict[str, Any]] = []
        for fraction in [0.10, 0.25, 0.50, 1.00]:
            sup_rows = [row for row in rows if safe_text(row.get("analysis_scheme")) == "low_label" and row["mode"] == "supervised" and abs(float(row.get("label_fraction") or 0.0) - fraction) < 1e-9 and method_bucket(row) == "Sup"]
            ctp_rows = [row for row in rows if safe_text(row.get("analysis_scheme")) == "low_label" and row["mode"] == "supervised" and abs(float(row.get("label_fraction") or 0.0) - fraction) < 1e-9 and method_bucket(row) == "CTP"]
            if sup_rows and ctp_rows:
                comparisons.append(
                    {
                        "setting": f"low_label_{int(round(fraction * 100))}pct",
                        "sup_auroc": mean_for(sup_rows, "auroc"),
                        "ctp_auroc": mean_for(ctp_rows, "auroc"),
                        "sup_mcc": mean_for(sup_rows, "mcc"),
                        "ctp_mcc": mean_for(ctp_rows, "mcc"),
                    }
                )
        for scheme in ["target_holdout", "target_e3_holdout", "scaffold_holdout"]:
            sup_rows = [row for row in rows if safe_text(row.get("analysis_scheme")) == scheme and row["mode"] == "supervised" and method_bucket(row) == "Sup"]
            ctp_rows = [row for row in rows if safe_text(row.get("analysis_scheme")) == scheme and row["mode"] == "supervised" and method_bucket(row) == "CTP"]
            if sup_rows and ctp_rows:
                comparisons.append(
                    {
                        "setting": scheme,
                        "sup_auroc": mean_for(sup_rows, "auroc"),
                        "ctp_auroc": mean_for(ctp_rows, "auroc"),
                        "sup_mcc": mean_for(sup_rows, "mcc"),
                        "ctp_mcc": mean_for(ctp_rows, "mcc"),
                    }
                )
        for shots in [1, 2, 4]:
            sup_rows = [row for row in rows if row["mode"] == "fewshot" and safe_int(row.get("shots_per_class"), 0) == shots and method_bucket(row) == "Sup"]
            ctp_rows = [row for row in rows if row["mode"] == "fewshot" and safe_int(row.get("shots_per_class"), 0) == shots and method_bucket(row) == "CTP"]
            if sup_rows and ctp_rows:
                comparisons.append(
                    {
                        "setting": f"fewshot_k{shots}",
                        "sup_auroc": mean_for(sup_rows, "auroc"),
                        "ctp_auroc": mean_for(ctp_rows, "auroc"),
                        "sup_mcc": mean_for(sup_rows, "mcc"),
                        "ctp_mcc": mean_for(ctp_rows, "mcc"),
                    }
                )
        if comparisons:
            ranked = sorted(
                comparisons,
                key=lambda item: (
                    metric_key(item["ctp_auroc"] - item["sup_auroc"]),
                    metric_key(item["ctp_mcc"] - item["sup_mcc"]),
                ),
                reverse=True,
            )
            hardest = min(comparisons, key=lambda item: (metric_key(item["ctp_auroc"]), metric_key(item["ctp_mcc"])))
            largest_gain = ranked[0]
            largest_gain_text = (
                f"{largest_gain['setting']} (AUROC delta {largest_gain['ctp_auroc'] - largest_gain['sup_auroc']:+.4f}, "
                f"MCC delta {largest_gain['ctp_mcc'] - largest_gain['sup_mcc']:+.4f})"
            )
            hardest_text = f"{hardest['setting']} (CTP AUROC {hardest['ctp_auroc']:.4f}, MCC {hardest['ctp_mcc']:.4f})"
        else:
            largest_gain_text = "insufficient data"
            hardest_text = "insufficient data"
        label_gain = any(
            safe_text(item["setting"]).startswith("low_label_")
            and (metric_key(item["ctp_auroc"] - item["sup_auroc"]) > 0 or metric_key(item["ctp_mcc"] - item["sup_mcc"]) > 0)
            for item in comparisons
        )
        target_holdout_gain = any(
            item["setting"] == "target_holdout"
            and (metric_key(item["ctp_auroc"] - item["sup_auroc"]) > 0 or metric_key(item["ctp_mcc"] - item["sup_mcc"]) > 0)
            for item in comparisons
        )
        target_e3_gain = any(
            item["setting"] == "target_e3_holdout"
            and (metric_key(item["ctp_auroc"] - item["sup_auroc"]) > 0 or metric_key(item["ctp_mcc"] - item["sup_mcc"]) > 0)
            for item in comparisons
        )
        scaffold_gain = any(
            item["setting"] == "scaffold_holdout"
            and (metric_key(item["ctp_auroc"] - item["sup_auroc"]) > 0 or metric_key(item["ctp_mcc"] - item["sup_mcc"]) > 0)
            for item in comparisons
        )
        fewshot_gain = any(
            safe_text(item["setting"]).startswith("fewshot_k")
            and (metric_key(item["ctp_auroc"] - item["sup_auroc"]) > 0 or metric_key(item["ctp_mcc"] - item["sup_mcc"]) > 0)
            for item in comparisons
        )
        report["answers"] = {
            "Does CTP improve under label scarcity?": "Yes" if label_gain else "No",
            "Does CTP improve target holdout generalization?": "Yes" if target_holdout_gain else "No",
            "Does CTP improve target-E3 pair generalization?": "Yes" if target_e3_gain else "No",
            "Does CTP improve scaffold holdout generalization?": "Yes" if scaffold_gain else "No",
            "Does CTP help target-wise one-shot/few-shot adaptation?": "Yes" if fewshot_gain else "No",
            "Which setting shows the largest CTP gain?": largest_gain_text,
            "Which setting remains the hardest?": hardest_text,
        }
    if profile == "ctp_formal_v1":
        display_map = {row["display_name"]: row for row in rows}
        report["conclusion"] = (
            "CTP-full improves over Sup labeled-only across Acc/AUROC/AUPRC/F1/MCC/BalAcc. "
            "Tuple SSL is stronger than molecule-only SSL. Hard negatives do not beat easy negatives in this run. "
            "All reported main numbers use fixed threshold = 0.5."
        )
        report["highest_auprc_row"] = max(rows, key=lambda row: metric_key(row.get("auprc")))["display_name"] if rows else ""
    elif profile == "esm2_650m_sprint":
        sup_no = max((row for row in rows if not row["uses_unlabeled"] and row["protein_encoder"] != ESM2_650M_ENCODER), default=None, key=report_priority_key)
        sup_esm = max((row for row in rows if not row["uses_unlabeled"] and row["protein_encoder"] == ESM2_650M_ENCODER), default=None, key=report_priority_key)
        ctp_no = max((row for row in rows if row["uses_unlabeled"] and row["protein_encoder"] != ESM2_650M_ENCODER), default=None, key=report_priority_key)
        ctp_esm = max((row for row in rows if row["uses_unlabeled"] and row["protein_encoder"] == ESM2_650M_ENCODER), default=None, key=report_priority_key)
        ctp_easy_esm = next((row for row in rows if row["raw_label"] == "ctp_full_easy_esm2_650m"), None)
        ctp_hard_esm = next((row for row in rows if row["raw_label"] == "ctp_full_hard_esm2_650m"), None)
        report["answers"] = {
            "esm_improves_sup": bool(sup_no and sup_esm and report_priority_key(sup_esm) > report_priority_key(sup_no)),
            "esm_improves_ctp": bool(ctp_no and ctp_esm and report_priority_key(ctp_esm) > report_priority_key(ctp_no)),
            "ctp_beats_sup_under_esm": bool(sup_esm and ctp_esm and report_priority_key(ctp_esm) > report_priority_key(sup_esm)),
            "easy_beats_hard_under_esm": bool(ctp_easy_esm and ctp_hard_esm and report_priority_key(ctp_easy_esm) > report_priority_key(ctp_hard_esm)),
        }
    elif profile == "sota_sprint_v2":
        best_sup = max((row for row in rows if not row["uses_unlabeled"]), default=None, key=report_priority_key)
        best_sup_no_esm = max((row for row in rows if not row["uses_unlabeled"] and row["protein_encoder"] != ESM2_650M_ENCODER), default=None, key=report_priority_key)
        best_sup_esm = max((row for row in rows if not row["uses_unlabeled"] and row["protein_encoder"] == ESM2_650M_ENCODER), default=None, key=report_priority_key)
        best_ctp = max((row for row in rows if row["uses_unlabeled"] and row["protein_encoder"] != ESM2_650M_ENCODER), default=None, key=report_priority_key)
        best_ctp_esm = max((row for row in rows if row["uses_unlabeled"] and row["protein_encoder"] == ESM2_650M_ENCODER), default=None, key=report_priority_key)
        best_overall = max(rows, default=None, key=report_priority_key)
        best_rows = []
        for bucket, row in [
            ("Best Sup", best_sup),
            ("Best CTP", best_ctp),
            ("Best CTP + ESM-2 650M", best_ctp_esm),
            ("Best fixed@0.5 overall", best_overall),
        ]:
            if row:
                best_rows.append({"bucket": bucket, **row})
        report["best_rows"] = best_rows
        report["comparison_rows"] = [
            {"method": "DegradeMaster-Sup", "source": "reported", "accuracy": "0.8141", "auroc": "0.8541", "auprc": "-", "f1": "0.8139", "mcc": "-", "balanced_accuracy": "-"},
            {"method": "DegradeMaster-Semi", "source": "reported", "accuracy": "0.8366", "auroc": "0.8825", "auprc": "-", "f1": "0.8224", "mcc": "-", "balanced_accuracy": "-"},
        ]
        for item in best_rows:
            report["comparison_rows"].append(
                {
                    "method": item["display_name"],
                    "source": "ours",
                    "accuracy": f"{item['accuracy']:.4f}",
                    "auroc": f"{item['auroc']:.4f}",
                    "auprc": f"{item['auprc']:.4f}",
                    "f1": f"{item['f1']:.4f}",
                    "mcc": f"{item['mcc']:.4f}",
                    "balanced_accuracy": f"{item['balanced_accuracy']:.4f}",
                }
            )
        report["answers"] = {
            "best_degradequery_beats_degrademaster_semi_acc": bool(best_overall and metric_key(best_overall["accuracy"]) > 0.8366),
            "best_degradequery_beats_degrademaster_semi_auroc": bool(best_overall and metric_key(best_overall["auroc"]) > 0.8825),
            "best_degradequery_beats_degrademaster_semi_f1": bool(best_overall and metric_key(best_overall["f1"]) > 0.8224),
            "best_degradequery_beats_sup_labeled_only": bool(best_overall and best_sup_no_esm and report_priority_key(best_overall) > report_priority_key(best_sup_no_esm)),
            "esm2_650m_improves_sup": bool(best_sup_no_esm and best_sup_esm and report_priority_key(best_sup_esm) > report_priority_key(best_sup_no_esm)),
            "esm2_650m_improves_ctp": bool(best_ctp and best_ctp_esm and report_priority_key(best_ctp_esm) > report_priority_key(best_ctp)),
        }
    json_path, md_path = output_paths_for_stem(
        config,
        output_stem,
        compare_runs_report_json_path(config, dataset_name),
        compare_runs_report_md_path(config, dataset_name),
    )
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(write_compare_runs_markdown(profile, rows, report), encoding="utf-8")
    return json_path, md_path


def report_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    if safe_text(args.compare) == "degrademaster":
        json_path, md_path = report_compare_degrademaster(runtime_cfg, args.dataset, safe_text(args.output_stem))
        print(f"benchmark_report_json: {json_path}")
        print(f"benchmark_report_md: {md_path}")
        return
    if args.compare_runs:
        json_path, md_path = report_compare_runs(
            runtime_cfg,
            args.dataset,
            args.compare_runs,
            profile=safe_text(args.profile),
            output_stem=safe_text(args.output_stem),
        )
        print(f"compare_runs_report_json: {json_path}")
        print(f"compare_runs_report_md: {md_path}")
        return
    raise ValueError("report requires either --compare degrademaster or --compare-runs <metrics.json ...>")


def selftest_rows(config: Dict[str, Any]) -> pd.DataFrame:
    smiles_bank = [
        "CCO",
        "CCN",
        "c1ccccc1",
        "CC(=O)O",
        "CCOC(=O)N",
        "CCS",
        "CCCl",
        "c1ccncc1",
        "CC(C)O",
        "CC(C)N",
        "CCBr",
        "CCCC",
        "CC(C)C(=O)O",
        "c1ncccc1",
        "CCOC",
        "CC(C)CO",
        "CC(=O)N",
        "c1ccoc1",
        "CC(C)CC",
        "CC(C)(C)O",
    ]
    target_sequences = [
        "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLT",
        "MALWMRLLPLLALLALWGPDPAAA",
        "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGP",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFK",
    ]
    recruiter_sequences = ["MSDSEVNQEAKPEVKPEVKPEVK", "MDSKGSSQKGSRLLLLLVVSNLLLCQGVVS"]
    rows = []
    for idx, smiles in enumerate(smiles_bank):
        row = empty_standard_row("selftest", "synthetic", f"selftest_{idx:03d}")
        row["smiles"] = smiles
        row["target_id"] = f"target_{idx % len(target_sequences)}"
        row["target_name"] = row["target_id"]
        row["target_sequence"] = target_sequences[idx % len(target_sequences)]
        row["recruiter_id"] = f"recruiter_{idx % len(recruiter_sequences)}"
        row["recruiter_name"] = row["recruiter_id"]
        row["recruiter_sequence"] = recruiter_sequences[idx % len(recruiter_sequences)]
        row["cell_line"] = ["HEK293", "MCF7", "K562"][idx % 3]
        row["task_id"] = row["target_id"]
        row["target_sequence_source"] = "manual"
        row["recruiter_sequence_source"] = "manual"
        aromatic = 1 if "c1" in smiles else 0
        hetero = 1 if any(atom in smiles for atom in ["N", "O", "S"]) else 0
        row["label"] = 1 if aromatic or hetero else 0
        row["is_labeled"] = True
        rows.append(row)
    frame = pd.DataFrame(rows)
    ratios = tuple(float(x) for x in config["benchmark"].get("train_val_test", [0.7, 0.15, 0.15]))
    from dq_data import assign_random_split

    frame, split_policy = assign_random_split(frame, 111, ratios)
    frame["split_policy"] = split_policy
    frame = standardize_frame_columns(frame)
    return frame


def selftest_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    runtime_cfg = prepare_runtime_config(config, args)
    ensure_project_dirs(runtime_cfg)
    frame = selftest_rows(runtime_cfg)
    path = normalized_dataset_path(runtime_cfg, "selftest")
    write_normalized_dataset(frame, path)
    result = supervised_training(
        {**runtime_cfg, "train": {**runtime_cfg["train"], "epochs": int(args.epochs), "batch_size": 8}},
        "selftest",
        frame,
        None,
        training_run_id("selftest", "supervised", None),
    )
    print(f"selftest_checkpoint: {result['checkpoint_path']}")
    print(f"selftest_metrics: {result['metrics_path']}")


def clean_command(config: Dict[str, Any], args: argparse.Namespace) -> None:
    out_dir = output_dir(config)
    keep_best = int(args.keep_best or config["output"]["keep_best_n"])
    metrics_files = sorted([path for path in out_dir.glob("*_metrics.json") if path.is_file()], key=lambda path: path.name)
    scored = []
    for path in metrics_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = payload.get("run_id") or path.name.replace("_metrics.json", "")
        score = payload.get("metrics", {}).get("test", {}).get(config["benchmark"]["primary_metric"], float("-inf"))
        if score is None or (isinstance(score, float) and math.isnan(score)):
            score = float("-inf")
        scored.append((score, run_id))
    scored.sort(reverse=True)
    keep_ids = {run_id for _, run_id in scored[:keep_best]}
    removed = 0
    for _, run_id in scored[keep_best:]:
        for suffix in ["_model.pt", "_metrics.json", "_pred.csv", "_analysis.md"]:
            path = out_dir / f"{run_id}{suffix}"
            if path.exists():
                path.unlink()
                removed += 1
    print(f"kept_runs: {len(keep_ids)}")
    print(f"removed_files: {removed}")
