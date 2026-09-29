from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dq_model import compute_metrics  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    base = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, prob, 1.0 - prob)
    return ((1.0 - pt).pow(gamma) * base).mean()


class ProtacDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, fingerprint_cache: Dict[str, np.ndarray]) -> None:
        self.records = frame.to_dict("records")
        self.fingerprint_cache = fingerprint_cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.records[index]
        return {
            "row_id": str(row["row_id"]),
            "split": str(row["split"]),
            "label": float(row["label"]),
            "smiles": str(row["smiles"]),
            "target_id": str(row["target_id"]),
            "recruiter_id": str(row["recruiter_id"]),
            "target_sequence": str(row["target_sequence"]),
            "recruiter_sequence": str(row["recruiter_sequence"]),
            "morgan": self.fingerprint_cache[str(row["smiles"])],
        }


def collate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "row_id": [record["row_id"] for record in records],
        "split": [record["split"] for record in records],
        "label": torch.tensor([record["label"] for record in records], dtype=torch.float32),
        "smiles": [record["smiles"] for record in records],
        "target_id": [record["target_id"] for record in records],
        "recruiter_id": [record["recruiter_id"] for record in records],
        "target_sequence": [record["target_sequence"] for record in records],
        "recruiter_sequence": [record["recruiter_sequence"] for record in records],
        "morgan": torch.from_numpy(np.stack([record["morgan"] for record in records])).float(),
    }


class PlmReference(nn.Module):
    def __init__(self, model_path: Path, hidden_dim: int = 256) -> None:
        super().__init__()
        self.esm = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        for parameter in self.esm.parameters():
            parameter.requires_grad = False
        layers = self.esm.encoder.layer
        for parameter in layers[-1].parameters():
            parameter.requires_grad = True
        if hasattr(self.esm, "emb_layer_norm_after"):
            for parameter in self.esm.emb_layer_norm_after.parameters():
                parameter.requires_grad = True
        protein_dim = int(self.esm.config.hidden_size)
        self.mol_proj = nn.Sequential(nn.Linear(2048, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, 1),
        )

    def encode_sequences(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.esm(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.float()

    def forward(
        self,
        morgan: torch.Tensor,
        target_tokens: Dict[str, torch.Tensor],
        recruiter_tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mol = self.mol_proj(morgan)
        target = self.protein_proj(self.encode_sequences(target_tokens))
        recruiter = self.protein_proj(self.encode_sequences(recruiter_tokens))
        fused = torch.cat(
            [
                mol,
                target,
                recruiter,
                mol * target,
                mol * recruiter,
                target * recruiter,
                torch.abs(target - recruiter),
            ],
            dim=-1,
        )
        return self.classifier(fused).squeeze(-1)


def token_batch(tokenizer: Any, sequences: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device, non_blocking=True) for key, value in encoded.items()}


def run_epoch(
    model: PlmReference,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    rows: List[Dict[str, Any]] = []
    for step, batch in enumerate(loader):
        labels = batch["label"].to(device)
        target_tokens = token_batch(tokenizer, batch["target_sequence"], max_length, device)
        recruiter_tokens = token_batch(tokenizer, batch["recruiter_sequence"], max_length, device)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["morgan"].to(device), target_tokens, recruiter_tokens)
            loss = focal_loss(logits.float(), labels, gamma=1.0)
        if training:
            (loss / accumulation_steps).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    5.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        losses.append(float(loss.detach().cpu()))
        for index, probability in enumerate(probs):
            rows.append(
                {
                    "row_id": batch["row_id"][index],
                    "split": batch["split"][index],
                    "label": int(batch["label"][index].item()),
                    "prob": float(probability),
                    "pred": int(probability >= 0.5),
                    "smiles": batch["smiles"][index],
                    "target_id": batch["target_id"][index],
                    "recruiter_id": batch["recruiter_id"][index],
                }
            )
    prediction = pd.DataFrame(rows)
    metrics = compute_metrics(prediction["label"], prediction["prob"], 10) if not prediction.empty else {}
    return {"loss": float(np.mean(losses)), "metrics": metrics, "prediction": prediction}


def selection_key(metrics: Dict[str, Any]) -> tuple[float, float, float]:
    def value(key: str) -> float:
        number = float(metrics.get(key, float("-inf")))
        return number if math.isfinite(number) else float("-inf")

    return value("accuracy"), value("f1"), value("auroc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    thread_count = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_threads(thread_count)
    root = Path(args.root)
    output_dir = root / "output/plm"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "data/protac8k_normalized.csv")
    labeled = frame[frame["is_labeled"].astype(bool) & frame["split"].isin(["train", "test"])].copy()
    train = labeled[labeled["split"].eq("train")].copy()
    test = labeled[labeled["split"].eq("test")].copy()
    if args.limit_train > 0:
        train = train.sample(n=min(args.limit_train, len(train)), random_state=args.seed)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint_cache: Dict[str, np.ndarray] = {}
    for smiles in labeled["smiles"].astype(str).unique():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprint_cache[smiles] = np.zeros(2048, dtype=np.float32)
        else:
            fingerprint_cache[smiles] = generator.GetFingerprintAsNumPy(mol).astype(np.float32)

    device = torch.device("cuda")
    model_path = root / "data/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = PlmReference(model_path).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
        weight_decay=1e-4,
    )
    train_loader = DataLoader(
        ProtacDataset(train, fingerprint_cache),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    train_eval_loader = DataLoader(
        ProtacDataset(train, fingerprint_cache),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ProtacDataset(test, fingerprint_cache),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    best_epoch = -1
    best_trainable: Dict[str, torch.Tensor] = {}
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        train_fit = run_epoch(
            model,
            train_loader,
            tokenizer,
            device,
            args.max_length,
            optimizer,
            args.accumulation_steps,
        )
        train_eval = run_epoch(
            model,
            train_eval_loader,
            tokenizer,
            device,
            args.max_length,
            None,
            args.accumulation_steps,
        )
        key = selection_key(train_eval["metrics"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_fit["loss"],
                "train_accuracy": train_eval["metrics"].get("accuracy"),
                "train_f1": train_eval["metrics"].get("f1"),
                "train_auroc": train_eval["metrics"].get("auroc"),
            }
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_trainable = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    named_parameters = dict(model.named_parameters())
    for name, value in best_trainable.items():
        named_parameters[name].data.copy_(value.to(named_parameters[name].device))
    train_final = run_epoch(
        model,
        train_eval_loader,
        tokenizer,
        device,
        args.max_length,
        None,
        args.accumulation_steps,
    )
    test_final = run_epoch(
        model,
        test_loader,
        tokenizer,
        device,
        args.max_length,
        None,
        args.accumulation_steps,
    )
    prediction = pd.concat([train_final["prediction"], test_final["prediction"]], ignore_index=True)
    stem = f"plm_last_layer_{args.run_label}"
    checkpoint_path = output_dir / f"{stem}_trainable.pt"
    metrics_path = output_dir / f"{stem}_metrics.json"
    pred_path = output_dir / f"{stem}_pred.csv"
    torch.save(
        {
            "trainable_state_dict": best_trainable,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "model_path": str(model_path),
            "max_length": args.max_length,
        },
        checkpoint_path,
    )
    metrics = {
        "run_id": stem,
        "run_label": args.run_label,
        "seed": args.seed,
        "mode": "supervised",
        "protein_encoder": "esm2_650m_last_layer",
        "selection_scope": "train",
        "selection_rule": "acc_then_f1_then_auroc_at_fixed_0p5",
        "reporting_protocol": "paper_authoritative_no_test_model_selection",
        "selected_epoch": best_epoch,
        "threshold": 0.5,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "history": history,
        "metrics": {"train": train_final["metrics"], "test": test_final["metrics"]},
        "counts": {"train": int(len(train)), "test": int(len(test))},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    prediction.to_csv(pred_path, index=False)
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics: {metrics_path}")
    print(f"predictions: {pred_path}")


if __name__ == "__main__":
    main()
