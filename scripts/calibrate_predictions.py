from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dq_model import compute_metrics


def nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    truth = np.asarray(labels, dtype=float)
    return float(-(truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability)).mean())


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability))

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        return nll(labels, calibrated)

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(math.exp(result.x))


def metric_record(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = compute_metrics(labels, probabilities, 10, threshold=0.5)
    metrics["nll"] = nll(labels, probabilities)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature on validation predictions and evaluate on test predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_path = Path(args.output)
    frame = pd.read_csv(predictions_path)
    required = {"split", "label", "prob"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    validation = frame[frame["split"].eq("val")].copy()
    test = frame[frame["split"].eq("test")].copy()
    if validation.empty or test.empty:
        raise ValueError("Prediction file must contain non-empty val and test partitions.")

    validation_labels = validation["label"].astype(int).to_numpy()
    validation_probabilities = validation["prob"].astype(float).to_numpy()
    test_labels = test["label"].astype(int).to_numpy()
    test_probabilities = test["prob"].astype(float).to_numpy()
    temperature = fit_temperature(validation_labels, validation_probabilities)
    calibrated_probabilities = apply_temperature(test_probabilities, temperature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prediction_file": str(predictions_path),
        "temperature_fit_split": "val",
        "evaluation_split": "test",
        "temperature": temperature,
        "counts": {"val": int(len(validation)), "test": int(len(test))},
        "raw": metric_record(test_labels, test_probabilities),
        "temperature_scaled": metric_record(test_labels, calibrated_probabilities),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"calibration: {output_path}")


if __name__ == "__main__":
    main()
