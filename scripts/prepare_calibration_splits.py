from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dq_data import (  # noqa: E402
    default_analysis_overlay,
    load_config,
    load_normalized_dataset,
    standardize_analysis_split_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="splits/calibration")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_normalized_dataset(config, "protac8k")
    output_dir = Path(config["project"]["root"]) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = default_analysis_overlay(frame, "calibration_holdout", 0)
    official_train = frame["split"].eq("train")
    labeled_train = official_train & frame["is_labeled"].astype(bool)
    reports = {}

    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        val_ids = []
        for label in (0, 1):
            ids = frame.loc[labeled_train & frame["label"].astype(str).eq(str(label)), "row_id"].astype(str).to_numpy()
            rng.shuffle(ids)
            take = max(1, int(round(len(ids) * args.val_fraction)))
            val_ids.extend(ids[:take].tolist())
        val_id_set = set(val_ids)
        overlay = base.copy()
        overlay["analysis_scheme"] = "calibration_holdout"
        overlay["seed"] = seed
        val_mask = overlay["row_id"].isin(val_id_set)
        overlay.loc[val_mask, "split"] = "val"
        overlay["pretrain_use"] = official_train.to_numpy()
        overlay["supervised_use"] = overlay["split"].eq("train") & overlay["is_labeled"]
        overlay["label_visible"] = overlay["is_labeled"]
        overlay = standardize_analysis_split_overlay(overlay)
        path = output_dir / f"calibration_holdout_seed{seed}.csv"
        overlay.to_csv(path, index=False)
        reports[str(seed)] = {
            "path": str(path),
            "seed": seed,
            "val_fraction": args.val_fraction,
            "train_labeled": int((overlay["supervised_use"] & overlay["split"].eq("train")).sum()),
            "val_labeled": int((overlay["is_labeled"] & overlay["split"].eq("val")).sum()),
            "test_labeled": int((overlay["is_labeled"] & overlay["split"].eq("test")).sum()),
            "val_positive": int(overlay.loc[overlay["is_labeled"] & overlay["split"].eq("val"), "label"].astype(int).sum()),
        }

    report_path = output_dir / "calibration_split_report.json"
    report_path.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
