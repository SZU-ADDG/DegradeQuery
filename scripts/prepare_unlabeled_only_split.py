from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dq_data import (
    default_analysis_overlay,
    load_config,
    load_normalized_dataset,
    standardize_analysis_split_overlay,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output-dir",
        default="splits/official",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_normalized_dataset(config, "protac8k")
    root = Path(config["project"]["root"])
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = default_analysis_overlay(frame, "official_unlabeled_only", 0)
    official_train = overlay["split"].eq("train")
    labeled = overlay["is_labeled"].astype(bool)
    overlay["pretrain_use"] = official_train & ~labeled
    overlay["supervised_use"] = official_train & labeled
    overlay["label_visible"] = labeled
    overlay = standardize_analysis_split_overlay(overlay)

    path = output_dir / "official_unlabeled_only.csv"
    overlay.to_csv(path, index=False)
    test_mask = overlay["split"].eq("test") & overlay["is_labeled"]
    train_mask = overlay["supervised_use"]
    pretrain_mask = overlay["pretrain_use"]
    report = {
        "scheme": "official_unlabeled_only",
        "path": str(path),
        "sha256": sha256(path),
        "rows_total": int(len(overlay)),
        "pretrain_rows": int(pretrain_mask.sum()),
        "pretrain_labeled_rows": int((pretrain_mask & overlay["is_labeled"]).sum()),
        "supervised_train_rows": int(train_mask.sum()),
        "supervised_train_positive": int(overlay.loc[train_mask, "label"].astype(int).sum()),
        "test_rows": int(test_mask.sum()),
        "test_positive": int(overlay.loc[test_mask, "label"].astype(int).sum()),
        "official_test_row_ids_sha256": hashlib.sha256(
            "\n".join(sorted(overlay.loc[test_mask, "row_id"].astype(str))).encode()
        ).hexdigest(),
    }
    report_path = output_dir / "unlabeled_only_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"split: {path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
