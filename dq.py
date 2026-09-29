from __future__ import annotations

import argparse

from dq_data import (
    DEFAULT_CONFIG_PATH,
    audit_command,
    ensure_project_dirs,
    fetch_command,
    fetchseq_command,
    init_command,
    load_config,
    prep_command,
    split_command,
    snapshot_command,
)
from dq_model import (
    baseline_command,
    clean_command,
    embedseq_command,
    eval_command,
    fewshot_command,
    pretrain_command,
    predict_command,
    report_command,
    selftest_command,
    train_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DegradeQuery minimal degradation predictor")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Optional runtime seed override")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Ensure project directories exist")

    snapshot_parser = subparsers.add_parser("snapshot", help="Rotate source snapshots")
    snapshot_parser.add_argument("--tag", default="", help="Optional snapshot tag")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch benchmark assets")
    fetch_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    fetch_parser.add_argument("--force", action="store_true")

    audit_parser = subparsers.add_parser("audit", help="Audit benchmark assets and official metadata")
    audit_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    audit_parser.add_argument("--check-leakage", action="store_true")

    prep_parser = subparsers.add_parser("prep", help="Normalize dataset tables")
    prep_parser.add_argument("--dataset", required=True, choices=["protac8k", "molglue", "mgtbind"])
    prep_parser.add_argument("--input", default="", help="Input table for molglue or mgtbind")

    fetchseq_parser = subparsers.add_parser("fetchseq", help="Recover protein sequences for a normalized dataset")
    fetchseq_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    fetchseq_parser.add_argument("--source", default="auto", choices=["auto"])

    split_parser = subparsers.add_parser("split", help="Build custom analysis split overlays")
    split_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    split_parser.add_argument(
        "--scheme",
        required=True,
        choices=[
            "official",
            "low_label",
            "target_holdout",
            "target_e3_holdout",
            "recruiter_holdout",
            "target_family_holdout",
            "scaffold_holdout",
            "fewshot_target",
        ],
    )
    split_parser.add_argument("--seed", required=True, type=int)
    split_parser.add_argument("--label-fraction", type=float, default=None)
    split_parser.add_argument("--test-frac", type=float, default=None)
    split_parser.add_argument("--shots-per-class", type=int, default=None)

    embedseq_parser = subparsers.add_parser("embedseq", help="Build frozen protein embedding cache")
    embedseq_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    embedseq_parser.add_argument("--encoder", required=True, choices=["esm2_650m"])

    pretrain_parser = subparsers.add_parser("pretrain", help="Run contrastive tuple pretraining on train split rows")
    pretrain_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    pretrain_parser.add_argument("--split-file", default="", help="Optional analysis split overlay CSV")
    pretrain_parser.add_argument("--tasks", required=True, help="Comma-separated tasks: tuple_contrast,mol_contrast")
    pretrain_parser.add_argument("--epochs", type=int, default=None)
    pretrain_parser.add_argument("--batch-size", type=int, default=None)
    pretrain_parser.add_argument(
        "--tuple-negative-policy",
        default=None,
        choices=[
            "easy",
            "full",
            "target_only",
            "e3_only",
            "joint_only",
            "full_mixed",
            "frequency_matched",
            "target_seq_cluster",
            "protein_nn",
            "hard_mixed",
        ],
    )
    pretrain_parser.add_argument(
        "--positive-construction",
        default=None,
        choices=["observed", "shuffled_pair", "shuffled_independent"],
    )
    pretrain_parser.add_argument("--lambda-tuple", type=float, default=None)
    pretrain_parser.add_argument("--lambda-mol", type=float, default=None)
    pretrain_parser.add_argument(
        "--temperature-tuple",
        type=float,
        default=None,
        help="Tuple contrastive temperature; defaults to pretrain.temperature for compatibility",
    )
    pretrain_parser.add_argument(
        "--temperature-mol",
        type=float,
        default=None,
        help="Molecule contrastive temperature; defaults to pretrain.temperature for compatibility",
    )
    pretrain_parser.add_argument("--protein-encoder", default=None, choices=["aac", "esm2_650m"])
    pretrain_parser.add_argument("--run-label", default="", help="Optional metadata label appended to run names")

    train_parser = subparsers.add_parser("train", help="Train supervised or semi-supervised model")
    train_parser.add_argument("--dataset", required=True, help="Dataset name")
    train_parser.add_argument("--mode", required=True, choices=["supervised", "semi"])
    train_parser.add_argument("--split-file", default="", help="Optional analysis split overlay CSV")
    train_parser.add_argument(
        "--ablation",
        default="",
        choices=[
            "",
            "smiles_only",
            "smiles_graph_fp",
            "smiles_target_id",
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
            "smiles_target",
            "smiles_target_recruiter",
            "smiles_target_recruiter_cell",
        ],
    )
    train_parser.add_argument("--init", default="", help="Optional pretrain checkpoint used to initialize shared encoder weights")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--pseudo-threshold", type=float, default=None)
    train_parser.add_argument("--pseudo-selector", default=None, choices=["threshold", "balanced_topk"])
    train_parser.add_argument("--pseudo-topk", type=int, default=None)
    train_parser.add_argument("--teacher-checkpoint", default="", help="Optional supervised checkpoint used as the pseudo-label teacher")
    train_parser.add_argument("--pseudo-weight", type=float, default=None)
    train_parser.add_argument("--pseudo-rounds", type=int, default=None)
    train_parser.add_argument("--protein-encoder", default=None, choices=["aac", "esm2_650m"])
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=None)
    train_parser.add_argument("--dropout", type=float, default=None)
    train_parser.add_argument("--loss", default=None, choices=["bce", "focal"])
    train_parser.add_argument("--pos-weight", default=None, help="auto or float")
    train_parser.add_argument("--focal-gamma", type=float, default=None)
    train_parser.add_argument("--focal-alpha", default=None, help="auto or float")
    train_parser.add_argument("--run-label", default="", help="Optional metadata label appended to run names")
    train_parser.add_argument("--allow-nonbenchmark", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--split-file", default="", help="Optional analysis split overlay CSV")
    eval_parser.add_argument("--threshold", default="0.5")
    eval_parser.add_argument("--run-label", default="", help="Optional metadata label appended to eval outputs")
    eval_parser.add_argument(
        "--test-filter",
        default="",
        choices=["", "no_exact_smiles_overlap", "no_target_smiles_overlap", "no_target_recruiter_smiles_overlap"],
    )

    fewshot_parser = subparsers.add_parser("fewshot", help="Run classifier-head-only target-wise few-shot adaptation")
    fewshot_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    fewshot_parser.add_argument("--split-file", required=True, help="Few-shot analysis split overlay CSV")
    fewshot_parser.add_argument("--checkpoint", required=True)
    fewshot_parser.add_argument("--shots-per-class", required=True, type=int)
    fewshot_parser.add_argument("--adapt", required=True, choices=["classifier_head"])
    fewshot_parser.add_argument("--adapt-steps", type=int, default=20)
    fewshot_parser.add_argument("--adapt-lr", type=float, default=1e-4)
    fewshot_parser.add_argument("--threshold", default="0.5")
    fewshot_parser.add_argument("--run-label", default="", help="Optional metadata label appended to outputs")

    predict_parser = subparsers.add_parser("predict", help="Predict on a query CSV")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--input", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Run local benchmark baselines")
    baseline_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    baseline_parser.add_argument("--methods", required=True)

    report_parser = subparsers.add_parser("report", help="Write benchmark comparison or run summary reports")
    report_parser.add_argument("--dataset", required=True, choices=["protac8k"])
    report_parser.add_argument("--compare", default="")
    report_parser.add_argument("--compare-runs", nargs="*", default=[])
    report_parser.add_argument("--profile", default="", help="Optional summary profile")
    report_parser.add_argument("--output-stem", default="", help="Optional output filename stem without extension")

    selftest_parser = subparsers.add_parser("selftest", help="Run synthetic end-to-end test")
    selftest_parser.add_argument("--epochs", type=int, default=2)

    clean_parser = subparsers.add_parser("clean", help="Keep only the best N runs")
    clean_parser.add_argument("--keep-best", type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)

    if args.command == "init":
        init_command(config, args)
    elif args.command == "snapshot":
        snapshot_command(config, args)
    elif args.command == "fetch":
        fetch_command(config, args)
    elif args.command == "audit":
        audit_command(config, args)
    elif args.command == "prep":
        prep_command(config, args)
    elif args.command == "fetchseq":
        fetchseq_command(config, args)
    elif args.command == "split":
        split_command(config, args)
    elif args.command == "embedseq":
        embedseq_command(config, args)
    elif args.command == "pretrain":
        pretrain_command(config, args)
    elif args.command == "train":
        train_command(config, args)
    elif args.command == "eval":
        eval_command(config, args)
    elif args.command == "fewshot":
        fewshot_command(config, args)
    elif args.command == "predict":
        predict_command(config, args)
    elif args.command == "baseline":
        baseline_command(config, args)
    elif args.command == "report":
        report_command(config, args)
    elif args.command == "selftest":
        selftest_command(config, args)
    elif args.command == "clean":
        clean_command(config, args)
    else:  # pragma: no cover
        parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
