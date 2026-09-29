# Reproducing the experiments

Run all commands from the repository root after preparing PROTAC-8K as described in the main README. Every command writes its configuration, metrics, predictions, and checkpoint under `output/`. The fixed split overlays under `splits/` identify every row used for pretraining, supervised training, validation, and evaluation.

## Main benchmark

The five-seed supervised and full-CTP comparison is automated by:

```bash
python scripts/reproduce_official.py --seeds 1,2,3,4,5
```

Add `--prepare-data` on a fresh checkout. This runner uses the submitted architecture, fixed threshold 0.5, focal loss with gamma 1.0, and train-side model selection. The full CTP setting combines `tuple_contrast` and `mol_contrast` with three counterfactual replacements per observed tuple.

## Two-stage CTP pattern

The remaining CTP experiments use the same two commands. First run pretraining:

```bash
python dq.py --seed 1 pretrain \
  --dataset protac8k \
  --tasks tuple_contrast,mol_contrast \
  --epochs 60 \
  --batch-size 64 \
  --tuple-negative-policy full \
  --temperature-tuple 0.1 \
  --temperature-mol 0.1 \
  --lambda-tuple 1.0 \
  --lambda-mol 0.1 \
  --protein-encoder aac \
  --run-label ctp_pretrain_s1
```

The command prints the checkpoint path as `checkpoint: ...`. Pass that path to supervised fine-tuning:

```bash
python dq.py --seed 1 train \
  --dataset protac8k \
  --mode supervised \
  --ablation ctp_finetune \
  --init PATH_PRINTED_BY_PRETRAIN \
  --epochs 120 \
  --batch-size 64 \
  --lr 0.0001 \
  --loss focal \
  --focal-gamma 1.0 \
  --protein-encoder aac \
  --run-label ctp_s1
```

Repeat with the reported seed set. When a split overlay is used, pass the same `--split-file` to both stages.

## Component attribution

Molecule-only self-supervision:

```bash
python dq.py --seed 1 pretrain \
  --dataset protac8k --tasks mol_contrast \
  --epochs 60 --batch-size 64 \
  --temperature-mol 0.1 --lambda-mol 0.1 \
  --protein-encoder aac --run-label mol_only_pretrain_s1
```

Tuple-only pretraining:

```bash
python dq.py --seed 1 pretrain \
  --dataset protac8k --tasks tuple_contrast \
  --epochs 60 --batch-size 64 \
  --tuple-negative-policy full \
  --temperature-tuple 0.1 --lambda-tuple 1.0 --lambda-mol 0.0 \
  --protein-encoder aac --run-label tuple_only_pretrain_s1
```

Fine-tune each checkpoint with the two-stage pattern above. The component results use seeds 1, 2, and 3.

## Unlabeled-only CTP

The supplied overlay retains all supervised rows for downstream training and restricts pretraining to 7,134 label-missing training records:

```bash
python dq.py --seed 1 pretrain \
  --dataset protac8k \
  --split-file splits/official/official_unlabeled_only.csv \
  --tasks tuple_contrast,mol_contrast \
  --epochs 60 --batch-size 64 \
  --tuple-negative-policy full \
  --positive-construction observed \
  --temperature-tuple 0.1 --temperature-mol 0.1 \
  --lambda-tuple 1.0 --lambda-mol 0.1 \
  --run-label unlabeled_only_pretrain_s1
```

Fine-tune with the same overlay and the printed checkpoint:

```bash
python dq.py --seed 1 train \
  --dataset protac8k --mode supervised \
  --split-file splits/official/official_unlabeled_only.csv \
  --ablation ctp_finetune --init PATH_PRINTED_BY_PRETRAIN \
  --epochs 120 --batch-size 64 --lr 0.0001 \
  --loss focal --focal-gamma 1.0 \
  --run-label unlabeled_only_s1
```

The reported analysis uses seeds 1 through 5.

## Counterfactual construction and sensitivity

Use the two-stage pattern with one of the following `--tuple-negative-policy` values:

- `target_only`
- `e3_only`
- `joint_only`
- `full_mixed`
- `frequency_matched`
- `hard_mixed`

The policy comparison uses seeds 1, 2, and 3. Temperature sensitivity changes one of `--temperature-tuple` or `--temperature-mol` to 0.05 or 0.20 while holding the other at 0.10. Loss-weight sensitivity changes `--lambda-mol` to 0.05, 0.20, or 0.50 while keeping `--lambda-tuple 1.0`.

The relational-null control keeps the complete training protocol fixed and changes only positive tuple construction:

```bash
python dq.py --seed 1 pretrain \
  --dataset protac8k \
  --tasks tuple_contrast,mol_contrast \
  --epochs 60 --batch-size 64 \
  --tuple-negative-policy full \
  --positive-construction shuffled_pair \
  --temperature-tuple 0.1 --temperature-mol 0.1 \
  --lambda-tuple 1.0 --lambda-mol 0.1 \
  --run-label shuffled_pair_pretrain_s1
```

## ESM2-650M comparison

Build the frozen protein embedding cache:

```bash
python dq.py embedseq --dataset protac8k --encoder esm2_650m
```

Run the supervised reference by replacing `--protein-encoder aac` with `--protein-encoder esm2_650m` in the supervised command. Run CTP with the same replacement in both pretraining and fine-tuning. In both cases, ESM2 supplies target and E3 representations inside the same molecule-target-E3 predictor.

The matched last-layer fine-tuning reference expects the Hugging Face ESM2-650M files under `data/esm2_t33_650M_UR50D`:

```bash
python scripts/train_plm_last_layer.py \
  --seed 1 --epochs 10 --batch-size 2 \
  --accumulation-steps 16 --max-length 512 \
  --run-label esm2_last_layer_s1
```

The comparison uses seeds 1, 2, and 3. BF16 is used for the ESM2 forward pass, and only the final ESM2 encoder layer, its final normalization where present, and the PROTAC prediction layers are trainable.

## Target-E3 group holdout

Ten deterministic overlays are provided under `splits/target_e3/`. For each seed, run supervised training and CTP with the matching overlay:

```bash
python dq.py --seed 1 train \
  --dataset protac8k --mode supervised \
  --split-file splits/target_e3/target_e3_holdout_seed1.csv \
  --epochs 120 --batch-size 64 --lr 0.0001 \
  --loss focal --focal-gamma 1.0 \
  --run-label target_e3_sup_s1
```

For CTP, add the same split path to both commands in the two-stage pattern. Replace the seed and filename together for seeds 1 through 10.

## Scaffold holdout

Generate a deterministic split and train both methods on the resulting overlay:

```bash
python dq.py split \
  --dataset protac8k --scheme scaffold_holdout \
  --seed 1 --test-frac 0.2
```

The split command prints the generated overlay path. Supply that path through `--split-file` in supervised training and in both CTP stages.

## Target-wise few-shot adaptation

Generate the requested episode definition:

```bash
python dq.py split \
  --dataset protac8k --scheme fewshot_target \
  --seed 1 --shots-per-class 1
```

Adapt only the classifier head of a trained Sup or CTP checkpoint:

```bash
python dq.py fewshot \
  --dataset protac8k \
  --split-file PATH_PRINTED_BY_SPLIT \
  --checkpoint TRAINED_CHECKPOINT \
  --shots-per-class 1 \
  --adapt classifier_head \
  --adapt-steps 20 --adapt-lr 0.0001 \
  --threshold 0.5 \
  --run-label fewshot_k1_s1
```

Use `--shots-per-class 3` or `5` for the other reported settings.

## Train-side calibration splits

The supplied calibration overlays place a stratified 20% portion of the labeled training set in `val`; the official evaluation set remains unchanged:

```bash
python dq.py --seed 1 train \
  --dataset protac8k --mode supervised \
  --split-file splits/calibration/calibration_holdout_seed1.csv \
  --epochs 120 --batch-size 64 --lr 0.0001 \
  --loss focal --focal-gamma 1.0 \
  --run-label calibration_sup_s1
```

Run CTP with the same overlay through the two-stage pattern. Seeds 1, 2, and 3 correspond to the three supplied calibration partitions. The calibration experiment fits temperature only on the `val` predictions and evaluates the fitted temperature on the unchanged test predictions.

Apply temperature scaling to each run's prediction file:

```bash
python scripts/calibrate_predictions.py \
  --predictions OUTPUT_PREDICTION_CSV \
  --output output/calibration_sup_s1.json
```

## Result records

The JSON file printed after every run contains the selected epoch, selection scope, fixed threshold, complete runtime configuration, split metadata, counts, and all evaluation metrics. The adjacent prediction CSV enables independent recomputation of AUROC, AUPRC, accuracy, balanced accuracy, F1, MCC, Brier score, and ECE. `output/run_manifest.jsonl` provides a machine-readable index of completed runs.
