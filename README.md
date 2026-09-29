# DegradeQuery

Official implementation of **DegradeQuery: Counterfactual Tuple Pretraining for Context-Aware PROTAC Degradation Prediction**.

DegradeQuery learns from label-missing molecule-target-E3 records by contrasting recorded tuples with counterfactual target and E3 replacements. The pretrained tuple encoder is then fine-tuned for binary PROTAC degradation prediction.

## Repository layout

```text
DegradeQuery/
  dq.py                 Command-line interface
  dq_data.py            Data download, normalization, auditing, and split construction
  dq_model.py           Model, pretraining, fine-tuning, evaluation, and reporting
  config.yaml           Paper configuration
  requirements.txt      Python dependencies
  scripts/              Reproduction and split-generation utilities
  splits/               Fixed split overlays used in the paper analyses
  results/              Aggregate values reported in the paper
  docs/                 Detailed reproduction instructions
  data/                  Downloaded and processed data, created locally
  output/                Checkpoints, predictions, metrics, and run manifests
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The reported experiments were run with Python 3.11.15, PyTorch 2.5.1, CUDA 12.1, NumPy 2.4.3, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, RDKit 2026.03.1, and Transformers 5.5.0.

## Quick verification

```bash
python dq.py init
python dq.py selftest --epochs 2
```

The self-test creates a small synthetic dataset, trains the complete supervised pipeline, and writes a checkpoint and metrics file under `output/`.

## Prepare PROTAC-8K

```bash
python dq.py fetch --dataset protac8k
python dq.py audit --dataset protac8k --check-leakage
python dq.py prep --dataset protac8k
python dq.py fetchseq --dataset protac8k --source auto
```

The benchmark source remains governed by its original terms. Downloaded data and model caches are kept under `data/` and excluded from version control.

## Reproduce the official Sup and CTP runs

The command below prepares the benchmark when requested and runs the five paper seeds for the supervised baseline and full CTP:

```bash
python scripts/reproduce_official.py --prepare-data --seeds 1,2,3,4,5
```

For an already prepared dataset:

```bash
python scripts/reproduce_official.py --seeds 1,2,3,4,5
```

Each CTP run uses 60 pretraining epochs, three counterfactuals per recorded tuple, tuple and molecule temperatures of 0.1, loss weights of 1.0 and 0.1, and 120 supervised fine-tuning epochs. Fine-tuning uses focal loss with gamma 1.0, learning rate 1e-4, and a fixed decision threshold of 0.5. Epoch selection uses train-side validation data when available; evaluation rows do not participate in training, model selection, or threshold tuning.

## Fixed analysis splits

- `splits/official/official_unlabeled_only.csv`: pretraining uses 7,134 label-missing rows and zero labeled rows.
- `splits/target_e3/target_e3_holdout_seed*.csv`: ten fixed target-E3 group holdouts.
- `splits/calibration/calibration_holdout_seed*.csv`: three train-side calibration partitions.

The split files can be regenerated deterministically:

```bash
python scripts/prepare_unlabeled_only_split.py
python scripts/prepare_target_e3_splits.py
python scripts/prepare_calibration_splits.py
```

Detailed commands for the component analysis, negative-sampling policies, ESM2-650M comparison, target-E3 group holdout, scaffold holdout, few-shot adaptation, relational-null control, and calibration analysis are provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The aggregate values reported for these experiments are collected in [results/paper_results.csv](results/paper_results.csv).

## Outputs

Every training run writes:

- a model checkpoint;
- a JSON metrics record containing the configuration and protocol metadata;
- test predictions when enabled;
- an entry in `output/run_manifest.jsonl`.

## Citation

```bibtex
@inproceedings{xu2026degradequery,
  title={DegradeQuery: Counterfactual Tuple Pretraining for Context-Aware PROTAC Degradation Prediction},
  author={Xu, Dong and Yang, Zhangfan and Wu, Jiantao and Zhu, Zexuan and Li, Jianqiang and Ji, Junkai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

The source code is released under the MIT License. Third-party datasets and pretrained models retain their original licenses and terms.
