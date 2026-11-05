# Fixed split overlays

Each CSV is joined to the normalized PROTAC-8K table through `row_id`. The overlay records the train, validation, test, support, or query role used by the corresponding analysis.

- `official/`: official-split unlabeled-only pretraining control.
- `target_e3/`: ten independently constructed target-E3 group holdouts with held-out pairs excluded from pretraining and supervised training.
- `calibration/`: three train-side calibration partitions used for post-hoc temperature scaling.
- [`fewshot/`](fewshot): nine original target-wise splits for Table 4. Each file corresponds to 2, 4 or 8 support examples per class and one of three seeds. Use the same split for base training and classifier-head adaptation. [Reproduction commands](../docs/REPRODUCIBILITY.md#target-wise-few-shot-adaptation)

The accompanying JSON reports record the split counts and construction parameters.
