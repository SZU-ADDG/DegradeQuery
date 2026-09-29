# Fixed split overlays

Each CSV is joined to the normalized PROTAC-8K table through `row_id`. The overlay records the train, validation, test, support, or query role used by the corresponding analysis.

- `official/`: official-split unlabeled-only pretraining control.
- `target_e3/`: ten independently constructed target-E3 group holdouts with held-out pairs excluded from pretraining and supervised training.
- `calibration/`: three train-side calibration partitions used for post-hoc temperature scaling.

The accompanying JSON reports record the split counts and construction parameters.
