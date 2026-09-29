# Data directory

Run the following commands from the repository root:

```bash
python dq.py fetch --dataset protac8k
python dq.py audit --dataset protac8k --check-leakage
python dq.py prep --dataset protac8k
python dq.py fetchseq --dataset protac8k --source auto
```

The commands populate this directory with the downloaded PROTAC-8K assets, normalized table, protein-sequence cache, and feature caches. Dataset files are excluded from version control and remain subject to the original source terms.
