# Runbook

This file lists the canonical commands to run common tasks.

## Step 0 — Scaffold verification

```bash
python -m scripts.train_teacher --help
python -m scripts.train_student_kd --help
python -m scripts.evaluate --help
python -m scripts.train_teacher --smoke --run-id smoke-test
```

## Step 1 — Data pipeline smoke

Set `data.path` in `configs/data.yaml` (CSV/Parquet), then:

```bash
python -m scripts.data_smoke --config configs/data.yaml --run-id data-smoke
```

