# Project Plan — Distributed Knowledge Distillation Transformer for Financial Fraud Detection

This is the **single authoritative implementation plan** for this repo (modeled after the “master-plan + workflow gates” style used in `ssm_calender` and `Jyotish AI`).

It translates the paper **“A Distributed Knowledge Distillation Framework for Financial Fraud Detection Based on Transformer” (IEEE Access, 2024)** into an end-to-end build spec with:

- a clear **system definition**
- an explicit **status ledger** (what’s done vs not done)
- a **decision tree** for what to do next
- a strict **ship workflow** (implement → verify → docs)
- a complete **module and script checklist**

If something conflicts with other files, this plan wins for technical direction.

---

## 0) Identity, scope, and success criteria

### 0.1 One-sentence mission

Build a **Transformer-based tabular fraud detector** and a **distributed knowledge distillation framework** that trains **smaller domain student models** (manufacturing vs other industries) and evaluates them against classical ML baselines—**reproducibly**.

### 0.2 What “distributed” means in this repo

We will implement the paper’s practical distributed setting:

- **Domain split**: `manufacturing` vs `other` industries
- **Student models**: 2 students (one per domain)
- **Teacher**: support both:
  - **single teacher** (baseline) and
  - **multi-teacher** (optional extension: per-domain teachers)

### 0.3 Definition of “done” (non-negotiable)

This repo is complete when:

- **Teacher training** runs end-to-end and saves checkpoints + metrics.
- **KD student training** runs end-to-end and saves:
  - a manufacturing student checkpoint
  - an other-industries student checkpoint
- **Evaluation** outputs:
  - accuracy/precision/recall/F1 + ROC-AUC (MCC optional)
  - ROC/PR plots
  - comparison vs sklearn baselines
- **Reproducibility**:
  - configs exist for data/model/train
  - seeds are set
  - every run writes artifacts (metrics + plots + config snapshot)

---

## 1) Project status ledger (update as you implement)

### 1.1 Implemented (as of initial setup)

- `docs/PROJECT_PLAN.md` exists (this plan)
- repo initialized; `.gitignore` includes `*.pdf` (paper PDF should not be committed)

### 1.2 Not implemented yet (must be built)

- `src/` codebase (data/model/train/eval)
- dependency file (`requirements.txt` or `pyproject.toml`)
- scripts/CLIs
- tests
- experiment artifact logging

---

## 2) Decision tree — what to do next (every session)

Follow this decision tree (same spirit as Jyotish/SSM “gates”):

```
┌─ Do we have a runnable environment (deps installed) + any checks?
│    │
│    ├─ NO  → Implement Step 0 (scaffold + deps + minimal smoke command).
│    │
│    └─ YES → Do checks pass?
│              │
│              ├─ NO  → Fix the failing check first. Do not add new features.
│              │
│              └─ YES → Is dataset ingestion + preprocessing done?
│                        │
│                        ├─ NO  → Implement Step 1 (dataset + splits + preprocessing).
│                        │
│                        └─ YES → Build forward in Step order (2 → 8).
```

---

## 3) Ship workflows (modeled after SSM/Jyotish)

These are the only two workflows agents should use.

### 3.1 Ship Feature workflow (default)

Implement → verify → docs.

1) Read: relevant section of this plan + target code files.
2) Implement: smallest change that compiles/runs.
3) Verify: run the relevant commands in §9.
4) Docs: update `docs/` with any new decision/interface/run command.
5) Report back with: what changed + commands + next step.

### 3.2 Ship Fix workflow (when anything is broken)

Investigate → reproduce → fix → regression proof.

1) Reproduce the failure (tests/script crash).
2) Identify root cause (do not guess).
3) Minimal fix.
4) Add/extend a test or a deterministic smoke check.
5) Re-run verification (must be green).

---

## 4) What we are building (system overview)

### 1.1 Inputs / outputs (core task)

- **Input**: a row (company-year / company-period) of financial indicators + an industry label/group.
- **Output**: probability of fraud (binary classification).
- **Constraints**:
  - Handle **heterogeneous indicators** across industries (missingness, different relevance)
  - Handle **class imbalance** (fraud is usually rare)
  - Produce a **smaller/faster model** for deployment (student)

### 1.2 High-level components

We will implement these components as separate modules:

- **Data layer**
  - dataset loading and preprocessing
  - splitting into **manufacturing** vs **other industries**
  - train/val/test splits
  - dataloaders, normalization, missing-value handling
- **Models**
  - **Teacher model**: Transformer encoder classifier (larger)
  - **Student model**: smaller Transformer encoder classifier (fewer heads)
- **Training pipelines**
  - teacher training
  - student training with knowledge distillation loss
- **Evaluation**
  - metrics: accuracy, precision, recall, F1, AUC (and MCC optionally)
  - plots: ROC and PR curves
  - baselines: LR, linear SVM, DT, RF, XGBoost, AdaBoost
- **Distributed inference wrapper**
  - route input to the correct student model (manufacturing vs other) during inference
- **Reproducibility**
  - configs, seeds, logging, experiment artifacts, checkpointing

---

## 2) Paper requirements mapped to implementation details

### 2.1 Transformer fraud detection model (Section III)

The paper uses a standard Transformer encoder block on tabular “feature tokens”.

#### 2.1.1 Tokenization / feature representation (tabular → tokens)

We must define a deterministic mapping from a tabular feature vector to an input matrix:

- Let there be \(m\) financial indicators (after preprocessing).
- Each indicator becomes a “token”.
- Each token is embedded into \(d_{model}\).

Implementation plan:

- **Numeric feature value embedding**:
  - Option A (simple, standard): `Linear(1 → d_model)` applied per feature value, plus a learned feature-id embedding.
  - Option B (more expressive): small MLP on the scalar value per feature, plus feature-id embedding.
- **Missing values**:
  - Keep a `missing_mask` per feature (1 if missing else 0), embed it and add to token embedding, or impute with 0 after normalization and provide mask embedding.
- **Normalization**:
  - Fit scaler on training split only (per domain if needed).

Deliverable: `TabularTokenizer` that returns `X_tokens: (batch, m, d_model)` (and optionally `mask`).

#### 2.1.2 Encoder block (multi-head attention + FFN)

Use PyTorch’s `nn.TransformerEncoderLayer` OR implement manually. Requirements from paper:

- Multi-head attention (self-attention on feature tokens)
- Residual connections + layer norm
- FFN with ReLU and `dim_ff=1024`
- Dropout `p=0.2`
- Encoder depth `layers=2`

Deliverable: `TransformerFraudClassifier` with configurable:

- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
- `dropout`
- classifier head (pooling strategy defined below)

#### 2.1.3 Pooling + classifier head

Transformer outputs per-token embeddings. We need a sample-level embedding to classify.

Implementation options (choose one and keep consistent):

- **Mean pooling** across tokens (simple, stable)
- **[CLS] token** appended and use its representation

Deliverable: stable pooling method, included in `TransformerFraudClassifier`.

#### 2.1.4 Teacher loss

Teacher is trained with hard-label cross entropy.

Deliverable: `train_teacher.py` that outputs:

- trained teacher checkpoint(s)
- metrics on validation/test

---

### 2.2 Distributed knowledge distillation framework (Section IV)

#### 2.2.1 Domain split (“distributed”)

The paper uses the TipDM dataset and splits:

- manufacturing industry (large)
- other industries (all remaining)

Deliverable:

- dataset splitter that maps each record to `domain ∈ {manufacturing, other}`
- training separate student models:
  - `student_manufacturing`
  - `student_other`

#### 2.2.2 Teacher(s)

The text describes “multi-teacher”, but Algorithm 1 outputs a single teacher parameter set. We will implement a plan that supports both while staying faithful:

- **Phase 1 (baseline)**: a single teacher trained on all data (or on combined domains).
- **Phase 2 (paper-leaning extension)**: teachers per domain/industry group, i.e.:
  - `teacher_manufacturing`
  - `teacher_other`

Deliverable: a `TeacherEnsemble` interface:

- `predict_logits(batch, domain) -> logits`
- supports one-teacher or multi-teacher setups without changing student training code.

#### 2.2.3 Student model size (Algorithm 2 hyperparams)

Student is a smaller Transformer with:

- `nhead=2`
- `layers=2`
- `dim_ff=1024`
- `dropout=0.2`

Deliverable: student config(s) and checkpoints per domain.

#### 2.2.4 Distillation temperature

Temperature \(T\) (paper uses `Tem=7`).

Deliverable: implement temperature softmax consistently:

- `soft_targets_teacher = softmax(logits_teacher / T)`
- `soft_targets_student = softmax(logits_student / T)`

#### 2.2.5 Distillation loss + student loss + combined loss

Paper equations:

- KD loss: KL divergence between teacher and student soft targets, scaled by \(T^2\).
- Student loss: cross entropy between student predictions (T=1) and hard labels.
- Total: \(L_{tr} = \alpha L_{cls} + \beta L_{KD}\)

Deliverable:

- `losses.py` implementing:
  - `kd_kl_loss(logits_s, logits_t, T)`
  - `ce_loss(logits_s, y)`
  - `total_loss(alpha, beta, ...)`

Note: paper excerpt doesn’t provide exact \(\alpha,\beta\). We will:

- default to a common KD setting such as `alpha=0.5, beta=0.5` (or `alpha=1.0, beta=1.0` depending on scaling)
- document chosen values in configs and allow overrides

---

## 3) Repository structure we will create

Proposed folder layout (all under this repo):

- `docs/`
  - `PROJECT_PLAN.md` (this file)
  - additional docs added later (design notes, dataset notes, usage)
- `src/`
  - `config/` (YAML/JSON configs)
  - `data/`
    - loaders, preprocessing, splits
  - `models/`
    - teacher/student Transformer model
    - tokenization/embedding for tabular inputs
  - `train/`
    - teacher training loop
    - student KD training loop
  - `eval/`
    - metrics, plotting, baseline comparisons
  - `utils/`
    - seed, logging, checkpoint IO, device utils
- `scripts/`
  - CLI entrypoints: `train_teacher`, `train_student_kd`, `evaluate`, etc.
- `configs/`
  - `teacher.yaml`, `student.yaml`, `data.yaml` (or domain-specific variants)
- `artifacts/` (gitignored)
  - runs, checkpoints, plots

---

## 4) Step-by-step build plan (everything we must implement)

This is the execution roadmap. Each step includes “definition of done”.

### Step 0 — Project scaffolding and reproducibility

**Implement**
- Python project skeleton (`src/`, `scripts/`, `configs/`)
- dependency management (`requirements.txt` or `pyproject.toml`)
- CLI argument parsing for main scripts
- deterministic runs:
  - random seeds (`python`, `numpy`, `torch`)
  - device selection (CPU/GPU)
- logging:
  - console logs
  - metrics saved to JSON/CSV

**Done when**
- `python -m scripts.train_teacher --help` works
- runs create an artifact folder with config snapshot + metrics

### Step 1 — Dataset ingestion and preprocessing

**Implement**
- A dataset loader for TipDM format (we’ll support CSV/Parquet).
- A preprocessing pipeline:
  - select feature columns
  - handle non-numeric columns (drop or encode if needed)
  - impute missing values (strategy documented)
  - scale numeric features (fit on train split, apply to val/test)
- A domain splitter:
  - manufacturing vs other
- Train/val/test splits:
  - 70/30 as paper (and optionally an extra test split if we want)
  - stratification by label if possible

**Done when**
- You can iterate batches:
  - returns `(X, y, domain, optional industry_id)`
  - `X` shape is consistent across domains

### Step 2 — Tabular tokenization (feature tokens)

**Implement**
- `TabularTokenizer`:
  - converts `X_tabular: (batch, m)` to `X_tokens: (batch, m, d_model)`
  - adds feature-id embeddings
  - supports missing-value mask embedding if used

**Done when**
- Unit-level check: same input batch → same output shape
- Model forward can run on a batch without errors

### Step 3 — Teacher Transformer model

**Implement**
- `TransformerFraudClassifier`:
  - encoder stack (`layers=2`)
  - attention heads:
    - teacher: `nhead=6` (paper)
  - FFN dim: `1024`
  - dropout `0.2`
  - pooling + classifier head to logits `(batch, 2)`

**Done when**
- Forward pass produces logits
- Parameter count is larger than student’s (sanity check)

### Step 4 — Teacher training pipeline (Algorithm 1)

**Implement**
- Training loop with:
  - optimizer: Adam, lr=0.001
  - iterations/epochs: align to paper’s `T=100` (map to epochs)
  - batch size: 32
- Validation loop
- Checkpoint saving: best model by F1/AUC

**Done when**
- You can train a teacher end-to-end and save checkpoint(s)
- Metrics computed and logged

### Step 5 — Student model + KD loss (Algorithm 2)

**Implement**
- Student model config:
  - `nhead=2`
  - `layers=2`
  - `dim_ff=1024`
  - `dropout=0.2`
- Distillation:
  - compute teacher logits (teacher frozen)
  - compute student logits
  - KD loss: KL(soft_student_T || soft_teacher_T) * T^2
  - CE loss: CE(student logits, hard labels) with T=1
  - combine: `alpha * CE + beta * KD`
- Per-domain student training:
  - train `student_manufacturing` on manufacturing subset
  - train `student_other` on other subset

**Done when**
- Both student models train and produce checkpoints
- Student inference time is measurably faster than teacher (optional benchmark)

### Step 6 — Distributed inference wrapper

**Implement**
- A small module that:
  - loads both student checkpoints
  - routes samples based on domain/industry to the right student
  - returns fraud probability

**Done when**
- Single inference entrypoint works for both domains

### Step 7 — Evaluation + plots (Section V)

**Implement**
- Metrics:
  - accuracy, precision, recall, F1
  - AUC (ROC-AUC)
  - MCC (optional but recommended for imbalance)
- Plots:
  - ROC curves for:
    - student vs baselines
  - PR curves for:
    - student vs baselines
- Baseline models (scikit-learn):
  - logistic regression
  - linear SVM
  - decision tree
  - random forest
  - XGBoost (if dependency allowed; else document fallback)
  - AdaBoost
- Ensure identical preprocessing for baselines and deep model

**Done when**
- A single script runs evaluation and outputs:
  - metrics JSON/CSV
  - ROC/PR plots to `artifacts/`

### Step 8 — Paper-style reporting + sanity checks

**Implement**
- Compare:
  - teacher vs student metrics
  - student vs baselines metrics
- Optional:
  - inference-time benchmark (CPU/GPU) similar to Table 4

**Done when**
- A `report.json` (or markdown report) summarises all key results and where plots live

---

## 5) Configs we will standardize (so runs are reproducible)

We will create configs for:

### 5.1 Data config
- dataset path
- label column
- industry column
- domain mapping rule (manufacturing vs other)
- feature columns list
- preprocessing parameters (imputation, scaling)
- split ratios (70/30)
- random seed

### 5.2 Teacher config (paper defaults)
- `d_model` (chosen based on feature count and capacity; configurable)
- `nhead=6`
- `layers=2`
- `dim_feedforward=1024`
- `dropout=0.2`
- `lr=0.001`
- `batch_size=32`
- `epochs=100` (or equivalent)

### 5.3 Student config (paper defaults)
- `nhead=2`
- `layers=2`
- `dim_feedforward=1024`
- `dropout=0.2`
- `lr=0.001`
- `batch_size=32`
- `epochs=100`
- distillation:
  - `temperature=7`
  - `alpha`
  - `beta`

---

## 6) Engineering decisions we must document (non-optional)

These are not fully specified in the paper, but we must implement them clearly and consistently:

- **Tabular-to-token embedding** strategy (value embedding + feature-id embedding)
- **Pooling method** (mean pooling vs CLS token)
- **Exact KL divergence direction and reduction** (batchmean vs mean)
- **Alpha/Beta values** for KD loss weighting
- **Handling of class imbalance**:
  - optional class weights in CE loss
  - and/or sampling strategy
- **Industry/domain mapping** (manufacturing vs other) based on dataset labels

All of these must be captured in configs and described in `docs/`.

---

## 7) Artifact conventions (modeled after “runs/artifacts” discipline)

All experiments must write to an artifacts directory (gitignored), with a consistent structure:

- `artifacts/`
  - `runs/<run_id>/`
    - `config_snapshot.yaml` (or json)
    - `metrics.json` (train/val/test)
    - `plots/` (roc/pr curves)
    - `checkpoints/` (teacher/student)

Every run must record:

- git commit hash (if available)
- seed
- dataset fingerprint (row count, feature count, label ratio, domain split counts)

---

## 8) Trackers we will maintain in `docs/` (like SSM/Jyotish)

We will add these files as the codebase grows:

- `docs/RUNBOOK.md`: exact commands for common tasks (train/eval/plots)
- `docs/EXPERIMENTS.md`: run IDs + summary metrics for key runs (paper-like tables)
- `docs/DECISIONS.md`: underspecified paper choices (tokenization/pooling/alpha-beta/etc.)
- `docs/BUGS.md`: known issues + fixes

These trackers are not “nice-to-have”: they prevent agent drift.

---

## 9) Verification commands (must be kept up-to-date)

Once dependencies and scripts exist, maintain a canonical verification set here.

Initial target (once implemented):

```bash
# 1) Environment (example)
python -V

# 2) Tests
pytest -q

# 3) Smoke: show CLIs exist
python -m scripts.train_teacher --help
python -m scripts.train_student_kd --help
python -m scripts.evaluate --help

# 4) Smoke: data pipeline (requires data.path configured)
python -m scripts.data_smoke --help
```

As soon as lint/format/typecheck are added, they must be listed here.

---

## 10) Acceptance criteria (what “finished” means)

This project is complete when:

- **Teacher training** works end-to-end and produces a checkpoint and metrics.
- **KD student training** works end-to-end and produces:
  - a manufacturing student checkpoint
  - an other-industries student checkpoint
- **Evaluation** produces:
  - accuracy/precision/recall/F1/AUC (and MCC optional)
  - ROC and PR plots
  - comparison vs sklearn baselines
- **Distributed inference wrapper** can load both students and run prediction.
- The repo contains clear run instructions and configs so another person can reproduce results.

