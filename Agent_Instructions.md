# Agent Instructions — Financial Fraud Detection KD Transformer

> **Read this file first. Read it completely. Do not skip sections.**
>
> This is the single entry point for any AI coding agent working on this repository.
> It explains the project goal, the current status, the non‑negotiable rules, and the exact workflow:
> **checkout → verify → fix → implement next**.
>
> If something in this file conflicts with other docs, **this file wins for process**.
> For technical requirements, **`docs/PROJECT_PLAN.md` wins**.

---

## 0) One-sentence mission

> **Implement the paper’s Transformer-based financial fraud detector and the distributed (domain-split) multi-teacher → student knowledge distillation training + evaluation pipeline, reproducibly, in this repo.**

Source paper (already present locally but must NOT be committed to git):

- `A_Distributed_Knowledge_Distillation_Framework_for_Financial_Fraud_Detection_Based_on_Transformer (1) (1).pdf`

---

## 1) Project identity

- **Repo**: `shreyasvp26/financial-fraud-detection-kd`
- **Local workspace**: `/Users/shreyasvp/Desktop/Projects/Financial Fraud Detection`
- **Primary spec**: `docs/PROJECT_PLAN.md`

What we are building (minimum):

- **Teacher model**: Transformer encoder fraud classifier for tabular financial indicators
- **Student models**: smaller Transformers trained with KD
- **Distributed/domain split**: manufacturing vs other industries (paper’s setup)
- **Evaluation**: metrics + ROC/PR curves + baseline ML models
- **Reproducible scripts/configs**: deterministic runs, saved artifacts, checkpoints

---

## 2) Current status (authoritative as of this checkout)

### 2.1 What is done

- GitHub repo created and connected to this folder (`origin` set)
- Initial commit pushed with:
  - `README.md`
  - `.gitignore` (includes `*.pdf`)
- Documentation scaffold created:
  - `docs/PROJECT_PLAN.md` (full step-by-step implementation plan)
  - `docs/README.md`

### 2.2 What is NOT done yet

- No `src/` code exists yet
- No dataset loader / preprocessing exists yet
- No training scripts exist yet
- No tests exist yet
- No dependency management (`requirements.txt` / `pyproject.toml`) exists yet

### 2.3 “Next work” (the immediate frontier)

Start implementing **Step 0 and Step 1** from `docs/PROJECT_PLAN.md`:

- repository scaffolding (folders, configs, dependency file)
- dataset ingestion + preprocessing + domain split

---

## 3) The non-negotiable rules (process + safety)

1) **Checkout first, then audit.**
   - Always start by pulling latest and inspecting repo state (commands in §4).

2) **Verify before building forward.**
   - If any tests/checks exist, run them first. If they fail, fix them before adding features.

3) **Do not commit PDFs or datasets.**
   - The paper PDF is local reference only. Datasets and large artifacts must be gitignored.

4) **No guessing file contents.**
   - Read files before editing. Use repo search to confirm symbols/paths.

5) **Reproducibility is a feature.**
   - Every script must accept config/CLI args, save outputs to an artifacts directory, and set seeds.

6) **Stay faithful to the paper, but document unavoidable choices.**
   - Where the paper is underspecified (e.g., tabular tokenization, \(\alpha/\beta\) for KD), implement a sensible default and record it in config + docs.

7) **Small, reviewable increments.**
   - Prefer small commits/PRs that leave the repo runnable. Don’t create huge untestable drops.

---

## 4) Start-of-session checklist (do this every time)

Run these commands in the repo root:

```bash
git status -sb
git log --oneline -20
git remote -v
```

Then read:

- `README.md`
- `docs/PROJECT_PLAN.md`

If there is code already:

- run the project’s verification commands (tests/lint/typecheck) exactly as documented in `docs/` or the repo root.

---

## 5) Universal workflow loop (for any task)

Follow this loop strictly:

1) **READ**: read relevant files + `docs/PROJECT_PLAN.md` section.
2) **PLAN**: write a short plan of intended changes and outputs.
3) **IMPLEMENT**: smallest change that keeps repo runnable.
4) **VERIFY**: run checks/tests (or at minimum, run the script you changed).
5) **DOC**: update `docs/` if you introduced a new decision, interface, or run command.
6) **REPORT**: tell the user what changed + how to run it.

If verification fails, go back to step 3. Do not “move on” with failing checks.

---

## 6) What “verification” means in this repo (as it evolves)

Right now, there is no code. As soon as the implementation begins, the repo must gain a standard verification suite.

When those exist, agents must run (and keep green) the equivalents of:

- unit tests (e.g., `pytest -q` or `python -m pytest`)
- formatting/lint/type checks (if configured)
- a small smoke run (tiny dataset subset) to ensure training/eval scripts start

Until tests exist:

- every new script must include a `--help` and a minimal dry-run/smoke mode

---

## 7) Implementation priorities (in order)

1) **Data correctness** (splits, preprocessing, consistent feature ordering)
2) **Correct loss implementation** (KD temperature, KL scaling, CE)
3) **Reproducibility** (configs, seeds, deterministic runs)
4) **Evaluation parity** (metrics computed correctly; baselines comparable)
5) **Performance** (only after correctness)

---

## 8) Quick map: “user request” → “first file to read”

| User says… | First file to read |
|------------|---------------------|
| “What do we need to build?” | `docs/PROJECT_PLAN.md` |
| “Start implementation” | `docs/PROJECT_PLAN.md` Step 0–1 |
| “Implement teacher model” | `docs/PROJECT_PLAN.md` §2.1 + Step 3–4 |
| “Implement distillation” | `docs/PROJECT_PLAN.md` §2.2 + Step 5 |
| “Run evaluation / plots” | `docs/PROJECT_PLAN.md` Step 7 |
| “Make it reproducible” | `docs/PROJECT_PLAN.md` Step 0 + configs section |

---

## 9) Handoff protocol (how an agent should report back)

When finishing a work session, report:

1) **What changed** (3–6 bullets)
2) **How to run/verify** (exact commands)
3) **Decisions made** (esp. anything underspecified by paper)
4) **What’s next** (the next Step number(s) from `docs/PROJECT_PLAN.md`)

---

**End of Agent Instructions.** Any agent starting work should re-run §4 every session and follow §5 for every task.

