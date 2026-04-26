# Expense Account Classification — Peakflo Take-Home

Classifies expense bill line items into one of 103 accounting account names.

**Result:** **87.5% accuracy** (macro F1 0.63, weighted F1 0.87) on 5-fold `GroupKFold(groups=_id)` out-of-fold predictions — above the 85% bar set in the brief, with stable per-fold variance (±1.3%).

## Files

| File | What |
|---|---|
| `expense_classification.ipynb` | full deliverable: EDA, baselines, main model, error analysis, charts |
| `report.md` | written description (Executive Summary, Data Analysis, Methodology, Results, Discussion) |
| `pyproject.toml` + `uv.lock` | pinned dependencies for `uv` |
| `accounts-bills.json` | provided dataset |

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) (install once: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
```

This reads `pyproject.toml`, locks the dependency graph in `uv.lock`, and creates a `.venv/`.

## Run the notebook

Open interactively:

```bash
uv run jupyter lab expense_classification.ipynb
```

Or re-execute end-to-end (regenerates all outputs, takes ~10–15 min):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace expense_classification.ipynb --ExecutePreprocessor.timeout=1800
```

Random seeds are fixed (`SEED = 42`); results are deterministic.

## Results summary

5-fold `GroupKFold` (grouped by bill `_id` to prevent same-bill leakage between train and test):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| B0 — Majority class | 0.241 | 0.004 | 0.094 |
| B1 — Vendor mode lookup | 0.689 | 0.423 | 0.651 |
| B2 — TF-IDF + LogReg (text only) | 0.720 | 0.577 | 0.731 |
| M1 — LogReg (full features) | 0.852 | 0.628 | 0.852 |
| M2 — LinearSVC C=1 | 0.872 | 0.634 | 0.871 |
| **M_final — LinearSVC C=4** | **0.875** | **0.631** | **0.872** |

Final pipeline:

- TF-IDF word n-grams (1, 2) on `itemName + itemDescription`
- TF-IDF `char_wb` n-grams (3, 5) on the same text
- TF-IDF on `vendorId`
- `log1p(|amount|)` + `sign(amount)` + `text_len`, standardised
- `LinearSVC(C=4, class_weight="balanced")`

Full per-class breakdown, top confusion pairs, feature ablation, and visualisations are in §6 of the notebook. Methodology and discussion are in `report.md`.
