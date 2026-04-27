# Expense Account Classification: Peakflo Take-Home

> Classifies expense bill line items into one of **103** accounting categories.
> **87.5% accuracy** on 5-fold `GroupKFold(_id)` out-of-fold predictions, ±1.3% per-fold std. Above the 85% bar set in the brief.

<img width="930" height="857" alt="image" src="https://github.com/user-attachments/assets/c595b149-10aa-4021-8051-37929e9c8f05" />


---

## Quick start

```bash
uv sync
uv run jupyter lab expense_classification.ipynb
```

That's it. The notebook is self-contained. Random seed is fixed (`SEED = 42`); results are deterministic.

---

## What's in here

| File | What |
|---|---|
| `expense_classification.ipynb` | the deliverable: EDA, baselines, main model, error analysis, charts |
| `report.md` | written description: Executive Summary, Data Analysis, Methodology, Results, Discussion |
| `pyproject.toml` + `uv.lock` | uv-pinned dependencies |
| `accounts-bills.json` | dataset (provided), gitignored, not redistributed |

---

## Method, in one picture

The single most important methodology choice is **bill-level grouping in cross-validation**. 801 bills in this dataset have multiple line items; a random train/test split lets line items from the same bill leak across, inflating accuracy.

<img width="1099" height="407" alt="image" src="https://github.com/user-attachments/assets/409dea0c-b2aa-4ecb-9839-5330743711f2" />

`GroupKFold(n_splits=5, groups=_id)` keeps every line item of a bill in the same fold. Every number reported below is honest under this split.

---

## Results

5-fold `GroupKFold` out-of-fold predictions:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| B0 (majority class) | 0.241 | 0.004 | 0.094 |
| B1 (vendor mode lookup) | 0.689 | 0.423 | 0.651 |
| B2 (TF-IDF + LogReg, text only) | 0.720 | 0.577 | 0.731 |
| M1 (LogReg, full features) | 0.852 | 0.628 | 0.852 |
| M2 (LinearSVC, C=1) | 0.872 | 0.634 | 0.871 |
| **M_final (LinearSVC, C=4)** | **0.875** | **0.631** | **0.872** |

Per-fold accuracy on M_final: 0.888, 0.876, 0.884, 0.855, 0.872. Mean ± std = **0.875 ± 0.013**.

Macro F1 (0.63) is dragged down by 34 categories with fewer than 5 samples. Covered in §6 of the notebook.

---

## Reproduce end-to-end

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace \
  expense_classification.ipynb --ExecutePreprocessor.timeout=1800
```

Re-runs every cell from EDA through error-analysis charts, baking outputs back into the notebook. Takes ~10 to 15 min.

---

## Where to look

- **Methodology + ablation rationale** in `report.md`
- **Per-class scores, top confusion pairs, feature ablation, and visualisations** in §6 of the notebook
- **Decision history (which choice, why)** in §3 of the notebook (decision-matrix table)
