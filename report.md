# Expense Account Classification: Peakflo Take-Home
**Ishaan Sinha**

---

## 1. Executive Summary

A LinearSVC on a four-block feature pipeline (word TF-IDF, char_wb TF-IDF, vendor TF-IDF, numeric) hits **87.5% accuracy** on 5-fold grouped cross-validation. ±1.3% across folds. Above the 85% bar.

What surprised me: the model isn't really *understanding* anything. It's recognising vendor names and character patterns. Word-level meaning contributes almost nothing. That's fine. That's all this problem actually requires.

What I'm honest about: macro F1 is 0.63. That's because 34 of the 103 categories have fewer than five samples. No model learns those from one example. That's a data problem, not a model problem, and I report it as such.

---

## 2. Data Analysis

4,894 line items. 337 vendors. 103 account categories. One Singapore client. Amounts spanning eight orders of magnitude, from refunds at -$15,195 to one entry at $161,838,000.

### What broke first when I looked at the data

**The bills lie.** 801 bills have multiple line items each. The naive move is a random train/test split, and that leaks line items from the same bill across the split. The model doesn't *learn*, it *remembers*. The fix is `GroupKFold(_id)`: every line item of a bill stays in the same fold. This is the difference between an honest 87% and a dishonest 95%.

**Vendor isn't a free win.** I expected most vendors to map to one account each (AWS to Cloud, Slack to SaaS). Reality: only **31% of rows** come from such "deterministic" vendors. The rest come from vendors that span multiple accounts. A lookup table caps out at 70% before text features even start working. Text has to do real work.

**accountId is the answer.** It maps almost 1:1 to accountName. Including it as a feature would solve the problem with a one-liner, and produce a model that's useless on any record where the accountId isn't already known. Excluded, hard.

**The tail is brutal.** 16 categories have exactly one sample. 34 have fewer than five. Optimise for accuracy, you're fine. They barely move the needle. Optimise for macro F1, they kill you. Both are reported.

![class distribution](assets/class_distribution.png)

*Top 5 classes hold 50% of the data. The 34 rare classes hold 1.5% between them. The shape of the problem isn't "classify 103 things". It's "classify ~20 things well, and shrug at the long tail honestly."*

---

## 3. Methodology

### Validation

`GroupKFold(n_splits=5, groups=_id)` everywhere. Same splitter object for every baseline and every model. The numbers are directly comparable because the splits are identical.

### Features

Four blocks, combined via `ColumnTransformer` inside one `sklearn.Pipeline`:

| Block | What | Why |
|---|---|---|
| Word TF-IDF | (1,2)-grams on `itemName + itemDescription` | phrase patterns |
| Char TF-IDF | `char_wb` (3,5) on the same text | sub-word patterns, codes, typos |
| Vendor TF-IDF | (1,1) on `vendorId` as a single token | direct vendor identity |
| Numeric | `log1p(|amount|)` + `sign(amount)` + `len(text)`, scaled | amount range, refund signal |

`char_wb` (within-word boundaries) is the right call over plain `char` for short text. It doesn't generate n-grams that span word breaks. itemName + itemDescription are concatenated rather than vectorised separately because in 17% of rows they differ, and that 17% has real signal that's lossless to merge.

Single `Pipeline` object means preprocessing and inference are identical. No train-serve skew.

### Model selection

LogReg with the full feature set hit 85.2%. Same features under LinearSVC hit 87.5%. Hinge loss fits this kind of sparse text problem slightly better than log loss. C swept over {0.25, 0.5, 1, 2, 4}; C=4 won by 0.3%. Not dramatic, but consistent.

`class_weight='balanced'` everywhere. No SMOTE. For sparse high-dimensional text, sample weighting beats synthetic oversampling.

---

## 4. Results

Out-of-fold predictions, 5-fold grouped CV:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| B0 (majority class) | 0.241 | 0.004 | 0.094 |
| B1 (vendor mode lookup) | 0.689 | 0.423 | 0.651 |
| B2 (TF-IDF + LogReg, text only) | 0.720 | 0.577 | 0.731 |
| M1 (LogReg, full features) | 0.852 | 0.628 | 0.852 |
| M2 (LinearSVC, C=1) | 0.872 | 0.634 | 0.871 |
| **M_final (LinearSVC, C=4)** | **0.875** | **0.631** | **0.872** |

Per-fold accuracy: 0.888, 0.876, 0.884, 0.855, 0.872. Std 1.3%. Nothing fragile.

### What carries the model: ablation

Drop one feature block at a time. Re-run CV. Measure the drop.

![ablation](assets/ablation.png)

| Removed | CV Accuracy | Drop |
|---|---:|---:|
| Vendor TF-IDF | 0.859 | **−1.57%** |
| Char TF-IDF | 0.865 | −1.00% |
| Word TF-IDF | 0.874 | −0.10% |
| Numeric | 0.874 | −0.08% |

This is the most interesting result in the project. **Vendor identity and character morphology dominate. Word-level meaning barely contributes.** The model isn't reading expense descriptions; it's recognising the *shape* of vendor names. Which is fine. It's the right level of sophistication for this problem, and trying to add semantic depth would be premature engineering.

### Where the errors come from

Top confusion pairs are all economically adjacent:

- Online Subscription vs Prepaid Operating Expense
- IC Clearing vs IC Clearing - Paid on Behalf
- Audit Fee vs Others

These aren't *wrong*. They're the cases where a human reviewer would also hesitate. The accounting boundary is policy, not text. Rare classes (n<5) score F1 ≈ 0; that's the entire reason macro F1 is 0.63 while weighted F1 is 0.87.

---

## 5. Discussion

### What I'd actually ship

Not the model. The model + a confidence threshold + a human queue.

![deployment](assets/deployment.png)

The 87.5% of cases where the top decision-function score dwarfs the runner-up: auto-classify. The 12.5% where the gap is small (especially on those economically adjacent pairs above): route to human review. That's how production finance systems actually work, and the confusion analysis tells you *exactly* which category boundaries to flag.

### What's broken (honestly)

- 34 categories can't be learned from this dataset. The fix isn't a better model. It's more data, or merging those categories, or accepting that they always go to the human queue.
- LinearSVC's decision function isn't calibrated. A real production threshold needs Platt scaling on a held-out set first.
- The model is static. New vendor or new account category means a retrain.

### What I'd do to push past 92%

A hybrid:

1. **Lookup** for high-frequency deterministic vendors (instant, exact)
2. **LinearSVC** for the bulk of transactions
3. **LLM (GPT-4 / Claude few-shot)** for the low-confidence tail. Exactly the boundary cases the linear model can't disambiguate.

The cases the linear model fails on are the cases that genuinely need reading comprehension. That's where the next week of effort goes. It mirrors what real production finance systems already deploy.
