# Expense Account Classification — Peakflo Take-Home
**Ishaan Sinha**

---

## 1. Executive Summary

A LinearSVC classifier with a four-block feature pipeline (word TF-IDF, 
character n-gram TF-IDF, vendor identity, and numeric features) achieves 
**87.5% overall accuracy** (weighted F1: 0.872) on 5-fold grouped 
cross-validation — comfortably above the 85% threshold and stable across 
folds (std: ±1.3%). The dominant signal is vendor identity and sub-word 
character patterns rather than word-level semantics, a finding confirmed 
by ablation. Macro F1 of 0.63 reflects expected drag from 34 categories 
with fewer than 5 samples, which the model cannot reliably learn; this is 
reported honestly rather than papered over.

---

## 2. Data Analysis

### 2.1 Dataset Overview

The dataset contains 4,894 expense line items from a Singapore-based 
client, spanning 337 unique vendors and 103 unique account names across 
97 account IDs. Transaction amounts range from -$15,195 to $161,838,000 
SGD — an eight-order-of-magnitude spread that required log-transformation 
before use as a feature.

### 2.2 Key Findings

**Finding 1 — Multi-line-item bills require grouped validation.**
801 unique bill IDs contain more than one line item. Standard random 
train/test splitting would leak information across splits (line items 
from the same bill appearing in both train and test). GroupKFold(5) 
grouped by bill ID was used throughout to prevent this.

**Finding 2 — itemDescription adds marginal but real signal.**
itemDescription is identical to itemName in 83.3% of rows. Rather than 
dropping it, both fields were concatenated — the 16.7% of rows where 
they differ contain genuine additional signal, and concatenation is 
lossless.

**Finding 3 — Vendor determinism is lower than expected.**
Only 31% of rows belong to vendors that always map to a single account 
category. A pure lookup table approach would therefore cover less than 
a third of transactions, making text features the primary load-bearing 
signal.

**Finding 4 — Severe class imbalance in the tail.**
The top category (Online Subscription/Tool) contains 1,179 records. 
16 categories have exactly 1 sample each; 34 have fewer than 5. 
No classes were dropped — instead, class_weight='balanced' was used 
throughout to ensure the model does not simply ignore the tail.

**Finding 5 — accountId is a data leak and was excluded.**
accountId maps almost perfectly 1:1 to accountName. Including it as 
a feature would trivially solve the task but produce a model useless 
on any record where accountId is unknown (i.e., exactly the records 
we need to classify). It was hard-excluded from all models.

### 2.3 Data Quality

No missing values were found in the core fields (vendorId, itemName, 
accountName). itemTotalAmount contained negative values (credit notes / 
refunds) — the sign was preserved as a binary feature rather than 
treated as noise.

---

## 3. Methodology

### 3.1 Validation Strategy

GroupKFold(n_splits=5, groups=bill_id) was used for all models. This 
ensures no bill's line items appear in both train and test, giving 
conservative and honest performance estimates. All baseline and main 
model numbers are directly comparable because the same splitter object 
was reused throughout.

### 3.2 Baseline Progression

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| B0 — majority class | 0.241 | 0.004 | 0.094 |
| B1 — vendor mode lookup | 0.689 | 0.423 | 0.651 |
| B2 — TF-IDF + LogReg (text only) | 0.720 | 0.577 | 0.731 |
| M1 — LogReg (full features) | 0.852 | 0.628 | 0.852 |
| M2 — LinearSVC C=1 (full features) | 0.872 | 0.634 | 0.871 |
| **M_final — LinearSVC C=4** | **0.875** | **0.631** | **0.872** |

B1 at 68.9% establishes that vendor identity alone is a strong prior. 
The jump from B2 (72.0%) to M1 (85.2%) — a 13-point lift — came 
entirely from adding character n-gram TF-IDF, vendor TF-IDF, and 
log-amount features, not from changing the model family.

### 3.3 Feature Engineering

Four feature blocks were combined via ColumnTransformer inside a 
single sklearn Pipeline:

| Block | Detail | Rationale |
|---|---|---|
| Word TF-IDF | (1,2)-grams, sublinear_tf, min_df=2 on itemName+itemDescription | Phrase-level keywords |
| Char TF-IDF | char_wb (3,5)-grams, min_df=2 on same text | Sub-word patterns, typos, codes |
| Vendor TF-IDF | (1,1)-gram on vendorId as single token | Direct vendor identity signal |
| Numeric | log1p(|amount|), sign(amount), text_length — StandardScaled | Amount range and refund signal |

char_wb was chosen over plain char because it respects word boundaries, 
producing cleaner n-grams for short text like vendor and item names.

### 3.4 Model Selection

LinearSVC was chosen over LogisticRegression for its slight accuracy 
edge on short-text TF-IDF (confirmed by M1 vs M2 head-to-head: +2.0 
points). Regularisation strength was swept over C ∈ {0.25, 0.5, 1.0, 
2.0, 4.0} on the winning model; C=4 gave the best mean CV accuracy. 
max_iter was set to 10,000 for the final saved artifact to ensure full 
convergence.

class_weight='balanced' was applied throughout. Resampling was 
deliberately avoided — for sparse high-dimensional text features, 
sample weighting is preferable to synthetic oversampling.

---

## 4. Results

### 4.1 Final Performance

M_final (LinearSVC, C=4) on 5-fold GroupKFold OOF:

| Metric | Score |
|---|---:|
| Accuracy | 0.8749 |
| Macro F1 | 0.6307 |
| Weighted F1 | 0.8721 |

Per-fold accuracy: 0.888, 0.876, 0.884, 0.855, 0.872 
Mean ± std: **0.8749 ± 0.0127**

The low standard deviation (1.3%) confirms the result is not a lucky 
split — performance is consistent across all five held-out groups.

### 4.2 Feature Importance (Ablation)

| Block Removed | CV Accuracy | Drop vs Full |
|---|---:|---:|
| None (full model) | 0.8749 | — |
| Vendor TF-IDF | 0.8592 | −1.57% |
| Char TF-IDF | 0.8649 | −1.00% |
| Word TF-IDF | 0.8739 | −0.10% |
| Numeric | 0.8741 | −0.08% |

Contrary to intuition, word-level semantics contributed minimally 
(−0.1%). The dominant signal is vendor identity (−1.6%) and sub-word 
character patterns (−1.0%). This suggests most classifications are 
determined by recognising the vendor and its name morphology, not by 
understanding the semantic meaning of expense descriptions.

### 4.3 Error Analysis

**Top confusion pairs (economically plausible):**
- Online Subscription/Tool ↔ Prepaid Operating Expense — both are 
  forward-paid recurring costs; the boundary is an accounting policy 
  decision, not a textual one
- IC Clearing ↔ IC Clearing - Paid on Behalf — near-identical names, 
  distinguished only by intercompany relationship context absent from 
  the text
- Audit Fee ↔ Others — "Others" is a catch-all; ambiguous items near 
  its boundary are inherently hard to classify

All top confusion pairs are economically adjacent categories. 
This means model errors are not random — they are concentrated on 
cases where even a human reviewer would hesitate.

**Rare class performance:**
34 categories have fewer than 5 training samples. Their per-class F1 
scores are near zero. This is the primary driver of macro F1 (0.63) 
being lower than weighted F1 (0.87). With weighted F1, large classes 
dominate; with macro F1, every class counts equally regardless of 
support. Both are reported for transparency.

---

## 5. Discussion

### 5.1 Strengths

- Exceeds the 85% bar with a stable, well-validated result (GroupKFold 
  prevents bill-level leakage that would inflate most naive estimates)
- The full pipeline is a single sklearn Pipeline object — preprocessing 
  and inference are identical, eliminating train/serve skew
- Ablation provides principled justification for every feature 
  engineering decision
- Error analysis identifies the human-review queue candidates (the 
  confusion pairs above) rather than just reporting aggregate accuracy

### 5.2 Limitations

- 34 rare categories (< 5 samples) are effectively unlearnable from 
  this dataset size — the model will route unseen vendors in these 
  categories incorrectly
- The model is static — it has no mechanism to update as new vendors 
  or account categories are added without retraining
- Confidence scores use LinearSVC's raw decision function values, which 
  are not calibrated probabilities; a downstream threshold policy should 
  account for this

### 5.3 What I Would Do With More Time

**To push past 92%:** An LLM-based fallback (GPT-4o or Claude few-shot) 
for rare categories and ambiguous cases — specifically the confusion 
pairs identified above. The hybrid architecture would be: deterministic 
lookup for high-frequency vendors → LinearSVC for the bulk of 
transactions → LLM for low-confidence predictions (decision score below 
a threshold). This mirrors what production finance automation systems 
actually deploy.

**To make it production-ready:**
- Quarterly retraining pipeline as new vendor/category data accumulates
- Confidence threshold to route low-confidence predictions to a human 
  review queue (the confusion pairs above are the natural candidates)
- Calibrated probabilities via Platt scaling on a held-out set for 
  reliable confidence estimates
- Monitoring for distribution shift as the client adds new vendors

### 5.4 Business Considerations

The model's errors are not random — they cluster on economically 
adjacent categories where the correct classification is often an 
accounting policy decision rather than a textual one. Deploying with 
a human-in-the-loop queue for low-confidence predictions would capture 
the 87.5% of straightforward cases automatically while routing the 
genuinely ambiguous ones to a reviewer. This is more useful than a 
model that silently makes wrong predictions on hard cases.
