# Day 32 | PM Session – Decision Trees & Random Forest: Applied

**Week 6 – Machine Learning & AI**

This repository contains the take-home assignment solution for Day 32 PM Session covering Decision Trees, Random Forest, hyperparameter tuning, cost-sensitive evaluation, and model interpretability for an insurance fraud detection use case.

---

## Assignment Overview

| Detail | Info |
|---|---|
| Topic | DT vs RF comparison, hyperparameter tuning, feature importance, cost analysis |
| Estimated Time | 60–90 minutes |
| Submission | GitHub + Jupyter Notebook link in Slack #daily-standup |
| Due | Next day 09:15 AM |

---

## Repository Structure

```
D32_PM_DT_RF/
│
├── D32_PM_DT_RF_CaseStudy.ipynb     # Main Jupyter notebook (all code + outputs)
├── D32_PM_DT_RF_Solution.docx       # Part-wise written solution document
├── README.md                        # This file
```

---

## What's Covered

### Part A – Concept Application (40%)
- Synthetic insurance claims dataset (3000 records, 8 features)
- Decision Tree (max_depth=5) with top 3 fraud indicator rules extracted
- Random Forest tuned with `RandomizedSearchCV` optimising for **Recall**
- Full metrics comparison table (Accuracy, Precision, Recall, F1, AUC)
- Cost-sensitive evaluation: FN cost = 10 × FP cost
- 2-paragraph deployment recommendation

### Part B – Stretch: Gradient Boosting Preview (30%)
- Written answer: how Boosting differs from Bagging
- Reference resource linked (StatQuest YouTube series)

### Part C – Interview Ready (20%)
- **Q1 (Conceptual):** 1000-tree vs 100-tree RF tradeoffs
- **Q2 (Coding):** `compare_models(X, y, models_dict)` — 5-fold CV, returns accuracy/F1/time DataFrame
- **Q3 (Debug):** Why feature importances differ between two RF runs — root cause + fix

### Part D – AI-Augmented Task (10%)
- OOB error explained to a non-technical manager using an analogy
- AI response evaluated for accuracy
- Follow-up question answered and critiqued
- OOB vs test error verified programmatically

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shubhamrangdal2001-hash/Assignment_Day32PM.git
cd D32_PM_DT_RF
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

> Python 3.8+ recommended.

### 3. Launch the notebook

```bash
jupyter notebook D32_PM_DT_RF_CaseStudy.ipynb
```

Then run all cells top to bottom: **Kernel → Restart & Run All**

---

## Key Results

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Decision Tree (max_depth=5) | 0.8383 | 0.7544 | 0.7544 | 0.8723 |
| Random Forest (Tuned) | 0.8850 | 0.8421 | 0.8327 | 0.9312 |

**Business cost (FN=10×FP):**
- Decision Tree: 615 units
- Random Forest: 322 units ✓ (~47% reduction)

**Deployment strategy:** RF for automated scoring + DT rules as the explanation layer for regulators.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

---

## Notes

- `random_state=42` is set throughout for full reproducibility
- All CV uses `StratifiedKFold` via `cross_val_score` to handle class imbalance
- OOB score verified against test error (difference < 0.005)
