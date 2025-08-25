# ADE Classifier (PubMedBERT)

Binary text classifier for Adverse Drug Events (ADE) in short clinical/biomedical sentences.  
Selection policy is recall-first: choose a validation threshold to reach recall ≥ 0.90, then score the test set at that fixed threshold.  
**Final model:** PubMedBERT. For the full narrative report (plots + commentary), see **[00_main.ipynb][nb00]**.

---

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Reproduce](#reproduce)
- [Results (1-page)](#results-1-page)
- [Key Figures](#key-figures)
- [Deployment](#deployment)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview
- **Task:** ADE vs Not-ADE sentence classification.
- **Goal:** prioritize recall while keeping precision competitive for review.
- **Splits:** train / validation / test with stable IDs (seed 42).

---

## Data
- **Source:** Hugging Face `SetFit/ade_corpus_v2_classification` (sentence view of ADE Corpus v2).  
- **Cleaning & leakage guard (Notebook 1):**
  - Dropped empties and de-duped by `text_clean` (train 1,718; test 253).
  - Removed train↔test overlaps by `text_norm` (649 rows).
  - Stratified split: train 13,743; val 1,527; test 5,626.
  - Class weights from training labels: `{0: 0.372, 1: 1.628}`.

---

## Project Structure
Final_project/
├─ 00_main.ipynb # Narrative report notebook (mirrors this README)
├─ 01_Data_Prep.ipynb # Cleaning, leakage guard, stratified split
├─ 02_eda.ipynb # Exploratory data analysis
├─ 03_Baselines_TFIDF_LR_SVM.ipynb
├─ 04_transformer_distilbert.ipynb
├─ 05_transformer_biobert.ipynb
├─ 06_pubmedbert_training.ipynb
├─ 07_model_comparison.ipynb # Aggregates metrics and picks the winner
├─ src/
│ ├─ utilities.py
│ ├─ hf_utils.py
│ └─ plots.py
├─ artifacts/
│ ├─ baselines/
│ │ └─ tfidf/ # val/test probability CSVs for LR and SVM
│ └─ transformers/
│ ├─ distilbert/
│ ├─ biobert/
│ └─ pubmedbert/ # saved model, preds, thresholds, metrics
│ └─ comparison/
│ └─ all_models_summary.csv # used in Notebook 7 and 00_main
└─ plots/
├─ eda_*.png
├─ comparison/
│ ├─ cmp_precision_recall_bars.png
│ └─ cmp_error_composition_val_test.png
├─ pubmed_val_test_pr_roc_grid.png
├─ pubmed_test_confusion.png
└─ pubmed_test_calib_and_hist.png



---

## Getting Started
**Fastest:** open **[00_main.ipynb][nb00]** in Colab.  
**Local:** `pip install -r requirements.txt` (Python 3.10+), then run notebooks in order.

---

## Reproduce
Run these (in order) or open in Colab:
1. **Data prep:** [01_Data_Prep][nb01]  
2. **EDA:** [02_eda][nb02]  
3. **Models:** [03 Baselines][nb03], [04 DistilBERT][nb04], [05 BioBERT][nb05], [06 PubMedBERT][nb06]  
4. **Compare & select:** [07_model_comparison][nb07]  
5. Return to **[00_main][nb00]** to render the final report.

Artifacts produced include: `preds/val_probs.csv`, `preds/test_probs.csv`, `thresholds.json`, per-model `metrics_*.json`, and `artifacts/comparison/all_models_summary.csv`.

---

## Results (1-page)
Validation-chosen thresholds applied on test:

- **BioBERT:** Precision 0.896, Recall 0.900, F1 0.898, PR-AUC 0.946, ROC-AUC 0.978  
- **PubMedBERT:** Precision 0.888, Recall 0.908, F1 0.898, PR-AUC 0.948, ROC-AUC 0.983

**Error trade-off:** BioBERT FP 149 / FN 143 vs PubMedBERT FP 164 / FN 132.  
Under a recall-first policy, reducing FN is preferred ⇒ **choose PubMedBERT** (11 fewer FN for 15 more FP; F1 ties).

Full table: `artifacts/comparison/all_models_summary.csv`.

---

## Key Figures
- `plots/comparison/cmp_precision_recall_bars.png`  
- `plots/comparison/cmp_error_composition_val_test.png`  
- `plots/pubmed_val_test_pr_roc_grid.png`  
- `plots/pubmed_test_confusion.png`  
- `plots/pubmed_test_calib_and_hist.png`

(See **[00_main.ipynb][nb00]** for inline commentary.)

---

## Deployment

**Live demo:** https://huggingface.co/spaces/Daregay/ade_pubmedbert_demo
**Small further works still needed for demo**
---

## Acknowledgments
- Dataset: ADE Corpus v2 sentence classification on Hugging Face.
- Models: PubMedBERT, BioBERT, DistilBERT via Hugging Face Transformers.

---

## Notebooks
- **00 — Main report** — [Colab](https://colab.research.google.com/drive/1VdvHcuBEOoNfz0m4SP5nCG6Mx95_i3kW?usp=sharing)
- **01 — Data prep and split** — [Colab](https://colab.research.google.com/drive/1sLuzWFUAwlSKXFqqI851ZwTSgdnUzPzp)
- **02 — EDA** — [Colab](https://colab.research.google.com/drive/1ixGcdCv_moL-CagZ51qLHigC-btEgLCk)
- **03 — Baselines (TF-IDF LR/SVM)** — [Colab](https://colab.research.google.com/drive/1swGOnQTKFJ7AdwWeURLvads2kpBqzPDn)
- **04 — DistilBERT** — [Colab](https://colab.research.google.com/drive/1UgOQA5L0S4m-OsTCVY9f7aONCPcykmDV)
- **05 — BioBERT** — [Colab](https://colab.research.google.com/drive/1LQNZSgwLzN2GjOGNEZjUZJfOsTvFC6yl)
- **06 — PubMedBERT** — [Colab](https://colab.research.google.com/drive/1C3idcaweut3V4hxMf5zcCIPhbJTmDJVb)
- **07 — Model comparison** — [Colab](https://colab.research.google.com/drive/13uH5BAeFE8lAn_SQUXn-fzp_viRNqXMh)

