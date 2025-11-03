# Diabetic Nephropathy – South Korea Dataset

This repository provides data pipelines and models to predict renal outcomes in South Korean patients diagnosed with **Diabetic Nephropathy (DN)**. It includes preprocessing, covariate regression, model training, evaluation, and results aggregation using **glomerular**, **tubular**, and **proteomic** features.

---

## Prediction Tasks

We perform binary classification for three key outcomes:

- **End-Stage Kidney Disease (ESKD)**
- **2-Year Composite Outcome**
- **3-Year Composite Outcome**

## ⚙️ Modeling Pipeline

- Feature normalization via `StandardScaler`
- **Model types**:
  - `XGBoost` 
  - `SVM`
  - `RNN`
    
- **Evaluation**:
  - Stratified 10-fold cross-validation (fixed seed: 42)
  - Metrics: AUC-ROC, MCC, weighted Precision/Recall/F1, Specificity

---

## Results

### Notes
1. All metrics are averaged over 10 folds.
2. Precision, Recall, and F1 are **weighted** across classes.
3. Only 45 significant proteins used for **predictive modeling**.
4. **All proteins** used for correlation with glomeruli/tubules.
5. Covariate regression is only used for correlation.
6. Seed = `42` ensures identical folds across models.

---



## Run Commands

### Clone the Repository
```bash
git clone https://github.com/Srujith20/diabetic-nephropathy-dataset.git
cd diabetic-nephropathy-dataset


