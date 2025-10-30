# Diabetic Nephropathy – South Korea Dataset

This repository provides data pipelines and models to predict renal outcomes in South Korean patients diagnosed with **Diabetic Nephropathy (DN)**. It includes preprocessing, covariate regression, model training, evaluation, and results aggregation using **glomerular**, **tubular**, and **proteomic** features.

---

## Prediction Tasks

We perform binary classification for three key outcomes:

- **End-Stage Kidney Disease (ESKD)**
- **2-Year Composite Outcome**
- **3-Year Composite Outcome**

---

## Dataset Statistics

### Class Distribution Across Outcomes

The following figures show the class distribution for each binary outcome:

<div align="center">
  <img src="https://github.com/user-attachments/assets/f19b2cdb-42c9-46b1-89b0-1ac0019cc10a" alt="ESKD Class Balance" width="30%" height="200px" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/eb8e4a82-70aa-4a58-97d2-205beed5451a" alt="2-Years Class Balance" width="30%" height="200px" style="margin-right: 10px;" >
  <img src="https://github.com/user-attachments/assets/d2df57e3-69cb-43a7-95be-a9641ff16cc7" alt="3-Years Class Balance" width="30%" height="200px" style="margin-right: 10px;">
</div>

### Patient-Level Covariates
- **Binary:** `Sex`, `Hypertension`, `Ischemic_Heart_Disease`, `Stroke_History`
- **Continuous:** `Age`, `Weight`, `Height`, `Diabetic_Years`

---

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


