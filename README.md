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

### Glomerular Features
- **Embedding Size:** `1 × 315` per glomerulus
- **Glomeruli per Patient:**
  - **Minimum:** 2
  - **Maximum:** 45
  - **Mean:** 15.70

### Tubular Features
- **Embedding Size:** `1 × 207` per tubule
- **Tubules per Patient:**
  - **Minimum:** 178
  - **Maximum:** 3393
  - **Mean:** 1039
  - 
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
- Covariate regression (optional, fold-specific linear models)
- **Model types**:
  - `XGBoost` (with and without PCA)
  - `SVM`
  - `RNN` (Keras-based, sequence-level glomerular modeling)
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
5. **No covariate regression** used for the reported results.
6. Seed = `42` ensures identical folds across models.

---

### ESKD Prediction

| Feature Set | Model                   | AUC    | MCC    | Precision | Recall | F1 Score | Specificity |
|-------------|------------------------|--------|--------|-----------|--------|----------|-------------|
| **Glomeruli** | XGBoost + Correlation   | 0.7876 | 0.3684 | 0.7605    | 0.7442 | 0.7507   | 0.7969      |
|             | XGBoost + PCA           | 0.6271 | 0.3146 | 0.7414    | 0.7093 | 0.7208   | 0.7500      |
|             | XGBoost                 | 0.7500 | 0.3080 | 0.7366    | 0.7326 | 0.7345   | 0.8125      |
|             | RNN                     | 0.6705 | 0.2116 | 0.7001    | 0.6860 | 0.6923   | 0.7656      |
|             | SVM                     | 0.6435 | 0.3491 | 0.7522    | 0.7558 | 0.7539   | 0.8438      |
| **Tubules**  | XGBoost + Correlation   | 0.7727 | 0.3683 | 0.7595    | 0.7558 | 0.7576   | 0.8281      |
|             | XGBoost + PCA           | 0.8168 | 0.4670 | 0.7974    | 0.7907 | 0.7936   | 0.8438      |
|             | XGBoost                 | 0.7109 | 0.3482 | 0.7521    | 0.7442 | 0.7477   | 0.8125      |
|             | RNN                     | 0.7038 | 0.2441 | 0.7195    | 0.6395 | 0.6614   | 0.6406      |
|             | SVM                     | 0.6328 | 0.2296 | 0.7144    | 0.6279 | 0.6509   | 0.6250      |
| **Proteins** | XGBoost                 | **0.8835** | **0.5796** | **0.8422** | **0.8256** | **0.8309** | **0.8438** |

---

### 2-Year Composite Outcome

| Feature Set | Model                   | AUC    | MCC    | Precision | Recall | F1 Score | Specificity |
|-------------|------------------------|--------|--------|-----------|--------|----------|-------------|
| **Glomeruli** | XGBoost + Correlation   | 0.8006 | 0.4727 | 0.7592    | 0.7442 | 0.7481   | 0.7455      |
|             | XGBoost + PCA           | 0.6264 | 0.2281 | 0.6452    | 0.6279 | 0.6336   | 0.6554      |
|             | XGBoost                 | 0.8141 | 0.4668 | 0.7542    | 0.7558 | 0.7549   | 0.8182      |
|             | RNN                     | 0.6721 | 0.2542 | 0.6563    | 0.6512 | 0.6534   | 0.7091      |
|             | SVM                     | 0.6862 | 0.3968 | 0.7251    | 0.7326 | 0.7235   | 0.8545      |
| **Tubules**  | XGBoost + Correlation   | 0.6845 | 0.2237 | 0.6421    | 0.6395 | 0.6407   | 0.7091      |
|             | XGBoost + PCA           | **0.8094** | **0.5744** | **0.8039** | **0.8023** | **0.8030** | **0.8364** |
|             | XGBoost                 | 0.6815 | 0.2957 | 0.6763    | 0.6628 | 0.6673   | 0.6909      |
|             | RNN                     | 0.6557 | 0.2542 | 0.6563    | 0.6512 | 0.6534   | 0.7091      |
|             | SVM                     | 0.6903 | 0.2975 | 0.6827    | 0.6395 | 0.6466   | 0.6000      |
| **Proteins** | XGBoost                 | 0.7677 | 0.3748 | 0.7136    | 0.6977 | 0.7023   | 0.7091      |

---

### 3-Year Composite Outcome

| Feature Set | Model                   | AUC    | MCC    | Precision | Recall | F1 Score | Specificity |
|-------------|------------------------|--------|--------|-----------|--------|----------|-------------|
| **Glomeruli** | XGBoost + Correlation   | 0.6927 | 0.2148 | 0.6093    | 0.6047 | 0.6040   | 0.5556      |
|             | XGBoost + PCA           | **0.7268** | **0.3767** | **0.6903** | **0.6860** | **0.6858** | 0.6444      |
|             | XGBoost                 | 0.6970 | 0.3235 | 0.6625    | 0.6628 | 0.6626   | 0.6889      |
|             | RNN                     | 0.6959 | 0.3083 | 0.6564    | 0.6512 | 0.6506   | 0.6000      |
|             | SVM                     | 0.5854 | 0.2056 | 0.6080    | 0.6047 | 0.5940   | **0.7556** |
| **Tubules**  | XGBoost + Correlation   | 0.6098 | 0.1461 | 0.5749    | 0.5698 | 0.5685   | 0.5111      |
|             | XGBoost + PCA           | 0.6900 | 0.2182 | 0.6118    | 0.6047 | 0.6029   | 0.5333      |
|             | XGBoost                 | 0.6119 | 0.2148 | 0.6093    | 0.6047 | 0.6040   | 0.5556      |
|             | RNN                     | 0.5940 | 0.0921 | 0.5470    | 0.5465 | 0.5467   | 0.5556      |
|             | SVM                     | 0.6317 | 0.1593 | 0.5807    | 0.5814 | 0.5807   | 0.6222      |
| **Proteins** | XGBoost                 | 0.7187 | 0.2652 | 0.6357    | 0.6279 | 0.6263   | 0.5556      |

---

## Run Commands

### Clone the Repository
```bash
git clone https://github.com/Srujith20/diabetic-nephropathy-dataset.git
cd diabetic-nephropathy-dataset


