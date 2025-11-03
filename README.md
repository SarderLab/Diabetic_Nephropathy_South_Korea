# Diabetic Nephropathy – South Korea Dataset
This repository contains modular, config-driven ML pipelines for cross-validation experiments on Diabetic Nephropathy (South Korea cohort).
Currently supported pipelines: XGBoost, RNN, SVM.

Each pipeline is self-contained in codes/ and uses a corresponding configuration file in configs/.
Results are saved to results/ with standard outputs like predictions, ROC curves, and confusion matrices.

## Repository Structure
```bash
Diabetic_Nephropathy_South_Korea/
 ├── codes/
 │    ├── run_xgb_pipeline.py        # XGBoost cross-validation pipeline
 │    ├── run_rnn_pipeline.py        # RNN cross-validation pipeline
 │    └── run_svm_pipeline.py        # SVM cross-validation pipeline
 ├── configs/
 │    ├── xgb_config.json
 │    ├── rnn_config.json
 │    └── svm_config.json
 ├── environment.yaml
 └── results/
```

## Environment Setup
```bash
# Clone repository
git clone https://github.com/<your-username>/Diabetic_Nephropathy_South_Korea.git
cd Diabetic_Nephropathy_South_Korea

# Create and activate conda environment
conda env create -f environment.yaml -n dn-pipeline-env
conda activate dn-pipeline-env
```

## Prediction Tasks
We perform binary classification for three key outcomes:

- **End-Stage Kidney Disease (ESKD)**
- **2-Year Composite Outcome**
- **3-Year Composite Outcome**

## Training Commands

### Run the Pipeline
```bash
python codes/run_{model}_pipeline.py --config config/{model}_config.json
```

## Output Structure
```bash
results/
 ├── <pipeline_name>_results/
 │    ├── results_summary.txt       # Summary metrics and classification report
 │    ├── predictions.csv           # True labels, predicted labels, probabilities
 │    ├── confusion_matrix.png      # Confusion matrix visualization
 │    ├── roc_curve.png             # ROC curve visualization
 │    └── additional_plots/         # Optional folder for model-specific visualization
```












