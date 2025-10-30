#!/usr/bin/env python3
import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    matthews_corrcoef, recall_score, RocCurveDisplay
)
from sklearn.model_selection import StratifiedKFold

# --- Data Loading ---
def import_data(csv_file):
    df = pd.read_csv(csv_file)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    patient_ids = df.iloc[:, 0]
    features = df.iloc[:, 1:316]
    feats = []
    unique_ids = patient_ids.unique()
    for pid in unique_ids:
        patient_data = features[patient_ids == pid].values
        feats.append(patient_data.tolist())
    return feats

def import_labels(csv_file, label_column_name):
    df = pd.read_csv(csv_file)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    patient_ids = df.iloc[:, 0]
    labels_df = df[[patient_ids.name, label_column_name]].drop_duplicates(subset=patient_ids.name)
    labels_df = labels_df.set_index(patient_ids.name)
    unique_ids = patient_ids.unique()
    labels = labels_df.loc[unique_ids][label_column_name].values
    return labels.astype(np.int32)

# --- Feature Standardization ---
def standardize_features(feats, mean=None, std=None):
    if mean is None or std is None:
        mean_record = [np.mean(np.asarray(case), axis=0) for case in feats]
        std_record = [np.std(np.asarray(case), axis=0) for case in feats]
        mean = np.nanmean(mean_record, axis=0)
        std = np.nanstd(std_record, axis=0)
        std[std == 0] = 1.0
        std = np.nan_to_num(std)
        mean[np.isinf(mean)] = 0

    standardized_feats = []
    for patient in feats:
        patient = np.array(patient)
        patient -= mean
        patient /= std
        patient = np.nan_to_num(patient)
        standardized_feats.append(patient)
    return standardized_feats, mean, std

# --- Mean Pooling ---
def mean_pool(feats):
    return np.array([np.mean(np.array(patient), axis=0) for patient in feats])

# --- Main Training Pipeline ---
def run_svm_pipeline(config):
    data_path = config["data_path"]
    label_col = config["label_column"]
    output_root = config["output_root"]
    n_splits = config.get("n_splits", 10)
    seed = config.get("seed", None)

    X = import_data(data_path)
    y = import_labels(data_path, label_col)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true, all_pred, all_probas = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        X_train_std, mean, std = standardize_features(X_train)
        X_test_std, _, _ = standardize_features(X_test, mean, std)

        # Mean pooling
        X_train_pooled = mean_pool(X_train_std)
        X_test_pooled = mean_pool(X_test_std)

        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
        clf.fit(X_train_pooled, y_train)
        y_pred = clf.predict(X_test_pooled)
        y_proba = clf.predict_proba(X_test_pooled)[:, 1]

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_probas.extend(y_proba)

    overall_class_report = classification_report(all_true, all_pred, digits=4)
    overall_cm = confusion_matrix(all_true, all_pred)
    overall_auc = roc_auc_score(all_true, all_probas)
    overall_specificity = recall_score(all_true, all_pred, pos_label=0, zero_division=0)
    overall_mcc = matthews_corrcoef(all_true, all_pred)

    summary = (
        f"=== Overall Classification Report ===\n{overall_class_report}\n\n"
        f"Confusion Matrix:\n{overall_cm}\n\n"
        f"AUC: {overall_auc:.4f}\n"
        f"Specificity: {overall_specificity:.4f}\n"
        f"MCC: {overall_mcc:.4f}\n"
    )

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)

    results_csv_path = os.path.join(output_root, "all_predictions.csv")
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label", "predicted_probability"])
        for t, p, prob in zip(all_true, all_pred, all_probas):
            writer.writerow([f"{t:.4f}", f"{p:.4f}", f"{prob:.4f}"])

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(overall_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_root, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(all_true, all_probas)
    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.savefig(os.path.join(output_root, "roc_curve.png"))
    plt.close()

# --- Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SVM pipeline using JSON config file")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    run_svm_pipeline(config)
