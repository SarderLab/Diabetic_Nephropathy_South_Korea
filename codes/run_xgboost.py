#!/usr/bin/env python3
"""
XGBoost Cross-Validation Pipeline
Usage:
    python xgb_cv_runner.py --config config.json
"""

import os
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# -------------------------------
# Data Import
# -------------------------------
def import_data(csv_file, feature_start, feature_end):
    df = pd.read_csv(csv_file)
    patient_ids = df.iloc[:, 0].astype(str).str.strip()
    features = df.iloc[:, feature_start:feature_end]  # <-- use config range

    feats = []
    unique_ids = patient_ids.unique()
    for pid in unique_ids:
        patient_data = features[patient_ids == pid].values
        feats.append(patient_data)
    return feats, unique_ids


def import_labels(csv_file, label_column_name):
    df = pd.read_csv(csv_file)
    patient_ids = df.iloc[:, 0].astype(str).str.strip()
    labels_df = df[[df.columns[0], label_column_name]].copy()
    labels_df[df.columns[0]] = labels_df[df.columns[0]].astype(str).str.strip()
    labels_df = labels_df.drop_duplicates(subset=df.columns[0]).set_index(df.columns[0])
    unique_ids = patient_ids.unique()
    labels = labels_df.loc[unique_ids][label_column_name].values
    return labels.astype(np.int64)


# -------------------------------
# Feature Aggregation
# -------------------------------
def aggregate_patient_features(feats, method="mean"):
    agg_feats = []
    for patient in feats:
        if method == "mean":
            agg_feats.append(np.mean(patient, axis=0))
        elif method == "max":
            agg_feats.append(np.max(patient, axis=0))
        elif method == "min":
            agg_feats.append(np.min(patient, axis=0))
        else:
            raise ValueError("Unsupported aggregation method.")
    return np.array(agg_feats)


# -------------------------------
# Save Utilities
# -------------------------------
def save_results_csv(true_labels, pred_labels, pred_probs, filename):
    df = pd.DataFrame({
        "TrueLabel": true_labels,
        "PredLabel": pred_labels,
        "PredProb": pred_probs
    })
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")


def plot_and_save_confusion_matrix(true_labels, pred_labels, filename):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")


def plot_and_save_roc_curve(true_labels, pred_probs, filename):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Saved ROC curve to {filename}")


# -------------------------------
# Main Cross-Validation Function
# -------------------------------
def cross_validate_xgboost(config):
    csv_file = config["csv_file"]
    label_column = config["label_column"]
    folds = config.get("folds", 10)
    agg_method = config.get("agg_method", "mean")
    save_directory = config.get("save_directory", "./results")
    num_est = config.get("num_estimators", 100)
    seed = config.get("seed", 42)
    prob_threshold = config.get("prob_threshold", 0.35)

    print("Loading data...")
    feature_start = config.get("feature_start", 1)
    feature_end = config.get("feature_end", 208)
    feats, unique_ids = import_data(csv_file, feature_start, feature_end)
    labels = import_labels(csv_file, label_column)
    print(f"Total patients: {len(labels)}")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    all_true, all_pred, all_prob = [], [], []

    os.makedirs(save_directory, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(feats, labels), 1):
        print(f"\nFold {fold}/{folds}")

        feats_train = [feats[i] for i in train_idx]
        feats_test = [feats[i] for i in test_idx]

        # Fit scaler only on training data
        all_train = np.vstack(feats_train)
        scaler = StandardScaler().fit(all_train)

        feats_train = [scaler.transform(p) for p in feats_train]
        feats_test = [scaler.transform(p) for p in feats_test]

        X_train = aggregate_patient_features(feats_train, method=agg_method)
        X_test = aggregate_patient_features(feats_test, method=agg_method)
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
            max_depth=4,
            learning_rate=0.2,
            n_estimators=num_est,
            verbosity=0,
            random_state=seed
        )

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= prob_threshold).astype(int)

        all_true.extend(y_test)
        all_pred.extend(pred)
        all_prob.extend(prob)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        mcc = matthews_corrcoef(y_test, pred)

        print(f"Fold {fold} Acc: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")

    # --- Aggregate Metrics ---
    classification_rep = classification_report(all_true, all_pred, digits=4)
    overall_acc = accuracy_score(all_true, all_pred)
    overall_auc = roc_auc_score(all_true, all_prob)
    overall_mcc = matthews_corrcoef(all_true, all_pred)
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    specificity = tn / (tn + fp)

    print("\nClassification Report (Aggregated over all folds):")
    print(classification_rep)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall MCC: {overall_mcc:.4f}")
    print(f"Overall Specificity: {specificity:.4f}")

    # --- Save Results ---
    with open(os.path.join(save_directory, "results_summary.txt"), "w") as f:
        f.write("Classification Report (Aggregated over all folds):\n")
        f.write(classification_rep)
        f.write("\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"Overall AUC: {overall_auc:.4f}\n")
        f.write(f"Overall MCC: {overall_mcc:.4f}\n")
        f.write(f"Overall Specificity: {specificity:.4f}\n")

    save_results_csv(all_true, all_pred, all_prob, os.path.join(save_directory, "predictions.csv"))
    plot_and_save_confusion_matrix(all_true, all_pred, os.path.join(save_directory, "confusion_matrix.png"))
    plot_and_save_roc_curve(all_true, all_prob, os.path.join(save_directory, "roc_curve.png"))


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost Cross-Validation with Config File")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = json.load(f)

    cross_validate_xgboost(config)
