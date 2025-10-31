#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


# === Utility Functions ===
def import_data(csv_file, label_column):
    df = pd.read_csv(csv_file)
    df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip()
    patient_ids = df[df.columns[0]]
    labels = df[label_column].astype(int).values
    features = df.drop(columns=[df.columns[0], label_column])
    return patient_ids.values, features.values, labels, features.columns.tolist()


def save_results_csv(patient_ids, true_labels, pred_labels, pred_probs, filename):
    df = pd.DataFrame({
        "PatientID": patient_ids,
        "TrueLabel": true_labels,
        "PredLabel": pred_labels,
        "PredProb": pred_probs
    })
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")


def plot_and_save_confusion_matrix(true_labels, pred_labels, filename):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
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


# === Main Pipeline ===
def cross_validate_xgboost(config):
    print("Loading data...")
    patient_ids, X, y, feature_names = import_data(config["csv_file"], config["label_column"])

    skf = StratifiedKFold(n_splits=config.get("folds", 10), shuffle=True, random_state=config.get("seed", None))

    all_true, all_pred, all_prob, all_ids = [], [], [], []

    os.makedirs(config["save_directory"], exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{config.get('folds', 10)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_ids, test_ids = patient_ids[train_idx], patient_ids[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.2),
            n_estimators=config.get("num_est", 100),
            verbosity=0,
            random_state=config.get("seed", None)
        )

        model.fit(X_train_scaled, y_train)
        prob = model.predict_proba(X_test_scaled)[:, 1]
        pred = (prob >= config.get("prob_threshold", 0.5)).astype(int)

        all_true.extend(y_test)
        all_pred.extend(pred)
        all_prob.extend(prob)
        all_ids.extend(test_ids)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        mcc = matthews_corrcoef(y_test, pred)
        print(f"Fold {fold} Acc: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")

    # Aggregate results
    classification_rep = classification_report(all_true, all_pred, digits=4)
    overall_acc = accuracy_score(all_true, all_pred)
    overall_auc = roc_auc_score(all_true, all_prob)
    overall_mcc = matthews_corrcoef(all_true, all_pred)
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    specificity = tn / (tn + fp)

    summary_file = os.path.join(config["save_directory"], "results_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Classification Report (Aggregated over all folds):\n")
        f.write(classification_rep)
        f.write("\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"Overall AUC: {overall_auc:.4f}\n")
        f.write(f"Overall MCC: {overall_mcc:.4f}\n")
        f.write(f"Overall Specificity: {specificity:.4f}\n")
    print(f"Saved summary to {summary_file}")

    save_results_csv(all_ids, all_true, all_pred, all_prob, os.path.join(config["save_directory"], "predictions.csv"))
    plot_and_save_confusion_matrix(all_true, all_pred, os.path.join(config["save_directory"], "confusion_matrix.png"))
    plot_and_save_roc_curve(all_true, all_prob, os.path.join(config["save_directory"], "roc_curve.png"))


# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost cross-validation using JSON config")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    cross_validate_xgboost(config)
