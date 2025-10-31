#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import warnings
import json
import argparse

warnings.filterwarnings("ignore")


# ----------------- Data Import Functions ----------------- #
def import_data(csv_file):
    df = pd.read_csv(csv_file)
    patient_ids = df.iloc[:, 0].astype(str).str.strip()
    features = df.iloc[:, 1:316]  # Adjust if needed

    feats = []
    unique_ids = patient_ids.unique()
    for pid in unique_ids:
        patient_data = features[patient_ids == pid].values
        feats.append(patient_data)
    return feats, unique_ids, features.columns.tolist()


def import_labels(csv_file, label_column_name):
    df = pd.read_csv(csv_file)
    patient_ids = df.iloc[:, 0].astype(str).str.strip()
    labels_df = df[[df.columns[0], label_column_name]].copy()
    labels_df[df.columns[0]] = labels_df[df.columns[0]].astype(str).str.strip()
    labels_df = labels_df.drop_duplicates(subset=df.columns[0]).set_index(df.columns[0])
    unique_ids = patient_ids.unique()
    labels = labels_df.loc[unique_ids][label_column_name].values
    return labels.astype(np.int64)


def import_proteomics(proteomics_csv):
    df = pd.read_csv(proteomics_csv)
    df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip()
    df = df.set_index(df.columns[0])
    df = df.replace('#VALUE!', np.nan)
    df = df.astype(float)
    df = df.fillna(df.mean())
    return df


def import_covariates(covariates_csv):
    df = pd.read_csv(covariates_csv)
    df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip()
    df = df.set_index(df.columns[0])
    return df


# ----------------- Utility Functions ----------------- #
def regress_out(Z, M):
    model = LinearRegression().fit(Z, M)
    predictions = model.predict(Z)
    residuals = M - predictions
    return residuals


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


def save_results_csv(y_true, y_pred, y_prob, output_path, patient_ids=None):
    df = pd.DataFrame({
        "PatientID": patient_ids if patient_ids is not None else range(len(y_true)),
        "TrueLabel": y_true,
        "PredLabel": y_pred,
        "PredProb": y_prob
    })
    df.to_csv(output_path, index=False)


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


def spearman_corr_matrix(X, Y):
    from scipy.stats import rankdata
    X_ranked = np.apply_along_axis(rankdata, 0, X)
    Y_ranked = np.apply_along_axis(rankdata, 0, Y)
    X_ranked_c = X_ranked - X_ranked.mean(axis=0)
    Y_ranked_c = Y_ranked - Y_ranked.mean(axis=0)
    X_norm = X_ranked_c / np.linalg.norm(X_ranked_c, axis=0)
    Y_norm = Y_ranked_c / np.linalg.norm(Y_ranked_c, axis=0)
    corr_matrix = np.dot(X_norm.T, Y_norm)
    return corr_matrix


# ----------------- Main Pipeline ----------------- #
def cross_validate_xgboost_with_feature_selection(config):
    # Load inputs
    feats, unique_ids, feature_names = import_data(config["csv_file"])
    labels = import_labels(config["csv_file"], config["label_column"])
    proteomics_df = import_proteomics(config["proteomics_file"])
    covariates_df = import_covariates(config["covariates_file"])

    skf = StratifiedKFold(
        n_splits=config.get("folds", 10),
        shuffle=True,
        random_state=config.get("seed", None)
    )

    all_true, all_pred, all_prob, all_ids = [], [], [], []
    save_directory = config.get("save_directory", "./results")
    os.makedirs(save_directory, exist_ok=True)

    binary_cols = ["Sex", "Hipertension", "Ischemic_Heart_Disease", "Stroke_History"]
    cont_cols = ["Age", "Weight", "Height", "Diabetic_Years"]

    for fold, (train_idx, test_idx) in enumerate(skf.split(feats, labels), 1):
        feats_train = [feats[i] for i in train_idx]
        feats_test = [feats[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_ids, test_ids = unique_ids[train_idx], unique_ids[test_idx]

        feats_train_for_selection = [np.copy(p) for p in feats_train]

        covariates_train = covariates_df.loc[train_ids]
        covariates_test = covariates_df.loc[test_ids]
        cov_scaler = StandardScaler().fit(covariates_train[cont_cols])
        cont_train_scaled = cov_scaler.transform(covariates_train[cont_cols])
        cont_test_scaled = cov_scaler.transform(covariates_test[cont_cols])
        covariates_train_scaled = np.hstack([cont_train_scaled, covariates_train[binary_cols].values])
        covariates_test_scaled = np.hstack([cont_test_scaled, covariates_test[binary_cols].values])

        proteomics_train = proteomics_df.loc[train_ids].dropna(axis=1)
        proteomics_test = proteomics_df.loc[test_ids][proteomics_train.columns].fillna(proteomics_train.mean())
        proteomics_scaler = StandardScaler().fit(proteomics_train.values)
        proteomics_train_scaled = proteomics_scaler.transform(proteomics_train.values)
        proteomics_test_scaled = proteomics_scaler.transform(proteomics_test.values)

        all_train = np.vstack(feats_train)
        feats_scaler = StandardScaler().fit(all_train)
        feats_train_scaled = [feats_scaler.transform(p) for p in feats_train]
        feats_test_scaled = [feats_scaler.transform(p) for p in feats_test]
        feats_train_for_selection_scaled = [feats_scaler.transform(p) for p in feats_train_for_selection]

        X_train_model = aggregate_patient_features(feats_train_scaled)
        X_test_model = aggregate_patient_features(feats_test_scaled)
        X_train_for_selection = aggregate_patient_features(feats_train_for_selection_scaled)

        X_train_resid = regress_out(covariates_train_scaled, X_train_for_selection)
        Y_train_resid = regress_out(covariates_train_scaled, proteomics_train_scaled)

        corr_matrix = spearman_corr_matrix(X_train_resid, Y_train_resid)
        corr_matrix = np.nan_to_num(corr_matrix)
        mean_abs_corr = np.mean(np.abs(corr_matrix), axis=1)

        fold_dir = os.path.join(save_directory, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        rank_df = pd.DataFrame({
            'glom_feature_index': np.arange(X_train_model.shape[1]),
            'mean_abs_spearman_corr': mean_abs_corr
        })
        rank_df.to_csv(os.path.join(fold_dir, "feature_correlations.csv"), index=False)

        top_features_indices = np.argsort(-mean_abs_corr)[:config.get("top_features_num", 50)]
        top_feature_names = [feature_names[i] for i in top_features_indices]
        with open(os.path.join(fold_dir, "selected_feature_names.txt"), "w") as f:
            for name in top_feature_names:
                f.write(name + "\n")

        X_train = X_train_model[:, top_features_indices]
        X_test = X_test_model[:, top_features_indices]

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.2),
            n_estimators=config.get("num_est", 100),
            verbosity=0,
            random_state=config.get("seed", 42)
        )

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= config.get("prob_threshold", 0.35)).astype(int)

        all_true.extend(y_test)
        all_pred.extend(pred)
        all_prob.extend(prob)
        all_ids.extend(test_ids)

        print(f"Fold {fold} Acc: {accuracy_score(y_test, pred):.4f} | "
              f"AUC: {roc_auc_score(y_test, prob):.4f} | "
              f"MCC: {matthews_corrcoef(y_test, pred):.4f}")

    # Save summary
    classification_rep = classification_report(all_true, all_pred, digits=4)
    overall_acc = accuracy_score(all_true, all_pred)
    overall_auc = roc_auc_score(all_true, all_prob)
    overall_mcc = matthews_corrcoef(all_true, all_pred)
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    specificity = tn / (tn + fp)

    with open(os.path.join(save_directory, "results_summary.txt"), "w") as f:
        f.write("Classification Report (Aggregated over all folds):\n")
        f.write(classification_rep)
        f.write("\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"Overall AUC: {overall_auc:.4f}\n")
        f.write(f"Overall MCC: {overall_mcc:.4f}\n")
        f.write(f"Overall Specificity: {specificity:.4f}\n")

    save_results_csv(all_true, all_pred, all_prob, os.path.join(save_directory, "predictions.csv"), all_ids)
    plot_and_save_confusion_matrix(all_true, all_pred, os.path.join(save_directory, "confusion_matrix.png"))
    plot_and_save_roc_curve(all_true, all_prob, os.path.join(save_directory, "roc_curve.png"))


# ----------------- CLI Entry Point ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost CV pipeline with feature selection and covariate regression")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    cross_validate_xgboost_with_feature_selection(config)
