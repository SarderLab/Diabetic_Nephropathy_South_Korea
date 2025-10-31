import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef, roc_curve
)
import tensorflow as tf
import csv


# ===============================
# --- GPU Setup ---
# ===============================
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] {len(gpus)} GPU(s) available. Memory growth enabled.")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
    else:
        print("[INFO] No GPU found, running on CPU.")


# ===============================
# --- Data Loading ---
# ===============================
def import_data(csv_file):
    df = pd.read_csv(csv_file)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    patient_ids = df.iloc[:, 0]
    features = df.iloc[:, 1:208]  # Adjust column range as needed
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


# ===============================
# --- Feature Standardization ---
# ===============================
def standardize_features(feats, mean=None, std=None):
    if mean is None or std is None:
        mean_record = []
        std_record = []
        for case in feats:
            case_arr = np.asarray(case)
            mean_record.append(np.mean(case_arr, axis=0))
            std_record.append(np.std(case_arr, axis=0))
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


# ===============================
# --- Balanced Random Batcher ---
# ===============================
def balanced_random_batcher(feats, time_step, batch_size, labels, rng):
    num_features = len(feats[0][0])
    class0_idx = [i for i, lbl in enumerate(labels) if lbl == 0]
    class1_idx = [i for i, lbl in enumerate(labels) if lbl == 1]

    half_batch = batch_size // 2
    batched_data = np.zeros((batch_size, time_step, num_features))
    batched_labels = np.zeros((batch_size, 2))

    for i in range(half_batch):
        idx = rng.choice(class0_idx)
        case = np.array(feats[idx])
        label = labels[idx]
        batched_labels[i, label] = 1
        for j in range(time_step):
            sample_idx = rng.choice(len(case))
            batched_data[i, j, :] = case[sample_idx]

    for i in range(half_batch, batch_size):
        idx = rng.choice(class1_idx)
        case = np.array(feats[idx])
        label = labels[idx]
        batched_labels[i, label] = 1
        for j in range(time_step):
            sample_idx = rng.choice(len(case))
            batched_data[i, j, :] = case[sample_idx]

    if batch_size % 2 == 1:
        idx = rng.choice(class0_idx + class1_idx)
        case = np.array(feats[idx])
        label = labels[idx]
        batched_labels[-1, label] = 1
        for j in range(time_step):
            sample_idx = rng.choice(len(case))
            batched_data[-1, j, :] = case[sample_idx]

    return batched_data, batched_labels


# ===============================
# --- Full Sequence Batch Generator ---
# ===============================
def generate_full_sequence_batch(feats, time_step):
    batch_x = []
    for patient in feats:
        patient = np.array(patient)
        if len(patient) >= time_step:
            indices = np.linspace(0, len(patient) - 1, time_step, dtype=int)
            seq = patient[indices]
        else:
            repeats = (time_step // len(patient)) + 1
            extended_seq = np.tile(patient, (repeats, 1))
            seq = extended_seq[:time_step]
        batch_x.append(seq)
    return np.array(batch_x)


# ===============================
# --- Build RNN Model ---
# ===============================
def build_model(num_input, seq_len, num_classes, drop_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, num_input)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.Dense(num_input, activation=tf.nn.leaky_relu),
        tf.keras.layers.LSTM(75, return_sequences=True),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.LSTM(25),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ===============================
# --- Main K-Fold Training Function ---
# ===============================
def run_kfold_training(cfg):
    csv_file = cfg["csv_file"]
    label_column = cfg["label_column"]
    seq_len = cfg["seq_len"]
    drop = cfg["drop"]
    num_classes = cfg["num_classes"]
    training_steps = cfg["training_steps"]
    batch_size = cfg["batch_size"]
    save_interval = cfg["save_interval"]
    learning_rate = cfg["learning_rate"]
    output_root = cfg["output_root"]
    seed = cfg.get("seed", None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    print(f"[INFO] Running K-Fold training with config:\n{json.dumps(cfg, indent=4)}")

    raw_features = import_data(csv_file)
    all_labels = np.array(import_labels(csv_file, label_column))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    all_true, all_pred, all_probas = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw_features, all_labels)):
        print(f"\n[INFO] Starting Fold {fold + 1}/10")
        raw_train_feats = [raw_features[i] for i in train_idx]
        raw_test_feats = [raw_features[i] for i in test_idx]
        train_lbls = all_labels[train_idx]
        test_lbls = all_labels[test_idx]

        train_feats, train_mean, train_std = standardize_features(raw_train_feats)
        test_feats, _, _ = standardize_features(raw_test_feats, train_mean, train_std)

        num_input = len(train_feats[0][0])
        model = build_model(num_input, seq_len, num_classes, drop, learning_rate)

        fold_dir = os.path.join(output_root, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        metrics_file = os.path.join(fold_dir, "metrics.csv")

        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "accuracy", "precision", "recall", "specificity", "auc", "mcc", "loss"])

        for step in range(training_steps + 1):
            batch_x, batch_y = balanced_random_batcher(train_feats, seq_len, batch_size, train_lbls, rng)
            loss, acc = model.train_on_batch(batch_x, batch_y)

            if step % save_interval == 0 or step == training_steps:
                val_x = generate_full_sequence_batch(test_feats, seq_len)
                val_y = tf.keras.utils.to_categorical(test_lbls, num_classes=num_classes)
                preds_val = model.predict(val_x, verbose=0)
                y_true = np.argmax(val_y, axis=1)
                y_pred = np.argmax(preds_val, axis=1)
                probas = preds_val[:, 1]

                acc = np.mean(y_true == y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                auc_val = roc_auc_score(y_true, probas) if len(np.unique(y_true)) > 1 else float('nan')
                mcc = matthews_corrcoef(y_true, y_pred)

                print(f"[Fold {fold + 1} | Step {step}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, Spec: {spec:.4f}, AUC: {auc_val:.4f}, MCC: {mcc:.4f}, Loss: {loss:.4f}")

                with open(metrics_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, acc, prec, rec, spec, auc_val, mcc, loss])

        preds_full = model.predict(generate_full_sequence_batch(raw_test_feats, seq_len), verbose=0)
        all_true.extend(test_lbls.tolist())
        all_pred.extend(np.argmax(preds_full, axis=1).tolist())
        all_probas.extend(preds_full[:, 1].tolist())

    summarize_results(all_true, all_pred, all_probas, output_root)


# ===============================
# --- Summary ---
# ===============================
def summarize_results(all_true, all_pred, all_probas, output_root):
    os.makedirs(output_root, exist_ok=True)
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(all_true, all_pred, zero_division=0, digits=4)
    auc_val = roc_auc_score(all_true, all_probas) if len(np.unique(all_true)) > 1 else float('nan')
    spec = recall_score(all_true, all_pred, pos_label=0, zero_division=0)
    mcc = matthews_corrcoef(all_true, all_pred)

    summary = f"{report}\n\nConfusion Matrix:\n{cm}\nAUC: {auc_val:.4f}\nSpecificity: {spec:.4f}\nMCC: {mcc:.4f}"
    print(summary)

    with open(os.path.join(output_root, "overall_metrics.txt"), "w") as f:
        f.write(summary)


# ===============================
# --- Entry Point ---
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RNN model with JSON configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    setup_gpu()
    run_kfold_training(config)
