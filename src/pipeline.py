"""
Phase 1 reproduction pipeline:
  - Load & preprocess creditcard.csv
  - 5-fold stratified CV
  - Compare AGSS vs SMOTE, ADASYN, ROS using LSTM or RNN
  - Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC

Usage:
  python src/pipeline.py creditcard.csv [--model LSTM|RNN]
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS
from models import FraudLSTM, FraudRNN

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Hyperparameters ──────────────────────────────────────────────────────────
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
LR           = 1e-3
EPOCHS       = 20
BATCH_SIZE   = 256
THRESHOLD    = 0.5
N_FOLDS      = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_creditcard(path: str):
    df = pd.read_csv(path)
    df.dropna(subset=["Class"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y


def train_model(model, X_tr, y_tr):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_t = torch.from_numpy(X_tr).unsqueeze(1)  # (N, 1, F)
    y_t = torch.from_numpy(y_tr).float()
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def evaluate_model(model, X_val, y_val, threshold=THRESHOLD):
    model.eval()
    X_t = torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy":  accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall":    recall_score(y_val, preds, zero_division=0),
        "f1":        f1_score(y_val, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_val, probs),
        "pr_auc":    average_precision_score(y_val, probs),
    }


def run_cv(X, y, sampler, sampler_name: str, model_name: str = "LSTM"):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Resample ONLY on training fold
        X_tr_bal, y_tr_bal = sampler.fit_resample(X_tr, y_tr)
        y_tr_bal = y_tr_bal.astype(np.int64)

        cls = FraudRNN if model_name == "RNN" else FraudLSTM
        model = cls(
            n_features=X.shape[1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        train_model(model, X_tr_bal, y_tr_bal)
        metrics = evaluate_model(model, X_val, y_val)
        fold_metrics.append(metrics)
        print(f"  [{sampler_name}] fold {fold+1}/{N_FOLDS}  F1={metrics['f1']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}")

    elapsed = time.time() - t0
    keys = fold_metrics[0].keys()
    summary = {k: np.mean([m[k] for m in fold_metrics]) for k in keys}
    summary_std = {k + "_std": np.std([m[k] for m in fold_metrics]) for k in keys}
    summary.update(summary_std)
    summary["sampler"] = sampler_name
    summary["runtime_s"] = elapsed
    return summary


def main(data_path: str = None, model_name: str = "LSTM"):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")

    print(f"Loading data from {data_path} …")
    X, y = load_creditcard(data_path)
    print(f"  X shape: {X.shape}  |  fraud: {y.sum()} ({100*y.mean():.3f}%)")
    print(f"  Device: {DEVICE}  |  Model: {model_name}\n")

    samplers = {
        "ROS":   RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "ADASYN":ADASYN(random_state=RANDOM_STATE),
        "AGSS":  AGSS(eps=0.8, min_samples=3, n_neighbors=3, random_state=RANDOM_STATE),
    }

    results = []
    for name, sampler in samplers.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  |  Sampler: {name}")
        summary = run_cv(X, y, sampler, name, model_name)
        summary["model"] = model_name
        results.append(summary)
        print(f"  → mean F1={summary['f1']:.4f} ± {summary['f1_std']:.4f}  "
              f"Prec={summary['precision']:.4f}  Rec={summary['recall']:.4f}  "
              f"ROC-AUC={summary['roc_auc']:.4f}  ({summary['runtime_s']:.1f}s)")

    results_df = pd.DataFrame(results).set_index("sampler")
    fname = f"phase1_{model_name.lower()}_creditcard.csv"
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path)
    print(f"\nResults saved to {out_path}")

    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    print("\n" + "="*60)
    print(f"PHASE 1 RESULTS — {model_name} on creditcard.csv (5-fold CV mean)")
    print("="*60)
    print(results_df[cols].to_string(float_format="{:.4f}".format))
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default=None)
    parser.add_argument("--model", default="LSTM", choices=["LSTM", "RNN"])
    args = parser.parse_args()
    main(args.data_path, args.model)
