"""
Phase 2 — German credit-risk dataset pipeline.
Uses local german_credit_data.csv (9 features + Risk column).
Best config from tune_german.py: RNN, hidden=32, epochs=200, lr=5e-4, SMOTE, threshold=0.8.
F1 reported for class 0 (good credit / majority) — matches the paper's metric convention.
"""

import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# Best config from tune_german.py
HIDDEN_SIZE = 32
NUM_LAYERS  = 2
DROPOUT     = 0.3
LR          = 5e-4
EPOCHS      = 200
BATCH_SIZE  = 32
THRESHOLD   = 0.8      # optimised for F1 of good-credit class (paper's convention)
N_FOLDS     = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_german(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    y = (df["Risk"] == "bad").astype(int).values   # bad=1 (minority), good=0 (majority)
    df = df.drop("Risk", axis=1)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.mean(numeric_only=True))
    X = StandardScaler().fit_transform(df.values).astype(np.float32)
    return X, y


def make_model(model_name: str, n_features: int):
    cls = FraudRNN if model_name == "RNN" else FraudLSTM
    return cls(n_features=n_features, hidden_size=HIDDEN_SIZE,
               num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)


def train_model(model, X_tr, y_tr):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()


def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)

    return {
        # Paper's metric: F1 for class 0 (good credit = majority)
        "f1_good":   f1_score(y_val, preds, pos_label=0, zero_division=0),
        # Standard minority-class metrics for completeness
        "f1_bad":    f1_score(y_val, preds, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y_val, preds, average="weighted", zero_division=0),
        "accuracy":  accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, pos_label=0, zero_division=0),
        "recall":    recall_score(y_val, preds, pos_label=0, zero_division=0),
        "roc_auc":   roc_auc_score(y_val, probs),
        "pr_auc":    average_precision_score(1 - y_val, 1 - probs),
    }


def run_cv(X, y, sampler, sampler_name: str, model_name: str = "RNN"):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_tr_bal, y_tr_bal = sampler.fit_resample(X_tr, y_tr)
        y_tr_bal = y_tr_bal.astype(np.int64)

        model = make_model(model_name, X.shape[1])
        train_model(model, X_tr_bal, y_tr_bal)
        metrics = evaluate_model(model, X_val, y_val)
        fold_metrics.append(metrics)
        print(f"  [{sampler_name}] fold {fold+1}/{N_FOLDS}  "
              f"F1_good={metrics['f1_good']:.4f}  F1_bad={metrics['f1_bad']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")

    elapsed = time.time() - t0
    keys = fold_metrics[0].keys()
    summary = {k: np.mean([m[k] for m in fold_metrics]) for k in keys}
    summary.update({k + "_std": np.std([m[k] for m in fold_metrics]) for k in keys})
    summary["sampler"] = sampler_name
    summary["model"]   = model_name
    summary["runtime_s"] = elapsed
    return summary


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "german_credit_data.csv")
    print(f"Loading German credit dataset from {path} …")
    X, y = load_german(path)
    print(f"  X shape: {X.shape}  |  bad credit: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Device: {DEVICE}  |  threshold: {THRESHOLD}  |  metric: F1 for good credit (class 0)\n")

    samplers = {
        "ROS":   RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "ADASYN":ADASYN(random_state=RANDOM_STATE),
        "AGSS":  AGSS(eps=0.8, min_samples=2, n_neighbors=3, random_state=RANDOM_STATE),
    }

    results = []
    for model_name in ["RNN", "LSTM"]:
        for name, sampler in samplers.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_name}  |  Sampler: {name}")
            summary = run_cv(X, y, sampler, name, model_name)
            results.append(summary)
            print(f"  → F1_good={summary['f1_good']:.4f} ± {summary['f1_good_std']:.4f}  "
                  f"F1_bad={summary['f1_bad']:.4f}  Acc={summary['accuracy']:.4f}  "
                  f"ROC-AUC={summary['roc_auc']:.4f}  ({summary['runtime_s']:.1f}s)")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_german_final.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    cols = ["model", "sampler", "accuracy", "f1_good", "f1_bad", "f1_weighted", "roc_auc"]
    print("\n" + "="*65)
    print("PHASE 2 RESULTS — German credit (5-fold CV, F1 for good credit)")
    print("="*65)
    print(results_df[cols].to_string(index=False, float_format="{:.4f}".format))
    return results_df


if __name__ == "__main__":
    main()
