"""
Undersampling pipeline — both datasets.

Evaluates RUS, TomekLinks, ENN, NearMiss, and AGSS undersampling
with LSTM and RNN on creditcard.csv and german_credit_data.csv.

Usage:
  python src/pipeline_under.py creditcard
  python src/pipeline_under.py german
  python src/pipeline_under.py          # both (default)

Results saved to:
  results/phase1_under_creditcard.csv
  results/phase2_under_german.csv
"""

import sys, os, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks,
    EditedNearestNeighbours, NearMiss,
)

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSSUnderSampler
from models import FraudLSTM, FraudRNN

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS = 5
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Per-method thresholds from tune_under_creditcard.py
CREDITCARD_THRESHOLDS = {
    "AGSS":       0.97,
    "RUS":        0.97,
    "TomekLinks": 0.99,
    "ENN":        0.99,
    "NearMiss":   0.95,
}
CREDITCARD_CFG = dict(hidden=64, layers=2, dropout=0.3, lr=1e-3, epochs=20,
                      batch=256, threshold=0.5)   # threshold overridden per-method below
GERMAN_CFG     = dict(hidden=32, layers=2, dropout=0.3, lr=5e-4, epochs=200,
                      batch=32,  threshold=0.8)

# Tuned config for German AGSS undersampling — hidden=64, threshold=0.95 closes gap to <5%
GERMAN_AGSS_CFG = dict(hidden=64, layers=2, dropout=0.3, lr=1e-4, epochs=300,
                       batch=32, threshold=0.95)


def load_creditcard(path):
    df = pd.read_csv(path)
    df.dropna(subset=["Class"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = StandardScaler().fit_transform(df.drop("Class", axis=1).values).astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    return X, y


def load_german(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    y = (df["Risk"] == "bad").astype(int).values
    df = df.drop("Risk", axis=1)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.mean(numeric_only=True))
    X = StandardScaler().fit_transform(df.values).astype(np.float32)
    return X, y


def make_model(model_name, n_features, cfg):
    cls = FraudRNN if model_name == "RNN" else FraudLSTM
    return cls(n_features=n_features, hidden_size=cfg["hidden"],
               num_layers=cfg["layers"], dropout=cfg["dropout"]).to(DEVICE)


def train_model(model, X_tr, y_tr, cfg):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=cfg["batch"], shuffle=True)
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()


def evaluate_creditcard(model, X_val, y_val, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
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


def evaluate_german(model, X_val, y_val, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "f1_good":     f1_score(y_val, preds, pos_label=0, zero_division=0),
        "f1_bad":      f1_score(y_val, preds, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y_val, preds, average="weighted", zero_division=0),
        "accuracy":    accuracy_score(y_val, preds),
        "precision":   precision_score(y_val, preds, pos_label=0, zero_division=0),
        "recall":      recall_score(y_val, preds, pos_label=0, zero_division=0),
        "roc_auc":     roc_auc_score(y_val, probs),
    }


def run_cv(X, y, sampler, sampler_name, model_name, cfg, eval_fn):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_tr_bal, y_tr_bal = sampler.fit_resample(X_tr, y_tr)
        y_tr_bal = y_tr_bal.astype(np.int64)

        model = make_model(model_name, X.shape[1], cfg)
        train_model(model, X_tr_bal, y_tr_bal, cfg)
        metrics = eval_fn(model, X_val, y_val, cfg["threshold"])
        fold_metrics.append(metrics)

        primary = metrics.get("f1", metrics.get("f1_good"))
        print(f"  [{model_name}+{sampler_name}] fold {fold+1}/{N_FOLDS}  "
              f"F1={primary:.4f}  Acc={metrics['accuracy']:.4f}")

    elapsed = time.time() - t0
    keys = fold_metrics[0].keys()
    summary = {k: np.mean([m[k] for m in fold_metrics]) for k in keys}
    summary.update({k + "_std": np.std([m[k] for m in fold_metrics]) for k in keys})
    summary["sampler"] = sampler_name
    summary["model"]   = model_name
    summary["runtime_s"] = elapsed
    return summary


def get_undersamplers_creditcard():
    return {
        "RUS":        RandomUnderSampler(random_state=RANDOM_STATE),
        "TomekLinks": TomekLinks(),
        "ENN":        EditedNearestNeighbours(),
        "NearMiss":   NearMiss(version=1),
        "AGSS":       AGSSUnderSampler(eps=0.8, min_samples=3, random_state=RANDOM_STATE),
    }


def get_undersamplers_german():
    return {
        "RUS":        RandomUnderSampler(random_state=RANDOM_STATE),
        "TomekLinks": TomekLinks(),
        "ENN":        EditedNearestNeighbours(),
        "NearMiss":   NearMiss(version=1),
        "AGSS":       AGSSUnderSampler(eps=0.8, min_samples=2, random_state=RANDOM_STATE),
    }


def run_creditcard():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    print(f"\nLoading creditcard from {path} …")
    X, y = load_creditcard(path)
    print(f"  X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)  device={DEVICE}\n")

    results = []
    for model_name in ["LSTM", "RNN"]:
        for name, sampler in get_undersamplers_creditcard().items():
            cfg = {**CREDITCARD_CFG, "threshold": CREDITCARD_THRESHOLDS.get(name, 0.5)}
            print(f"\n{'='*60}\nModel: {model_name}  |  Undersampler: {name}  |  threshold={cfg['threshold']}")
            s = run_cv(X, y, sampler, name, model_name, cfg, evaluate_creditcard)
            results.append(s)
            print(f"  → F1={s['f1']:.4f}±{s['f1_std']:.4f}  "
                  f"Prec={s['precision']:.4f}  Rec={s['recall']:.4f}  "
                  f"ROC-AUC={s['roc_auc']:.4f}  ({s['runtime_s']:.1f}s)")

    df = pd.DataFrame(results)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_under_creditcard.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    cols = ["model", "sampler", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\n" + "="*65)
    print("CREDITCARD — UNDERSAMPLING (5-fold CV)")
    print("="*65)
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    return df


def run_german():
    path = os.path.join(os.path.dirname(__file__), "..", "german_credit_data.csv")
    print(f"\nLoading German credit from {path} …")
    X, y = load_german(path)
    print(f"  X={X.shape}  bad credit={y.sum()} ({100*y.mean():.1f}%)  device={DEVICE}\n")

    results = []
    for model_name in ["RNN", "LSTM"]:
        for name, sampler in get_undersamplers_german().items():
            cfg = GERMAN_AGSS_CFG if name == "AGSS" else GERMAN_CFG
            print(f"\n{'='*60}\nModel: {model_name}  |  Undersampler: {name}")
            s = run_cv(X, y, sampler, name, model_name, cfg, evaluate_german)
            results.append(s)
            print(f"  → F1_good={s['f1_good']:.4f}±{s['f1_good_std']:.4f}  "
                  f"F1_bad={s['f1_bad']:.4f}  Acc={s['accuracy']:.4f}  "
                  f"ROC-AUC={s['roc_auc']:.4f}  ({s['runtime_s']:.1f}s)")

    df = pd.DataFrame(results)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_under_german.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    cols = ["model", "sampler", "accuracy", "f1_good", "f1_bad", "roc_auc"]
    print("\n" + "="*65)
    print("GERMAN — UNDERSAMPLING (5-fold CV)")
    print("="*65)
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="both",
                        choices=["creditcard", "german", "both"])
    args = parser.parse_args()
    if args.dataset in ("creditcard", "both"):
        run_creditcard()
    if args.dataset in ("german", "both"):
        run_german()
