"""
Transformer pipeline — both datasets, both sampling strategies.

Usage:
  python src/pipeline_transformer.py creditcard_over
  python src/pipeline_transformer.py creditcard_under
  python src/pipeline_transformer.py german_over
  python src/pipeline_transformer.py german_under
  python src/pipeline_transformer.py          # all four (default)

Results saved to:
  results/phase1_transformer_over_creditcard.csv
  results/phase1_transformer_under_creditcard.csv
  results/phase2_transformer_over_german.csv
  results/phase2_transformer_under_german.csv
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
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks,
    EditedNearestNeighbours, NearMiss,
)

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS, AGSSUnderSampler
from models import FraudTransformer

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS = 5
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.975, 0.98, 0.985, 0.99]

CC_OVER_CFG  = dict(d_model=64, nhead=4, layers=2, dropout=0.3, lr=1e-4, epochs=20, batch=256)
CC_UNDER_CFG = dict(d_model=64, nhead=4, layers=2, dropout=0.3, lr=1e-4, epochs=50, batch=256)
DE_CFG       = dict(d_model=32, nhead=4, layers=2, dropout=0.3, lr=5e-4, epochs=300, batch=32)


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


def make_model(n_features, cfg):
    return FraudTransformer(
        n_features=n_features,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["layers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)


def train_model(model, X_tr, y_tr, cfg):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=cfg["batch"], shuffle=True)

    log_every = max(1, cfg["epochs"] // 5)
    model.train()
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch+1:3d}/{cfg['epochs']}  loss={epoch_loss/len(loader):.4f}",
                  flush=True)


def run_cv(X, y, sampler, sampler_name, cfg, is_german=False, tune_threshold=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        y_bal = y_bal.astype(np.int64)

        model = make_model(X.shape[1], cfg)
        train_model(model, X_bal, y_bal, cfg)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        all_probs.append(probs)
        all_y.append(y_val)

        f1_at_half = f1_score(y_val, (probs >= 0.5).astype(int), zero_division=0)
        print(f"  [Transformer+{sampler_name}] fold {fold+1}/{N_FOLDS}  "
              f"F1@0.5={f1_at_half:.4f}  ROC={roc_auc_score(y_val, probs):.4f}", flush=True)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)
    elapsed   = time.time() - t0

    if tune_threshold:
        best_f1, threshold = 0.0, 0.5
        for t in THRESHOLDS:
            f1 = f1_score(all_y, (all_probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, threshold = f1, t
    else:
        threshold = 0.5

    preds = (all_probs >= threshold).astype(int)

    if is_german:
        summary = {
            "f1_good":     f1_score(all_y, preds, pos_label=0, zero_division=0),
            "f1_bad":      f1_score(all_y, preds, pos_label=1, zero_division=0),
            "f1_weighted": f1_score(all_y, preds, average="weighted", zero_division=0),
            "accuracy":    accuracy_score(all_y, preds),
            "roc_auc":     roc_auc_score(all_y, all_probs),
        }
    else:
        summary = {
            "accuracy":  accuracy_score(all_y, preds),
            "precision": precision_score(all_y, preds, zero_division=0),
            "recall":    recall_score(all_y, preds, zero_division=0),
            "f1":        f1_score(all_y, preds, zero_division=0),
            "roc_auc":   roc_auc_score(all_y, all_probs),
            "pr_auc":    average_precision_score(all_y, all_probs),
        }

    summary["threshold"] = threshold
    summary["sampler"]   = sampler_name
    summary["model"]     = "Transformer"
    summary["runtime_s"] = elapsed
    return summary


def save_and_print(results, out_path, cols, title):
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\n" + "="*65)
    print(title)
    print("="*65)
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    return df


def run_creditcard_over():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    print(f"\nLoading {path} …")
    X, y = load_creditcard(path)
    print(f"  X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)  device={DEVICE}\n")

    samplers = {
        "ROS":   RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "ADASYN":ADASYN(random_state=RANDOM_STATE),
        "AGSS":  AGSS(eps=0.8, min_samples=3, n_neighbors=3, random_state=RANDOM_STATE),
    }
    results = []
    for name, sampler in samplers.items():
        print(f"\n{'='*60}\nTransformer + {name} (creditcard oversampling)")
        s = run_cv(X, y, sampler, name, CC_OVER_CFG, is_german=False, tune_threshold=False)
        results.append(s)
        print(f"  → F1={s['f1']:.4f}  Prec={s['precision']:.4f}  "
              f"Rec={s['recall']:.4f}  ROC={s['roc_auc']:.4f}  ({s['runtime_s']:.1f}s)")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "phase1_transformer_over_creditcard.csv")
    return save_and_print(results, out,
                          ["sampler", "accuracy", "precision", "recall", "f1", "roc_auc"],
                          "Transformer — CREDITCARD OVERSAMPLING (5-fold CV)")


def run_creditcard_under():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    print(f"\nLoading {path} …")
    X, y = load_creditcard(path)
    print(f"  X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)  device={DEVICE}\n")

    samplers = {
        "AGSS":       AGSSUnderSampler(eps=0.8, min_samples=3, random_state=RANDOM_STATE),
        "RUS":        RandomUnderSampler(random_state=RANDOM_STATE),
        "TomekLinks": TomekLinks(),
        "ENN":        EditedNearestNeighbours(),
        "NearMiss":   NearMiss(version=1),
    }
    results = []
    for name, sampler in samplers.items():
        print(f"\n{'='*60}\nTransformer + {name} (creditcard undersampling)")
        s = run_cv(X, y, sampler, name, CC_UNDER_CFG, is_german=False, tune_threshold=True)
        results.append(s)
        print(f"  → F1={s['f1']:.4f}  Prec={s['precision']:.4f}  "
              f"Rec={s['recall']:.4f}  ROC={s['roc_auc']:.4f}  "
              f"thresh={s['threshold']}  ({s['runtime_s']:.1f}s)")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "phase1_transformer_under_creditcard.csv")
    return save_and_print(results, out,
                          ["sampler", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"],
                          "Transformer — CREDITCARD UNDERSAMPLING (5-fold CV, tuned threshold)")


def run_german_over():
    path = os.path.join(os.path.dirname(__file__), "..", "german_credit_data.csv")
    print(f"\nLoading {path} …")
    X, y = load_german(path)
    print(f"  X={X.shape}  bad={y.sum()} ({100*y.mean():.1f}%)  device={DEVICE}\n")

    samplers = {
        "ROS":   RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "ADASYN":ADASYN(random_state=RANDOM_STATE),
        "AGSS":  AGSS(eps=0.8, min_samples=2, n_neighbors=3, random_state=RANDOM_STATE),
    }
    results = []
    for name, sampler in samplers.items():
        print(f"\n{'='*60}\nTransformer + {name} (German oversampling)")
        s = run_cv(X, y, sampler, name, DE_CFG, is_german=True, tune_threshold=True)
        results.append(s)
        print(f"  → F1_good={s['f1_good']:.4f}  F1_bad={s['f1_bad']:.4f}  "
              f"Acc={s['accuracy']:.4f}  ROC={s['roc_auc']:.4f}  ({s['runtime_s']:.1f}s)")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "phase2_transformer_over_german.csv")
    return save_and_print(results, out,
                          ["sampler", "accuracy", "f1_good", "f1_bad", "roc_auc", "threshold"],
                          "Transformer — GERMAN OVERSAMPLING (5-fold CV, tuned threshold)")


def run_german_under():
    path = os.path.join(os.path.dirname(__file__), "..", "german_credit_data.csv")
    print(f"\nLoading {path} …")
    X, y = load_german(path)
    print(f"  X={X.shape}  bad={y.sum()} ({100*y.mean():.1f}%)  device={DEVICE}\n")

    samplers = {
        "AGSS":       AGSSUnderSampler(eps=0.8, min_samples=2, random_state=RANDOM_STATE),
        "RUS":        RandomUnderSampler(random_state=RANDOM_STATE),
        "TomekLinks": TomekLinks(),
        "ENN":        EditedNearestNeighbours(),
        "NearMiss":   NearMiss(version=1),
    }
    results = []
    for name, sampler in samplers.items():
        print(f"\n{'='*60}\nTransformer + {name} (German undersampling)")
        s = run_cv(X, y, sampler, name, DE_CFG, is_german=True, tune_threshold=True)
        results.append(s)
        print(f"  → F1_good={s['f1_good']:.4f}  F1_bad={s['f1_bad']:.4f}  "
              f"Acc={s['accuracy']:.4f}  ROC={s['roc_auc']:.4f}  ({s['runtime_s']:.1f}s)")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "phase2_transformer_under_german.csv")
    return save_and_print(results, out,
                          ["sampler", "accuracy", "f1_good", "f1_bad", "roc_auc", "threshold"],
                          "Transformer — GERMAN UNDERSAMPLING (5-fold CV, tuned threshold)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default="all",
                        choices=["creditcard_over", "creditcard_under",
                                 "german_over", "german_under", "all"])
    args = parser.parse_args()

    if args.experiment in ("creditcard_over", "all"):
        run_creditcard_over()
    if args.experiment in ("creditcard_under", "all"):
        run_creditcard_under()
    if args.experiment in ("german_over", "all"):
        run_german_over()
    if args.experiment in ("german_under", "all"):
        run_german_under()
