"""
Tune Transformer on German dataset — close the 14.7% gap vs paper (0.8489).
Current config: d_model=32 → too small for 20-feature German data.

Grid:
  d_model : [64, 128, 256]
  epochs  : [300, 500]
  lr      : [5e-4, 2e-4]

All other settings fixed: nhead=4, layers=2, dropout=0.3, batch=32
Saves best config to results/transformer_german_best.json
Saves full grid to results/transformer_german_grid.csv
"""

import sys, os, time, json, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS
from models import FraudTransformer

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS    = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

GRID = {
    "d_model": [64, 128, 256],
    "epochs":  [300, 500],
    "lr":      [5e-4, 2e-4],
}


def load_german():
    path = os.path.join(os.path.dirname(__file__), "..", "german_credit_data.csv")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    y = (df["Risk"] == "bad").astype(int).values
    df = df.drop("Risk", axis=1)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.mean(numeric_only=True))
    X = StandardScaler().fit_transform(df.values).astype(np.float32)
    return X, y


def train_model(model, X_tr, y_tr, epochs, lr, batch=32):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss(pos_weight=pos_weight)(model(xb), yb).backward()
            optimizer.step()


def run_cv(X, y, sampler, d_model, epochs, lr):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        y_bal = y_bal.astype(np.int64)

        model = FraudTransformer(
            n_features=X.shape[1], d_model=d_model, nhead=4,
            num_layers=2, dropout=0.3,
        ).to(DEVICE)
        train_model(model, X_bal, y_bal, epochs, lr)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        all_probs.append(probs)
        all_y.append(y_val)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_f1_good, best_t = 0.0, 0.5
    for t in THRESHOLDS:
        preds = (all_probs >= t).astype(int)
        f1g = f1_score(all_y, preds, pos_label=0, zero_division=0)
        if f1g > best_f1_good:
            best_f1_good, best_t = f1g, t

    preds = (all_probs >= best_t).astype(int)
    return {
        "f1_good":  best_f1_good,
        "f1_bad":   f1_score(all_y, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(all_y, preds),
        "roc_auc":  roc_auc_score(all_y, all_probs),
        "threshold": best_t,
    }


def main():
    print(f"Device: {DEVICE}")
    X, y = load_german()
    print(f"X={X.shape}  bad={y.sum()} ({100*y.mean():.1f}%)\n")

    sampler = AGSS(eps=0.8, min_samples=2, n_neighbors=3, random_state=RANDOM_STATE)
    combos  = list(itertools.product(GRID["d_model"], GRID["epochs"], GRID["lr"]))
    print(f"Configs: {len(combos)}  (current baseline: d_model=32, F1_good≈0.702)\n")

    rows = []
    for d_model, epochs, lr in combos:
        t0 = time.time()
        print(f"d_model={d_model} epochs={epochs} lr={lr} … ", end="", flush=True)
        res = run_cv(X, y, sampler, d_model, epochs, lr)
        elapsed = time.time() - t0
        res.update({"d_model": d_model, "epochs": epochs, "lr": lr})
        rows.append(res)
        print(f"F1_good={res['f1_good']:.4f}  F1_bad={res['f1_bad']:.4f}  "
              f"Acc={res['accuracy']:.4f}  ROC={res['roc_auc']:.4f}  "
              f"thresh={res['threshold']}  ({elapsed:.0f}s)")

    df = pd.DataFrame(rows).sort_values("f1_good", ascending=False)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "transformer_german_grid.csv"), index=False)

    best = df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "transformer_german_best.json"), "w") as f:
        json.dump({k: best[k] for k in
                   ["d_model","epochs","lr","f1_good","f1_bad","accuracy","roc_auc","threshold"]}, f, indent=2)

    print(f"\nBest: d_model={best['d_model']} epochs={best['epochs']} lr={best['lr']}"
          f"  →  F1_good={best['f1_good']:.4f}  (paper target: 0.8489, baseline: 0.7021)")
    print(f"Gap vs paper: {0.8489 - best['f1_good']:.4f}")


if __name__ == "__main__":
    main()
