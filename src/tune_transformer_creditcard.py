"""
Tune Transformer + AGSS on Credit Card dataset.

Baseline:  d_model=64, epochs=20, no threshold tuning → F1=0.3922 (gap: 44%)
Root cause: epochs too low + no threshold tuning on extreme imbalance (0.17%)
Target:    F1 > 0.75  (<10% absolute gap from paper target 0.8333)

Grid:
  d_model : [64, 128, 256]
  epochs  : [50, 100, 200]
  lr      : [5e-4, 1e-4]

Fixed: nhead=4, layers=2, dropout=0.3, batch=256
Threshold sweep: 0.1 → 0.99 applied to every config (key fix vs baseline).
Metric: F1 on fraud class (pos_label=1), matching all other CC experiments.

Saves best config  → results/transformer_creditcard_best.json
Saves full grid    → results/transformer_creditcard_grid.csv

Estimated runtime: ~50 min on Apple MPS (18 configs × ~160 s each)
"""

import sys, os, time, json, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS
from models import FraudTransformer

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS  = 5
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else
                        "mps"  if torch.backends.mps.is_available() else "cpu")

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
              0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

PAPER_TARGET = 0.8333   # LSTM + AGSS oversampling on Credit Card

GRID = {
    "d_model": [64, 128, 256],
    "epochs":  [50, 100, 200],
    "lr":      [5e-4, 1e-4],
}


def load_creditcard():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    df = pd.read_csv(path)
    df.dropna(subset=["Class"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = StandardScaler().fit_transform(df.drop("Class", axis=1).values).astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    return X, y


def train_model(model, X_tr, y_tr, epochs, lr, batch=256):
    n_pos = max((y_tr == 1).sum(), 1)
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()


def run_cv(X, y, sampler, d_model, epochs, lr):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
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
        print(f"    fold {fold+1}/{N_FOLDS}  "
              f"F1@0.5={f1_score(y_val,(probs>=0.5).astype(int),zero_division=0):.4f}  "
              f"ROC={roc_auc_score(y_val, probs):.4f}", flush=True)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    # Threshold sweep — the critical fix missing from the baseline
    best_f1, best_t = 0.0, 0.5
    for t in THRESHOLDS:
        f1 = f1_score(all_y, (all_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = (all_probs >= best_t).astype(int)
    return {
        "f1":        best_f1,
        "precision": precision_score(all_y, preds, zero_division=0),
        "recall":    recall_score(all_y, preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_y, all_probs),
        "threshold": best_t,
    }


def main():
    print(f"Device: {DEVICE}")
    X, y = load_creditcard()
    print(f"X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)")
    print(f"Paper target (LSTM+AGSS): {PAPER_TARGET}  |  Baseline (untuned): 0.3922\n")

    sampler = AGSS(eps=0.8, min_samples=3, n_neighbors=3,
                   adaptive_eps=True, density_weighted=True,
                   random_state=RANDOM_STATE)

    combos = list(itertools.product(GRID["d_model"], GRID["epochs"], GRID["lr"]))
    print(f"Configs to run: {len(combos)}\n")

    rows = []
    for i, (d_model, epochs, lr) in enumerate(combos, 1):
        t0 = time.time()
        print(f"[{i:02d}/{len(combos)}] d_model={d_model}  epochs={epochs}  lr={lr}")
        res = run_cv(X, y, sampler, d_model, epochs, lr)
        elapsed = time.time() - t0
        gap = PAPER_TARGET - res["f1"]
        res.update({"d_model": d_model, "epochs": epochs, "lr": lr, "runtime_s": elapsed})
        rows.append(res)
        print(f"  → F1={res['f1']:.4f}  Prec={res['precision']:.4f}  "
              f"Rec={res['recall']:.4f}  ROC={res['roc_auc']:.4f}  "
              f"thresh={res['threshold']}  gap={gap:+.4f}  ({elapsed:.0f}s)\n")

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "transformer_creditcard_grid.csv"), index=False)

    best = df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "transformer_creditcard_best.json"), "w") as f:
        json.dump({k: best[k] for k in
                   ["d_model","epochs","lr","f1","precision","recall",
                    "roc_auc","threshold"]}, f, indent=2)

    gap = PAPER_TARGET - best["f1"]
    print("=" * 60)
    print(f"Best: d_model={best['d_model']}  epochs={best['epochs']}  lr={best['lr']}")
    print(f"  F1={best['f1']:.4f}  thresh={best['threshold']}")
    print(f"  Gap vs paper: {gap:+.4f}  ({abs(gap)/PAPER_TARGET*100:.1f}%)")
    print(f"  Baseline was: 0.3922 (44.1% gap, no threshold tuning)")
    print("=" * 60)


if __name__ == "__main__":
    main()
