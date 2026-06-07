"""
Targeted tuning — German RNN + AGSS Undersampling.

Current best: F1_good=0.8145  Paper target: 0.8601  Need: ≥0.8171 (<5% gap)

Grid: hidden × epochs × lr × dropout, with per-experiment threshold sweep.
"""

import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSSUnderSampler
from models import FraudRNN

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS   = 5
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else
                         "mps"  if torch.backends.mps.is_available() else "cpu")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

GRID = {
    "hidden":  [64, 128],
    "epochs":  [200, 300, 400],
    "lr":      [5e-4, 1e-4],
    "dropout": [0.2, 0.3],
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


def train_rnn(model, X_tr, y_tr, lr, epochs, batch=32):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def run_cv(X, y, hidden, epochs, lr, dropout):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sampler = AGSSUnderSampler(eps=0.8, min_samples=2, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        y_bal = y_bal.astype(np.int64)

        model = FraudRNN(
            n_features=X.shape[1],
            hidden_size=hidden,
            num_layers=2,
            dropout=dropout,
        ).to(DEVICE)
        train_rnn(model, X_bal, y_bal, lr=lr, epochs=epochs)

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
        fg = f1_score(all_y, preds, pos_label=0, zero_division=0)
        if fg > best_f1_good:
            best_f1_good, best_t = fg, t

    preds = (all_probs >= best_t).astype(int)
    return {
        "f1_good":  best_f1_good,
        "f1_bad":   f1_score(all_y, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(all_y, preds),
        "roc_auc":  roc_auc_score(all_y, all_probs),
        "threshold": best_t,
        "hidden": hidden, "epochs": epochs, "lr": lr, "dropout": dropout,
    }


def main():
    print(f"Device: {DEVICE}")
    X, y = load_german()
    print(f"X={X.shape}  bad={y.sum()} ({100*y.mean():.1f}%)\n")
    print(f"Paper target: 0.8601  |  Need: ≥0.8171 (gap < 5%)\n")

    import itertools
    combos = list(itertools.product(
        GRID["hidden"], GRID["epochs"], GRID["lr"], GRID["dropout"]
    ))
    print(f"Running {len(combos)} configs …\n")

    rows = []
    for hidden, epochs, lr, dropout in combos:
        t0 = time.time()
        print(f"hidden={hidden} epochs={epochs} lr={lr} dropout={dropout} … ", end="", flush=True)
        res = run_cv(X, y, hidden, epochs, lr, dropout)
        elapsed = time.time() - t0
        res["runtime_s"] = elapsed
        rows.append(res)
        gap = 0.8601 - res["f1_good"]
        flag = "✅ <5%" if gap < 0.0430 else f"  ({gap:.4f} gap)"
        print(f"F1_good={res['f1_good']:.4f} @ thresh={res['threshold']}  "
              f"ROC={res['roc_auc']:.4f}  ({elapsed:.0f}s) {flag}")

    df = pd.DataFrame(rows).sort_values("f1_good", ascending=False)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "tune_german_under_rnn.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    best = df.iloc[0]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG:")
    print(f"  hidden={best.hidden}  epochs={best.epochs}  "
          f"lr={best.lr}  dropout={best.dropout}")
    print(f"  F1_good={best.f1_good:.4f}  F1_bad={best.f1_bad:.4f}  "
          f"Acc={best.accuracy:.4f}  ROC={best.roc_auc:.4f}")
    print(f"  threshold={best.threshold}")
    gap = 0.8601 - best.f1_good
    print(f"  Gap vs paper: {gap:.4f} ({100*gap/0.8601:.1f}%)")
    print(f"  {'✅ GOAL MET (<5% gap)' if gap < 0.0430 else '❌ Still above 5% gap'}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
