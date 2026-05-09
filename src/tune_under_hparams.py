"""
Hyperparameter tuning for creditcard AGSS undersampling.

Now that AGSSUnderSampler pre-subsamples before DBSCAN (0.2s per fold vs 2hrs),
we can do a proper grid search over model hyperparameters.

Grid:
  hidden_size : [64, 128]
  epochs      : [20, 30, 50]
  lr          : [1e-3, 5e-4, 1e-4]
  threshold   : [0.93, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]  (post-hoc)

Saves best config to results/creditcard_under_hparams_best.json
"""

import sys, os, json, time, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSSUnderSampler
from models import FraudLSTM, FraudRNN

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS    = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
THRESHOLDS = [0.93, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]

GRID = {
    "hidden_size": [64, 128],
    "epochs":      [20, 30, 50],
    "lr":          [1e-3, 5e-4, 1e-4],
    "model":       ["LSTM", "RNN"],
}


def load_creditcard(path):
    df = pd.read_csv(path)
    df.dropna(subset=["Class"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = StandardScaler().fit_transform(df.drop("Class", axis=1).values).astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    return X, y


def run_config(X, y, hidden_size, epochs, lr, model_name):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sampler = AGSSUnderSampler(eps=0.8, min_samples=3, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        y_bal = y_bal.astype(np.int64)

        n_pos = (y_bal == 1).sum()
        n_neg = (y_bal == 0).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

        cls = FraudRNN if model_name == "RNN" else FraudLSTM
        model = cls(n_features=X.shape[1], hidden_size=hidden_size,
                    num_layers=2, dropout=0.3).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_t = torch.from_numpy(X_bal).unsqueeze(1)
        y_t = torch.from_numpy(y_bal.astype(np.float32))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        all_probs.append(probs)
        all_y.append(y_val)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_f1, best_thresh = 0.0, 0.97
    thresh_f1 = {}
    for t in THRESHOLDS:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_y, preds, zero_division=0)
        thresh_f1[t] = f1
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds_best = (all_probs >= best_thresh).astype(int)
    return {
        "best_f1":        best_f1,
        "best_threshold": best_thresh,
        "precision":      precision_score(all_y, preds_best, zero_division=0),
        "recall":         recall_score(all_y, preds_best, zero_division=0),
        "roc_auc":        roc_auc_score(all_y, all_probs),
        **{f"f1_t{t}": thresh_f1[t] for t in THRESHOLDS},
    }


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    print(f"Loading {path} …")
    X, y = load_creditcard(path)
    print(f"  X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)  device={DEVICE}\n")

    combos = list(itertools.product(*[GRID[k] for k in GRID]))
    print(f"Total configs: {len(combos)}  (threshold swept post-hoc)\n")

    rows = []
    for i, vals in enumerate(combos):
        cfg = dict(zip(GRID.keys(), vals))
        print(f"[{i+1}/{len(combos)}] hidden={cfg['hidden_size']} epochs={cfg['epochs']} "
              f"lr={cfg['lr']} model={cfg['model']} … ", end="", flush=True)
        t0 = time.time()
        res = run_config(X, y,
                         hidden_size=cfg["hidden_size"],
                         epochs=cfg["epochs"],
                         lr=cfg["lr"],
                         model_name=cfg["model"])
        elapsed = time.time() - t0
        res.update(cfg)
        rows.append(res)
        print(f"F1={res['best_f1']:.4f} @ thresh={res['best_threshold']}  "
              f"prec={res['precision']:.4f}  rec={res['recall']:.4f}  "
              f"ROC={res['roc_auc']:.4f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(rows).sort_values("best_f1", ascending=False)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "creditcard_under_hparams.csv"), index=False)

    best = df.iloc[0].to_dict()
    best_cfg = {k: best[k] for k in list(GRID.keys()) + ["best_f1", "best_threshold",
                                                           "precision", "recall", "roc_auc"]}
    with open(os.path.join(out_dir, "creditcard_under_hparams_best.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    print("\n" + "="*70)
    print("TOP 5 CONFIGS — AGSS Undersampling (creditcard, 5-fold CV)")
    print("="*70)
    cols = ["model", "hidden_size", "epochs", "lr", "best_f1", "best_threshold", "roc_auc"]
    print(df[cols].head(5).to_string(index=False, float_format="{:.4f}".format))
    print(f"\nBest config: F1={best_cfg['best_f1']:.4f}  paper target: 0.8636")
    print(f"Gap from paper: {0.8636 - best_cfg['best_f1']:.4f}")
    print(f"\nSaved to results/creditcard_under_hparams_best.json")


if __name__ == "__main__":
    main()
