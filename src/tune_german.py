"""
Hyperparameter grid search for German credit-risk dataset (9-feature Kaggle version).
Threshold sweep applied post-hoc (no retraining needed).
Saves best config to results/german_best_config.json.
"""

import sys, os, json, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS
from models import FraudLSTM, FraudRNN

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
N_FOLDS  = 5
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

GRID = {
    "hidden_size": [32, 64],
    "epochs":      [100, 200],
    "lr":          [1e-3, 5e-4],
    "model":       ["LSTM", "RNN"],
    "sampler":     ["AGSS", "SMOTE"],
}
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Target metric: F1 for class 0 (good credit / majority) — matches the paper's convention


def load_german(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    y = (df["Risk"] == "bad").astype(int).values
    df = df.drop("Risk", axis=1)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.mean(numeric_only=True))
    X = StandardScaler().fit_transform(df.values).astype(np.float32)
    return X, y


def make_model(model_name, n_features, hidden_size):
    cls = FraudLSTM if model_name == "LSTM" else FraudRNN
    return cls(n_features=n_features, hidden_size=hidden_size, num_layers=2, dropout=0.3).to(DEVICE)


def make_sampler(name):
    if name == "AGSS":
        return AGSS(eps=0.8, min_samples=2, n_neighbors=3, random_state=RANDOM_STATE)
    return SMOTE(random_state=RANDOM_STATE)


def train_fold(model, X_tr, y_tr, lr, epochs, batch_size=32):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()


def run_config(X, y, hidden_size, epochs, lr, model_name, sampler_name):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_probs, fold_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_tr_bal, y_tr_bal = make_sampler(sampler_name).fit_resample(X_tr, y_tr)
        model = make_model(model_name, X.shape[1], hidden_size)
        train_fold(model, X_tr_bal, y_tr_bal.astype(np.int64), lr, epochs)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
        fold_probs.append(torch.sigmoid(torch.tensor(logits)).numpy())
        fold_y.append(y_val)

    all_probs = np.concatenate(fold_probs)
    all_y     = np.concatenate(fold_y)
    # Optimise for F1 of class 0 (good credit = majority), matching the paper's metric
    threshold_f1 = {t: f1_score(all_y, (all_probs >= t).astype(int),
                                pos_label=0, zero_division=0)
                    for t in THRESHOLDS}
    best_thresh = max(threshold_f1, key=threshold_f1.get)
    return threshold_f1[best_thresh], best_thresh, roc_auc_score(all_y, all_probs), threshold_f1


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "german_credit_data.csv")
    X, y = load_german(path)
    print(f"Loaded: X={X.shape}, bad credit={y.sum()} ({100*y.mean():.1f}%)")
    print(f"Device: {DEVICE}")

    combos = list(itertools.product(*[GRID[k] for k in GRID]))
    print(f"Total configs: {len(combos)}\n")

    rows = []
    for i, vals in enumerate(combos):
        cfg = dict(zip(GRID.keys(), vals))
        print(f"[{i+1}/{len(combos)}] {cfg} … ", end="", flush=True)
        best_f1, best_thresh, roc, thresh_f1 = run_config(
            X, y,
            hidden_size=cfg["hidden_size"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            model_name=cfg["model"],
            sampler_name=cfg["sampler"],
        )
        print(f"F1={best_f1:.4f} @ thresh={best_thresh}  ROC-AUC={roc:.4f}")
        rows.append({**cfg, "best_f1": best_f1, "best_threshold": best_thresh,
                     "roc_auc": roc, **{f"f1_t{t}": thresh_f1[t] for t in THRESHOLDS}})

    df = pd.DataFrame(rows).sort_values("best_f1", ascending=False)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "german_tuning_grid.csv"), index=False)

    best = df.iloc[0].to_dict()
    best_cfg = {k: best[k] for k in list(GRID.keys()) + ["best_f1", "best_threshold", "roc_auc"]}
    with open(os.path.join(out_dir, "german_best_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS")
    print("="*60)
    print(df[list(GRID.keys()) + ["best_f1", "best_threshold", "roc_auc"]].head(5).to_string(index=False))
    print(f"\nBest config saved to results/german_best_config.json")


if __name__ == "__main__":
    main()
