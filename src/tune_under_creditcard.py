"""
Threshold + ratio tuning for creditcard undersampling.

Two problems with the baseline:
  1. AGSS/RUS use 1:1 ratio → only ~984 training samples → precision collapses on test
  2. All methods use threshold=0.5 which is wrong when train/test distributions differ

Fix:
  - Sweep sampling ratios for AGSS & RUS: {majority : minority} = 1,10,50,100
  - Sweep thresholds 0.5→0.99 post-hoc (no retraining) for all methods
  - TomekLinks/ENN are cleaning methods (barely change ratio), so only threshold sweep applies

Saves best config per method to results/creditcard_under_best_configs.json
"""

import sys, os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss,
)

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSSUnderSampler
from models import FraudLSTM

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS    = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
RATIOS     = [1, 10, 50, 100]   # majority : minority

CFG = dict(hidden=64, layers=2, dropout=0.3, lr=1e-3, epochs=20, batch=256)


def load_creditcard(path):
    df = pd.read_csv(path)
    df.dropna(subset=["Class"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = StandardScaler().fit_transform(df.drop("Class", axis=1).values).astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    return X, y


def make_sampler(name, ratio):
    n_minority_approx = 394   # ~minority per training fold (492 * 0.8)
    n_majority_target = n_minority_approx * ratio
    strategy = min(n_majority_target / (394 * (ratio + 1) + n_majority_target), 0.9999)

    if name == "AGSS":
        s = AGSSUnderSampler(eps=0.8, min_samples=3, random_state=RANDOM_STATE)
        # Override to target the desired ratio using sampling_strategy
        s._ratio = ratio
        s._orig_fit_resample = s.fit_resample

        def fit_resample_ratio(X, y, _s=s, _ratio=ratio):
            n_min = int((y == 1).sum())
            n_target = min(n_min * _ratio, int((y == 0).sum()))
            rng = np.random.default_rng(RANDOM_STATE)
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            X_maj, X_min = X[y == 0], X[y == 1]
            labels = DBSCAN(eps=_s.eps, min_samples=_s.min_samples).fit_predict(X_maj)
            if len(set(labels) - {-1}) < 1:
                k = min(_s.min_samples, len(X_maj) - 1)
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_maj)
                dists, _ = nbrs.kneighbors(X_maj)
                eps2 = float(np.percentile(dists[:, k], 25))
                labels = DBSCAN(eps=eps2, min_samples=_s.min_samples).fit_predict(X_maj)
            X_clean = X_maj[labels != -1]
            if len(X_clean) == 0:
                X_clean = X_maj
            if len(X_clean) > n_target:
                idx = rng.choice(len(X_clean), size=n_target, replace=False)
                X_clean = X_clean[idx]
            X_out = np.vstack([X_clean, X_min]).astype(X.dtype)
            y_out = np.concatenate([np.zeros(len(X_clean), dtype=y.dtype),
                                    np.ones(len(X_min), dtype=y.dtype)])
            return X_out, y_out

        s.fit_resample = fit_resample_ratio
        return s

    if name == "RUS":
        n_min = n_minority_approx
        n_maj = min(n_min * ratio, 227451)
        return RandomUnderSampler(
            sampling_strategy={0: n_maj, 1: n_min},
            random_state=RANDOM_STATE,
        )
    if name == "TomekLinks":
        return TomekLinks()
    if name == "ENN":
        return EditedNearestNeighbours()
    if name == "NearMiss":
        return NearMiss(version=1)


def train_fold(X_tr, y_tr):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    model = FraudLSTM(n_features=X_tr.shape[1], hidden_size=CFG["hidden"],
                      num_layers=CFG["layers"], dropout=CFG["dropout"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    X_t = torch.from_numpy(X_tr).unsqueeze(1)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=CFG["batch"], shuffle=True)
    model.train()
    for _ in range(CFG["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model


def run_config(X, y, sampler_name, ratio):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        sampler = make_sampler(sampler_name, ratio)
        try:
            X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        except Exception as e:
            print(f"  Sampler error ({sampler_name}, ratio={ratio}): {e}")
            return None

        model = train_fold(X_bal, y_bal.astype(np.int64))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).unsqueeze(1).to(DEVICE)).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        all_probs.append(probs)
        all_y.append(y_val)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_f1, best_thresh = 0, 0.5
    thresh_results = {}
    for t in THRESHOLDS:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_y, preds, zero_division=0)
        thresh_results[t] = {
            "f1":        f1,
            "precision": precision_score(all_y, preds, zero_division=0),
            "recall":    recall_score(all_y, preds, zero_division=0),
        }
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    roc = roc_auc_score(all_y, all_probs)
    return {
        "sampler": sampler_name,
        "ratio":   ratio,
        "best_f1": best_f1,
        "best_threshold": best_thresh,
        "roc_auc": roc,
        **{f"f1_t{t}": thresh_results[t]["f1"] for t in THRESHOLDS},
        **{f"prec_t{t}": thresh_results[t]["precision"] for t in THRESHOLDS},
        **{f"rec_t{t}":  thresh_results[t]["recall"]    for t in THRESHOLDS},
    }


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    print(f"Loading {path} …")
    X, y = load_creditcard(path)
    print(f"  X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)  device={DEVICE}\n")

    # Methods that benefit from ratio sweep
    ratio_methods  = ["AGSS", "RUS"]
    # Cleaning methods — ratio doesn't apply, only threshold sweep
    clean_methods  = ["TomekLinks", "ENN", "NearMiss"]

    rows = []
    total = len(ratio_methods) * len(RATIOS) + len(clean_methods)
    done  = 0

    for name in ratio_methods:
        for ratio in RATIOS:
            done += 1
            print(f"[{done}/{total}] {name}  ratio={ratio}:1 … ", end="", flush=True)
            t0  = time.time()
            res = run_config(X, y, name, ratio)
            if res:
                rows.append(res)
                print(f"best F1={res['best_f1']:.4f} @ thresh={res['best_threshold']}  "
                      f"ROC-AUC={res['roc_auc']:.4f}  ({time.time()-t0:.1f}s)")

    for name in clean_methods:
        done += 1
        print(f"[{done}/{total}] {name}  (cleaning, ratio=N/A) … ", end="", flush=True)
        t0  = time.time()
        res = run_config(X, y, name, ratio=1)   # ratio ignored for cleaning methods
        if res:
            res["ratio"] = "N/A"
            rows.append(res)
            print(f"best F1={res['best_f1']:.4f} @ thresh={res['best_threshold']}  "
                  f"ROC-AUC={res['roc_auc']:.4f}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows).sort_values("best_f1", ascending=False)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "creditcard_under_tuning.csv"), index=False)

    # Save best config per sampler
    best_per = {}
    for name in ratio_methods + clean_methods:
        sub = df[df["sampler"] == name]
        if not sub.empty:
            best_per[name] = sub.iloc[0][["sampler","ratio","best_f1","best_threshold","roc_auc"]].to_dict()

    with open(os.path.join(out_dir, "creditcard_under_best_configs.json"), "w") as f:
        json.dump(best_per, f, indent=2)

    print("\n" + "="*65)
    print("TOP RESULTS — Creditcard Undersampling (tuned threshold + ratio)")
    print("="*65)
    cols = ["sampler", "ratio", "best_f1", "best_threshold", "roc_auc"]
    print(df[cols].head(10).to_string(index=False, float_format="{:.4f}".format))
    print(f"\nBest configs saved to results/creditcard_under_best_configs.json")


if __name__ == "__main__":
    main()
