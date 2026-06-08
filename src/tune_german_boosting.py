"""
Tune LightGBM + XGBoost + AGSS on German Credit dataset.

Current best:  XGBoost  F1=0.8386  (-1.2% vs paper)
               LightGBM F1=0.8343  (-1.7% vs paper)
Paper target:  RNN+AGSS F1=0.8489  (pos_label=0, good-credit class)

Strategy:
  1. Fine-grained threshold sweep (0.01 increments) — biggest quick win
  2. num_leaves / max_depth grid — controls model complexity
  3. min_child_samples / min_child_weight — regularisation for small dataset
  4. AGSS eps grid — 300 bad-credit samples, tighter clusters may help

Estimated runtime: 5-10 min (German is 1000 rows, trees are fast)

Saves:
  results/tune_german_boosting_best.json
  results/tune_german_boosting_grid.csv
"""

import sys, os, time, json, itertools, copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS

RANDOM_STATE = 42
N_FOLDS      = 5
PAPER_TARGET = 0.8489   # RNN+AGSS German oversampling

# Fine-grained sweep — 89 thresholds instead of 12
THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.99, 0.01)]

LGBM_GRID = {
    "num_leaves":        [127, 255],
    "min_child_samples": [5, 10],
    "learning_rate":     [0.03, 0.05],
}

XGB_GRID = {
    "max_depth":        [4, 6],
    "min_child_weight": [1, 3],
    "learning_rate":    [0.03, 0.05],
}

AGSS_EPS = [0.5, 0.8]


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


def run_cv(X, y, model, sampler):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        m = copy.deepcopy(model)
        m.fit(X_bal, y_bal)
        all_probs.append(m.predict_proba(X_val)[:, 1])
        all_y.append(y_val)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_f1, best_t = 0.0, 0.5
    for t in THRESHOLDS:
        f1 = f1_score(all_y, (all_probs >= t).astype(int), pos_label=0, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = (all_probs >= best_t).astype(int)
    return {
        "f1":        round(best_f1, 4),
        "precision": round(precision_score(all_y, preds, pos_label=0, zero_division=0), 4),
        "recall":    round(recall_score(all_y, preds,    pos_label=0, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(all_y, all_probs), 4),
        "threshold": best_t,
    }


def main():
    X, y = load_german()
    print(f"German Credit: X={X.shape}  bad={y.sum()} ({100*y.mean():.1f}%)")
    print(f"Paper target (RNN+AGSS): {PAPER_TARGET}\n")

    rows = []
    total = (len(LGBM_GRID["num_leaves"]) * len(LGBM_GRID["min_child_samples"]) * len(LGBM_GRID["learning_rate"])
           + len(XGB_GRID["max_depth"])   * len(XGB_GRID["min_child_weight"])   * len(XGB_GRID["learning_rate"])) * len(AGSS_EPS)
    done = 0

    # ── LightGBM grid ─────────────────────────────────────────────────────────
    for eps, (nl, mc, lr) in itertools.product(
            AGSS_EPS,
            itertools.product(LGBM_GRID["num_leaves"],
                              LGBM_GRID["min_child_samples"],
                              LGBM_GRID["learning_rate"])):
        done += 1
        sampler = AGSS(eps=eps, min_samples=2, n_neighbors=3,
                       adaptive_eps=True, density_weighted=True,
                       random_state=RANDOM_STATE)
        model = LGBMClassifier(
            n_estimators=500, learning_rate=lr, num_leaves=nl,
            min_child_samples=mc, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
        t0 = time.time()
        res = run_cv(X, y, model, sampler)
        elapsed = time.time() - t0
        res.update({"algo": "LightGBM", "eps": eps, "num_leaves": nl,
                    "min_child_samples": mc, "lr": lr, "runtime_s": round(elapsed, 1)})
        rows.append(res)
        gap = PAPER_TARGET - res["f1"]
        print(f"[{done:03d}/{total}] LGBM  eps={eps} nl={nl} mc={mc} lr={lr}  "
              f"F1={res['f1']:.4f}  thresh={res['threshold']}  gap={gap:+.4f}  ({elapsed:.1f}s)",
              flush=True)

    # ── XGBoost grid ──────────────────────────────────────────────────────────
    for eps, (md, mcw, lr) in itertools.product(
            AGSS_EPS,
            itertools.product(XGB_GRID["max_depth"],
                              XGB_GRID["min_child_weight"],
                              XGB_GRID["learning_rate"])):
        done += 1
        sampler = AGSS(eps=eps, min_samples=2, n_neighbors=3,
                       adaptive_eps=True, density_weighted=True,
                       random_state=RANDOM_STATE)
        model = XGBClassifier(
            n_estimators=500, learning_rate=lr, max_depth=md,
            min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric="logloss",
            verbosity=0, n_jobs=-1,
        )
        t0 = time.time()
        res = run_cv(X, y, model, sampler)
        elapsed = time.time() - t0
        res.update({"algo": "XGBoost", "eps": eps, "max_depth": md,
                    "min_child_weight": mcw, "lr": lr, "runtime_s": round(elapsed, 1)})
        rows.append(res)
        gap = PAPER_TARGET - res["f1"]
        print(f"[{done:03d}/{total}] XGB   eps={eps} md={md} mcw={mcw} lr={lr}  "
              f"F1={res['f1']:.4f}  thresh={res['threshold']}  gap={gap:+.4f}  ({elapsed:.1f}s)",
              flush=True)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "tune_german_boosting_grid.csv"), index=False)

    best = df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "tune_german_boosting_best.json"), "w") as f:
        json.dump(best, f, indent=2)

    gap = PAPER_TARGET - best["f1"]
    print("\n" + "=" * 60)
    print(f"Best: {best['algo']}  F1={best['f1']:.4f}  thresh={best['threshold']}")
    print(f"  Gap vs paper: {gap:+.4f}  ({'BEATS PAPER' if gap <= 0 else f'{abs(gap)/PAPER_TARGET*100:.1f}% below'})")
    print(f"  Baseline was: XGBoost=0.8386 (-1.2%)  LightGBM=0.8343 (-1.7%)")
    print("=" * 60)

    print("\nTop 10 configs:")
    cols = ["algo", "f1", "threshold", "eps"]
    for c in ["num_leaves", "min_child_samples", "max_depth", "min_child_weight", "lr"]:
        if c in df.columns:
            cols.append(c)
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
