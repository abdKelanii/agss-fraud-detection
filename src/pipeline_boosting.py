"""
LightGBM + XGBoost pipeline with AGSS oversampling.
Runs both models on both datasets, with threshold tuning.
Saves results to:
  results/boosting_creditcard.csv
  results/boosting_german.csv
"""

import sys, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS

RANDOM_STATE = 42
N_FOLDS      = 5
THRESHOLDS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

MODELS = {
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    ),
}


def load_creditcard():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    df = pd.read_csv(path)
    y = df["Class"].values
    X = StandardScaler().fit_transform(df.drop("Class", axis=1).values).astype(np.float32)
    return X, y


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


def run_cv(X, y, model_name, model_cls, sampler, pos_label=1):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)

        # Clone model for each fold
        import copy
        m = copy.deepcopy(model_cls)
        m.fit(X_bal, y_bal)

        probs = m.predict_proba(X_val)[:, 1]
        all_probs.append(probs)
        all_y.append(y_val)
        print(f"  fold {fold+1}/{N_FOLDS} done", flush=True)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    # Threshold sweep
    best_f1, best_t = 0.0, 0.5
    for t in THRESHOLDS:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_y, preds, pos_label=pos_label, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = (all_probs >= best_t).astype(int)
    return {
        "model":     model_name,
        "f1":        round(f1_score(all_y, preds, pos_label=pos_label, zero_division=0), 4),
        "precision": round(precision_score(all_y, preds, pos_label=pos_label, zero_division=0), 4),
        "recall":    round(recall_score(all_y, preds, pos_label=pos_label, zero_division=0), 4),
        "accuracy":  round(accuracy_score(all_y, preds), 4),
        "roc_auc":   round(roc_auc_score(all_y, all_probs), 4),
        "threshold": best_t,
    }


def run_dataset(name, X, y, pos_label=1):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  X={X.shape}  minority={y.sum()} ({100*y.mean():.1f}%)")
    print(f"{'='*60}")

    agss = AGSS(eps=0.8, min_samples=2 if name == "German" else 3,
                n_neighbors=3, random_state=RANDOM_STATE)

    rows = []
    for model_name, model_cls in MODELS.items():
        t0 = time.time()
        print(f"\n[{model_name}] + AGSS oversampling …")
        res = run_cv(X, y, model_name, model_cls, agss, pos_label=pos_label)
        elapsed = time.time() - t0
        res["runtime_s"] = round(elapsed, 1)
        rows.append(res)
        print(f"  → F1={res['f1']:.4f}  Prec={res['precision']:.4f}  "
              f"Rec={res['recall']:.4f}  ROC={res['roc_auc']:.4f}  "
              f"thresh={res['threshold']}  ({elapsed:.0f}s)")

    return pd.DataFrame(rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Credit card
    X, y = load_creditcard()
    df_cc = run_dataset("CreditCard", X, y, pos_label=1)
    df_cc.to_csv(os.path.join(out_dir, "boosting_creditcard.csv"), index=False)

    # German (majority class F1 = pos_label=0)
    X, y = load_german()
    df_de = run_dataset("German", X, y, pos_label=0)
    df_de.to_csv(os.path.join(out_dir, "boosting_german.csv"), index=False)

    print("\n" + "="*60)
    print("CREDIT CARD RESULTS (F1 = fraud class)")
    print("="*60)
    print(df_cc[["model","f1","precision","recall","roc_auc","threshold"]].to_string(index=False))

    print("\n" + "="*60)
    print("GERMAN RESULTS (F1 = good credit class, pos_label=0)")
    print("="*60)
    print(df_de[["model","f1","precision","recall","roc_auc","threshold"]].to_string(index=False))

    print("\nSaved to results/boosting_creditcard.csv and results/boosting_german.csv")


if __name__ == "__main__":
    main()
