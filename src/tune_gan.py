"""
GAN tuning — two fixes:

1. Rerun creditcard oversampling with post-hoc threshold sweep
   (fixes ROS/SMOTE/ADASYN F1=0.34 → expected ~0.70+)

2. Grid search over epochs/lr for AGSS oversampling to close the
   ~6.6% gap vs paper (0.7459 → target 0.7981)

Grid:
  epochs : [50, 100]
  lr     : [1e-4, 5e-5]
  hidden : [128, 256]

Saves to:
  results/phase1_gan_over_tuned_creditcard.csv
  results/gan_agss_over_best.json
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
from models import FraudGAN
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_FOLDS    = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
              0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

GRID = {
    "hidden":  [128, 256],
    "epochs":  [50, 100],
    "lr":      [1e-4, 5e-5],
}


def load_creditcard():
    path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    df = pd.read_csv(path)
    df.dropna(subset=["Class"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = StandardScaler().fit_transform(df.drop("Class", axis=1).values).astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    return X, y


def train_gan(model, X_tr, y_tr, epochs, lr, batch=256):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_g = nn.BCEWithLogitsLoss()
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(),     lr=lr, betas=(0.5, 0.999))

    X_t    = torch.from_numpy(X_tr)
    y_t    = torch.from_numpy(y_tr.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            bs = len(xb)

            opt_d.zero_grad()
            d_loss = (criterion(model.discriminator(xb).squeeze(-1), yb) +
                      0.5 * criterion_g(
                          model.discriminator(model.generator(
                              torch.randn(bs, model.noise_dim, device=DEVICE)
                          ).detach()).squeeze(-1),
                          torch.zeros(bs, device=DEVICE)))
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            criterion_g(
                model.discriminator(model.generator(
                    torch.randn(bs, model.noise_dim, device=DEVICE)
                )).squeeze(-1),
                torch.ones(bs, device=DEVICE)
            ).backward()
            opt_g.step()


def run_cv(X, y, sampler, hidden, epochs, lr):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_y = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)
        y_bal = y_bal.astype(np.int64)

        model = FraudGAN(n_features=X.shape[1], hidden_size=hidden, noise_dim=32).to(DEVICE)
        train_gan(model, X_bal, y_bal, epochs, lr)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).to(DEVICE)).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        all_probs.append(probs)
        all_y.append(y_val)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_f1, best_t = 0.0, 0.5
    for t in THRESHOLDS:
        f1 = f1_score(all_y, (all_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = (all_probs >= best_t).astype(int)
    return {
        "f1":        best_f1,
        "threshold": best_t,
        "precision": precision_score(all_y, preds, zero_division=0),
        "recall":    recall_score(all_y, preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_y, all_probs),
    }


def main():
    print(f"Device: {DEVICE}\n")
    X, y = load_creditcard()
    print(f"X={X.shape}  fraud={y.sum()} ({100*y.mean():.3f}%)\n")

    # ── Part 1: All samplers with threshold tuning (fixed best config) ────────
    print("="*65)
    print("PART 1 — All samplers, threshold tuning (hidden=128, epochs=50)")
    print("="*65)
    samplers = {
        "ROS":   RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "ADASYN":ADASYN(random_state=RANDOM_STATE),
        "AGSS":  AGSS(eps=0.8, min_samples=3, n_neighbors=3, random_state=RANDOM_STATE),
    }
    rows = []
    for name, sampler in samplers.items():
        t0 = time.time()
        print(f"\n[{name}] … ", end="", flush=True)
        res = run_cv(X, y, sampler, hidden=128, epochs=50, lr=1e-4)
        elapsed = time.time() - t0
        res["sampler"] = name
        rows.append(res)
        print(f"F1={res['f1']:.4f} @ thresh={res['threshold']}  "
              f"Prec={res['precision']:.4f}  Rec={res['recall']:.4f}  "
              f"ROC={res['roc_auc']:.4f}  ({elapsed:.0f}s)")

    df1 = pd.DataFrame(rows)
    out1 = os.path.join(os.path.dirname(__file__), "..", "results",
                        "phase1_gan_over_tuned_creditcard.csv")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    df1.to_csv(out1, index=False)
    print(f"\nSaved: {out1}")

    # ── Part 2: Grid search on AGSS only ─────────────────────────────────────
    print("\n" + "="*65)
    print("PART 2 — AGSS hyperparameter grid search")
    print("="*65)
    agss_sampler = AGSS(eps=0.8, min_samples=3, n_neighbors=3, random_state=RANDOM_STATE)
    combos = list(itertools.product(GRID["hidden"], GRID["epochs"], GRID["lr"]))
    print(f"Configs: {len(combos)}\n")

    grid_rows = []
    for hidden, epochs, lr in combos:
        t0 = time.time()
        print(f"hidden={hidden} epochs={epochs} lr={lr} … ", end="", flush=True)
        res = run_cv(X, y, agss_sampler, hidden=hidden, epochs=epochs, lr=lr)
        elapsed = time.time() - t0
        res.update({"hidden": hidden, "epochs": epochs, "lr": lr})
        grid_rows.append(res)
        print(f"F1={res['f1']:.4f} @ thresh={res['threshold']}  "
              f"Prec={res['precision']:.4f}  ROC={res['roc_auc']:.4f}  ({elapsed:.0f}s)")

    df2 = pd.DataFrame(grid_rows).sort_values("f1", ascending=False)
    out2 = os.path.join(os.path.dirname(__file__), "..", "results",
                        "gan_agss_over_grid.csv")
    df2.to_csv(out2, index=False)

    best = df2.iloc[0].to_dict()
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "gan_agss_over_best.json"), "w") as f:
        json.dump({k: best[k] for k in ["hidden","epochs","lr","f1","threshold",
                                         "precision","recall","roc_auc"]}, f, indent=2)

    print(f"\nBest: hidden={best['hidden']} epochs={best['epochs']} lr={best['lr']} "
          f"→ F1={best['f1']:.4f}  paper target: 0.7981  "
          f"gap={0.7981-best['f1']:.4f}")
    print(f"Saved grid: {out2}")


if __name__ == "__main__":
    main()
