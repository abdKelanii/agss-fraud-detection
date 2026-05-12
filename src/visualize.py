"""
Generate Phase 1 & 2 visualizations from saved results.
Produces:
  figures/fig_metrics_bar.png          — Phase 1 LSTM grouped bar chart
  figures/fig_f1_heatmap.png           — Phase 1 LSTM F1 heatmap
  figures/fig_precision_recall.png     — Phase 1 LSTM Precision vs Recall scatter
  figures/fig_phase1_lstm_rnn_f1.png  — Phase 1 LSTM vs RNN F1 comparison
  figures/fig_german_f1_bar.png        — Phase 2 F1 bar chart (good-class F1 per model+sampler)
  figures/fig_german_f1_heatmap.png    — Phase 2 F1 heatmap (model × sampler)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
CSV_PATH    = os.path.join(RESULTS_DIR, "phase1_lstm_creditcard.csv")

COLORS = {
    "ROS":    "#5B9BD5",
    "SMOTE":  "#ED7D31",
    "ADASYN": "#A9D18E",
    "AGSS":   "#FF0000",
}

def load():
    df = pd.read_csv(CSV_PATH, index_col="sampler")
    return df


def fig_metrics_bar(df):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    samplers = df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-(len(samplers)-1)/2, (len(samplers)-1)/2, len(samplers)) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, sampler in enumerate(samplers):
        vals = [df.loc[sampler, m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=sampler,
                      color=COLORS[sampler], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.65, 1.01)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("LSTM — Oversampling Method Comparison (creditcard, 5-fold CV)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Sampler", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Annotate F1 bars for AGSS and SMOTE
    for i, sampler in enumerate(samplers):
        f1_val = df.loc[sampler, "f1"]
        bar_x  = x[3] + offsets[i]
        ax.text(bar_x, f1_val + 0.003, f"{f1_val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_metrics_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_precision_recall(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for sampler in df.index:
        prec = df.loc[sampler, "precision"]
        rec  = df.loc[sampler, "recall"]
        f1   = df.loc[sampler, "f1"]
        color = COLORS[sampler]
        size  = 300 if sampler == "AGSS" else 160
        ax.scatter(rec, prec, s=size, color=color, zorder=3,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(f"{sampler}\nF1={f1:.3f}",
                    (rec, prec), textcoords="offset points",
                    xytext=(8, 4), fontsize=9,
                    color=color, fontweight="bold" if sampler == "AGSS" else "normal")

    # F1 iso-curves
    for f1_iso in [0.70, 0.75, 0.80, 0.85, 0.90]:
        r = np.linspace(0.01, 0.99, 300)
        p = f1_iso * r / (2 * r - f1_iso)
        mask = (p > 0) & (p <= 1)
        ax.plot(r[mask], p[mask], "--", color="gray", linewidth=0.7, alpha=0.5)
        ax.text(r[mask][-1] + 0.005, p[mask][-1], f"F1={f1_iso:.2f}",
                fontsize=7, color="gray", va="center")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Recall — LSTM Oversampling Methods\n(creditcard dataset, 5-fold CV mean)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0.60, 0.95)
    ax.set_ylim(0.65, 1.00)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_precision_recall.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_f1_heatmap(df):
    # Phase 1: single model (LSTM). Rows = models, cols = samplers
    data = pd.DataFrame({"LSTM": df["f1"]}).T
    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = ax.imshow(data.values, cmap="RdYlGn", vmin=0.70, vmax=0.95, aspect="auto")

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, fontsize=11)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=11)
    ax.set_title("F1-score Heatmap — LSTM × Oversampling Method\n(creditcard, 5-fold CV)",
                 fontsize=10, fontweight="bold")

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if 0.75 < val < 0.90 else "white")

    plt.colorbar(im, ax=ax, label="F1-score", fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_f1_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_german_f1_bar(path: str):
    if not os.path.exists(path):
        print(f"Skipping German bar chart — {path} not found yet.")
        return
    df = pd.read_csv(path)
    samplers = df["sampler"].unique().tolist()
    models   = df["model"].unique().tolist()
    x = np.arange(len(samplers))
    width = 0.35
    model_colors = {"RNN": "#E74C3C", "LSTM": "#3498DB"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [df[(df["model"] == model) & (df["sampler"] == s)]["f1_good"].values[0]
                for s in samplers]
        offset = (i - (len(models)-1)/2) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=model_colors.get(model, "#95A5A6"),
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.8436, color="red", linestyle="--", linewidth=1.2, label="Paper LSTM+AGSS (0.8436)")
    ax.axhline(0.8489, color="darkred", linestyle=":", linewidth=1.2, label="Paper RNN+AGSS (0.8489)")
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, fontsize=11)
    ax.set_ylim(0.75, 0.90)
    ax.set_ylabel("F1-score (good credit class)", fontsize=11)
    ax.set_title("Phase 2 — German Credit-Risk: F1 by Model & Sampler\n(5-fold CV, threshold=0.8)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_german_f1_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_german_f1_heatmap(path: str):
    if not os.path.exists(path):
        print(f"Skipping German heatmap — {path} not found yet.")
        return
    df = pd.read_csv(path)
    pivot = df.pivot(index="model", columns="sampler", values="f1_good")

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.78, vmax=0.88, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_title("F1 Heatmap (good credit class) — German Dataset\n(model × sampler, 5-fold CV)",
                 fontsize=10, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="F1 (good)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_german_f1_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_phase1_lstm_rnn_f1():
    lstm_path = os.path.join(RESULTS_DIR, "phase1_lstm_creditcard.csv")
    rnn_path  = os.path.join(RESULTS_DIR, "phase1_rnn_creditcard.csv")
    if not os.path.exists(rnn_path):
        print(f"Skipping LSTM vs RNN chart — {rnn_path} not found.")
        return

    lstm_df = pd.read_csv(lstm_path, index_col="sampler")
    rnn_df  = pd.read_csv(rnn_path,  index_col="sampler")

    samplers = lstm_df.index.tolist()
    x = np.arange(len(samplers))
    width = 0.35
    model_colors = {"LSTM": "#3498DB", "RNN": "#E74C3C"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, df) in enumerate([("LSTM", lstm_df), ("RNN", rnn_df)]):
        vals   = [df.loc[s, "f1"] for s in samplers]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=label,
                        color=model_colors[label], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(samplers, fontsize=11)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("F1-score (fraud class)", fontsize=11)
    ax.set_title("Phase 1 — LSTM vs RNN: F1 by Oversampling Method\n(creditcard dataset, 5-fold CV)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_phase1_lstm_rnn_f1.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_under_german(path):
    if not os.path.exists(path):
        print(f"Skipping German undersampling chart — {path} not found.")
        return
    df = pd.read_csv(path)
    samplers = df["sampler"].unique().tolist()
    models   = df["model"].unique().tolist()
    x = np.arange(len(samplers))
    width = 0.35
    model_colors = {"RNN": "#E74C3C", "LSTM": "#3498DB"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [df[(df["model"] == model) & (df["sampler"] == s)]["f1_good"].values[0]
                for s in samplers]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=model_colors.get(model, "#95A5A6"),
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.8601, color="darkred", linestyle=":", linewidth=1.2,
               label="Paper RNN+AGSS under (0.8601)")
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, fontsize=11)
    ax.set_ylim(0.68, 0.90)
    ax.set_ylabel("F1-score (good credit class)", fontsize=11)
    ax.set_title("Phase 2 — German Undersampling: F1 by Model & Sampler\n(5-fold CV, threshold=0.8)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_under_german.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_under_creditcard(path):
    if not os.path.exists(path):
        print(f"Skipping creditcard undersampling chart — {path} not found.")
        return
    df = pd.read_csv(path)
    samplers = df["sampler"].unique().tolist()
    models   = df["model"].unique().tolist()
    x = np.arange(len(samplers))
    width = 0.35
    model_colors = {"LSTM": "#3498DB", "RNN": "#E74C3C"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [df[(df["model"] == model) & (df["sampler"] == s)]["roc_auc"].values[0]
                for s in samplers]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=model_colors.get(model, "#95A5A6"),
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(samplers, fontsize=11)
    ax.set_ylim(0.80, 1.00)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("Phase 1 — Creditcard Undersampling: ROC-AUC by Model & Sampler\n"
                 "(5-fold CV — ROC-AUC used as F1 is unreliable at 1:1 subsampling ratio)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_under_creditcard.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


MODEL_COLORS = {
    "LSTM":        "#3498DB",
    "RNN":         "#E74C3C",
    "GAN":         "#F39C12",
    "Transformer": "#9B59B6",
}


def _grouped_bar(ax, data, models, samplers, ylabel, title, ylim, annotate=True):
    x = np.arange(len(samplers))
    width = 0.8 / len(models)
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width
    for i, model in enumerate(models):
        vals = [data.get((model, s), 0) for s in samplers]
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        if annotate:
            for bar, v in zip(bars, vals):
                if v > ylim[0] + 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="Model", fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def fig_phase1_over_all_models():
    lstm = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_lstm_creditcard.csv"))
    rnn  = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_rnn_creditcard.csv"))
    gan  = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_gan_over_tuned_creditcard.csv"))
    tfm  = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_transformer_over_creditcard.csv"))

    data = {}
    for row in lstm.itertuples(): data[("LSTM", row.sampler)] = row.f1
    for row in rnn.itertuples():  data[("RNN",  row.sampler)] = row.f1
    for row in gan.itertuples():  data[("GAN",  row.sampler)] = row.f1
    for row in tfm.itertuples():  data[("Transformer", row.sampler)] = row.f1

    samplers = ["ROS", "SMOTE", "ADASYN", "AGSS"]
    models   = ["LSTM", "RNN", "GAN", "Transformer"]
    fig, ax = plt.subplots(figsize=(12, 6))
    _grouped_bar(ax, data, models, samplers,
                 "F1-score (fraud class)",
                 "Phase 1 — Creditcard Oversampling: F1 by Model & Sampler\n"
                 "(5-fold CV; Transformer at thresh=0.5, AGSS adaptive eps=4.838)",
                 (0.0, 1.02))
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_phase1_over_all_models.png")
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved: {out}")


def fig_phase1_under_all_models():
    lstm_rnn = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_under_creditcard.csv"))
    gan      = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_gan_under_creditcard.csv"))
    tfm      = pd.read_csv(os.path.join(RESULTS_DIR, "phase1_transformer_under_creditcard.csv"))

    data = {}
    for row in lstm_rnn.itertuples():
        data[(row.model, row.sampler)] = row.f1
    for row in gan.itertuples():
        data[("GAN", row.sampler)] = row.f1
    for row in tfm.itertuples():
        data[("Transformer", row.sampler)] = row.f1

    samplers = ["AGSS", "RUS", "TomekLinks", "ENN", "NearMiss"]
    models   = ["LSTM", "RNN", "GAN", "Transformer"]
    fig, ax = plt.subplots(figsize=(14, 6))
    _grouped_bar(ax, data, models, samplers,
                 "F1-score (fraud class, tuned threshold)",
                 "Phase 1 — Creditcard Undersampling: F1 by Model & Sampler\n"
                 "(5-fold CV, post-hoc threshold tuning)",
                 (0.0, 1.02))
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_phase1_under_all_models.png")
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved: {out}")


def fig_phase2_over_all_models():
    lstm_rnn = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_german_final.csv"))
    gan      = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_gan_over_german.csv"))
    tfm      = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_transformer_over_german.csv"))

    data = {}
    for row in lstm_rnn.itertuples():
        data[(row.model, row.sampler)] = row.f1_good
    for row in gan.itertuples():
        data[("GAN", row.sampler)] = row.f1_good
    for row in tfm.itertuples():
        data[("Transformer", row.sampler)] = row.f1_good

    samplers = ["ROS", "SMOTE", "ADASYN", "AGSS"]
    models   = ["LSTM", "RNN", "GAN", "Transformer"]
    fig, ax = plt.subplots(figsize=(12, 6))
    _grouped_bar(ax, data, models, samplers,
                 "F1-score (good credit class)",
                 "Phase 2 — German Oversampling: F1 (good class) by Model & Sampler\n"
                 "(5-fold CV, tuned threshold)",
                 (0.55, 0.92), annotate=True)
    ax.axhline(0.8489, color="darkred", linestyle=":", linewidth=1.2,
               label="Paper RNN+AGSS (0.8489)")
    ax.legend(title="Model", fontsize=9)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_phase2_over_all_models.png")
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved: {out}")


def fig_phase2_under_all_models():
    lstm_rnn = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_under_german.csv"))
    gan      = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_gan_under_german.csv"))
    tfm      = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_transformer_under_german.csv"))

    data = {}
    for row in lstm_rnn.itertuples():
        data[(row.model, row.sampler)] = row.f1_good
    for row in gan.itertuples():
        data[("GAN", row.sampler)] = row.f1_good
    for row in tfm.itertuples():
        data[("Transformer", row.sampler)] = row.f1_good

    samplers = ["AGSS", "RUS", "TomekLinks", "ENN", "NearMiss"]
    models   = ["LSTM", "RNN", "GAN", "Transformer"]
    fig, ax = plt.subplots(figsize=(14, 6))
    _grouped_bar(ax, data, models, samplers,
                 "F1-score (good credit class)",
                 "Phase 2 — German Undersampling: F1 (good class) by Model & Sampler\n"
                 "(5-fold CV, tuned threshold)",
                 (0.55, 0.92), annotate=True)
    ax.axhline(0.8601, color="darkred", linestyle=":", linewidth=1.2,
               label="Paper RNN+AGSS under (0.8601)")
    ax.legend(title="Model", fontsize=9)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_phase2_under_all_models.png")
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved: {out}")


def fig_transformer_german_tuning():
    path = os.path.join(RESULTS_DIR, "transformer_german_grid.csv")
    if not os.path.exists(path):
        print(f"Skipping Transformer German tuning chart — {path} not found.")
        return
    df = pd.read_csv(path).sort_values("f1_good", ascending=False)

    labels = [f"d={int(r.d_model)}\ne={int(r.epochs)}\nlr={r.lr:.0e}" for _, r in df.iterrows()]
    colors = ["#9B59B6" if i == 0 else "#C39BD3" for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(df)), df["f1_good"], color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, df["f1_good"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.8489, color="darkred", linestyle=":", linewidth=1.2, label="Paper RNN+AGSS (0.8489)")
    ax.axhline(0.7021, color="gray", linestyle="--", linewidth=1.0, label="Baseline d_model=32 (0.7021)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.65, 0.90)
    ax.set_ylabel("F1-score (good credit class)", fontsize=11)
    ax.set_title("Transformer German Tuning — d_model Grid Search (AGSS oversampling, 5-fold CV)\n"
                 "Best: d_model=128, epochs=300, lr=5e-4 → F1=0.8241 (gap: 14.7% → 2.5%)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_transformer_german_tuning.png")
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved: {out}")


if __name__ == "__main__":
    df = load()
    fig_metrics_bar(df)
    fig_precision_recall(df)
    fig_f1_heatmap(df)
    fig_phase1_lstm_rnn_f1()

    german_path = os.path.join(RESULTS_DIR, "phase2_german_final.csv")
    fig_german_f1_bar(german_path)
    fig_german_f1_heatmap(german_path)

    fig_under_german(os.path.join(RESULTS_DIR, "phase2_under_german.csv"))
    fig_under_creditcard(os.path.join(RESULTS_DIR, "phase1_under_creditcard.csv"))

    fig_phase1_over_all_models()
    fig_phase1_under_all_models()
    fig_phase2_over_all_models()
    fig_phase2_under_all_models()
    fig_transformer_german_tuning()

    print("\nAll visualizations saved to figures/")
