"""
Generate visual comparisons of SMOTE vs AGSS for the presentation.
Saves PNG files to website/public/figures/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

# ── colour palette matching the dark website theme ──────────────────────────
BG       = '#0d1117'
MAJ_C    = '#334155'
MIN_C    = '#f87171'
SMOTE_C  = '#fbbf24'
AGSS_C   = '#34d399'
CLUSTER1 = '#818cf8'
CLUSTER2 = '#c084fc'

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'website', 'public', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def make_dataset(seed=42):
    rng = np.random.default_rng(seed)
    n_maj = 500
    X_maj = rng.multivariate_normal([0, 0], [[3, 1.2], [1.2, 2]], n_maj)

    c1 = rng.multivariate_normal([-1.8,  1.8], [[0.07, 0.02], [0.02, 0.07]], 14)
    c2 = rng.multivariate_normal([ 1.5, -1.5], [[0.08, -0.02], [-0.02, 0.06]], 12)

    # Noise points are far apart from each other (pairwise dist > 1.3)
    # so they cannot form a DBSCAN cluster at any reasonable eps.
    noise = np.array([[-0.4, 1.3], [0.9, -0.5], [0.0, 0.6]])

    X_min = np.vstack([c1, c2, noise])
    y = np.concatenate([np.zeros(n_maj), np.ones(len(X_min))])
    X = np.vstack([X_maj, X_min])
    return X, y, X_maj, X_min, c1, c2, noise


def smote_samples(X, y):
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    orig = X[y == 1]
    new_pts = [pt for pt in X_res[y_res == 1]
               if not any(np.allclose(pt, o, atol=1e-8) for o in orig)]
    return np.array(new_pts)


def agss_samples_viz(X_min, eps=0.45, min_samples=5, seed=42):
    """
    Implements the AGSS algorithm steps directly for visualization:
      1. DBSCAN on minority samples — points not in a dense cluster are noise.
      2. KNN fit *within* each cluster only.
      3. Curvature interpolation: x_syn = p + α(q-p) + γ·sin(θ)·perp(q-p)

    Using min_samples=5 ensures isolated noise points (< 5 mutual neighbours)
    are never used as synthesis seeds, guaranteeing no inter-cluster bridging.
    """
    rng = np.random.default_rng(seed)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_min)
    labels = db.labels_

    new_pts = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue                        # noise — AGSS skips these
        cluster_pts = X_min[labels == lbl]
        if len(cluster_pts) < 3:
            continue
        k = min(3, len(cluster_pts) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(cluster_pts)
        n_syn = int(len(cluster_pts) * 2.2)
        for _ in range(n_syn):
            idx = rng.integers(len(cluster_pts))
            p = cluster_pts[idx]
            _, nn_idx = nbrs.kneighbors([p])
            q = cluster_pts[rng.choice(nn_idx[0][1:])]
            alpha = rng.uniform(0, 1)
            gamma = rng.uniform(0, 0.08)
            theta = rng.uniform(0, 2 * np.pi)
            d = q - p
            perp = np.array([-d[1], d[0]])
            norm = np.linalg.norm(perp)
            if norm > 1e-10:
                perp /= norm
            new_pts.append(p + alpha * d + gamma * np.sin(theta) * perp)

    return np.array(new_pts) if new_pts else np.zeros((0, 2))


def cluster_circle(pts, pad=0.28):
    """Return (cx, cy, radius) enclosing all pts with padding."""
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)).max() + pad
    return cx, cy, r


# ── Figure 1: 3-panel ────────────────────────────────────────────────────────
def fig_three_panel():
    X, y, X_maj, X_min, c1, c2, noise = make_dataset()
    smote_pts = smote_samples(X, y)
    agss_pts  = agss_samples_viz(X_min)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.88, bottom=0.10)

    titles = ['Original Data', 'SMOTE', 'AGSS (ours)']
    for ax, title in zip(axes, titles):
        ax.set_facecolor(BG)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=10)

        ax.scatter(X_maj[:, 0], X_maj[:, 1], c=MAJ_C, s=12, alpha=0.45, linewidths=0)

        if title == 'SMOTE' and len(smote_pts):
            ax.scatter(smote_pts[:, 0], smote_pts[:, 1], c=SMOTE_C, s=28,
                       alpha=0.85, zorder=4, marker='D', linewidths=0)

        if title == 'AGSS (ours)':
            # Draw DBSCAN cluster boundaries first
            for cpts, col in [(c1, CLUSTER1), (c2, CLUSTER2)]:
                cx, cy, r = cluster_circle(cpts)
                circle = plt.Circle((cx, cy), r, color=col, fill=False,
                                    linewidth=1.5, linestyle='--', alpha=0.6, zorder=3)
                ax.add_patch(circle)
            if len(agss_pts):
                ax.scatter(agss_pts[:, 0], agss_pts[:, 1], c=AGSS_C, s=28,
                           alpha=0.85, zorder=4, marker='D', linewidths=0)

        ax.scatter(c1[:, 0], c1[:, 1], c=CLUSTER1, s=52, zorder=5, linewidths=0)
        ax.scatter(c2[:, 0], c2[:, 1], c=CLUSTER2, s=52, zorder=5, linewidths=0)
        ax.scatter(noise[:, 0], noise[:, 1], c=MIN_C, s=52,
                   zorder=5, marker='x', linewidths=1.5)

    legend_items = [
        mpatches.Patch(color=MAJ_C,    label='Majority class'),
        mpatches.Patch(color=CLUSTER1, label='Minority cluster 1'),
        mpatches.Patch(color=CLUSTER2, label='Minority cluster 2'),
        mpatches.Patch(color=MIN_C,    label='Noisy minority (excluded by AGSS)'),
        mpatches.Patch(color=SMOTE_C,  label='SMOTE synthetic'),
        mpatches.Patch(color=AGSS_C,   label='AGSS synthetic (within clusters only)'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=6,
               facecolor='#1e293b', edgecolor='#334155',
               labelcolor='white', fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    out = os.path.join(OUT_DIR, 'compare_smote_agss_3panel.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')


# ── Figure 2: 2-panel SMOTE vs AGSS ─────────────────────────────────────────
def fig_two_panel():
    X, y, X_maj, X_min, c1, c2, noise = make_dataset()
    smote_pts = smote_samples(X, y)
    agss_pts  = agss_samples_viz(X_min)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.88, bottom=0.14)

    for ax, title, syn_pts, syn_color, caption in [
        (ax1, 'SMOTE', smote_pts, SMOTE_C,
         '✗ Generates samples near noisy\n   boundary & between outliers'),
        (ax2, 'AGSS', agss_pts, AGSS_C,
         '✓ Samples stay within dense\n   cluster cores only'),
    ]:
        ax.set_facecolor(BG)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.set_title(title, color=syn_color, fontsize=16, fontweight='bold', pad=10)

        ax.scatter(X_maj[:, 0], X_maj[:, 1], c=MAJ_C, s=14, alpha=0.4, linewidths=0)

        if len(syn_pts):
            ax.scatter(syn_pts[:, 0], syn_pts[:, 1], c=syn_color, s=35,
                       alpha=0.82, zorder=4, marker='D', linewidths=0, label='Synthetic')

        ax.scatter(c1[:, 0], c1[:, 1], c=CLUSTER1, s=60, zorder=5, linewidths=0)
        ax.scatter(c2[:, 0], c2[:, 1], c=CLUSTER2, s=60, zorder=5, linewidths=0)
        ax.scatter(noise[:, 0], noise[:, 1], c=MIN_C, s=60,
                   zorder=6, marker='x', linewidths=2)

        if 'AGSS' in title:
            # DBSCAN boundaries — show containment
            for cpts, col in [(c1, CLUSTER1), (c2, CLUSTER2)]:
                cx, cy, r = cluster_circle(cpts)
                circle = plt.Circle((cx, cy), r, color=col, fill=False,
                                    linewidth=1.5, linestyle='--', alpha=0.55, zorder=3)
                ax.add_patch(circle)
            # Mark each noise point as excluded
            for pt in noise:
                ax.annotate('noise\n(excluded)', xy=pt,
                            xytext=(pt[0] + 0.45, pt[1] + 0.45),
                            color=MIN_C, fontsize=7, alpha=0.75,
                            arrowprops=dict(arrowstyle='->', color=MIN_C,
                                            alpha=0.45, lw=0.8))

        t_col = '#fbbf24' if 'SMOTE' in title else '#34d399'
        ax.text(0.03, 0.04, caption, transform=ax.transAxes,
                color=t_col, fontsize=9.5, va='bottom', style='italic',
                bbox=dict(facecolor='#0d1117cc', edgecolor='none', pad=4))

    out = os.path.join(OUT_DIR, 'compare_smote_agss_2panel.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')


# ── Figure 3: boundary zoom ───────────────────────────────────────────────────
def fig_boundary_zoom():
    rng = np.random.default_rng(42)
    n_maj = 300
    X_maj = rng.multivariate_normal([0, 0], [[2.5, 0.8], [0.8, 2.5]], n_maj)
    cluster      = rng.multivariate_normal([-0.4, 0.6], [[0.05, 0.01], [0.01, 0.05]], 12)
    near_boundary = rng.multivariate_normal([0.1, 0.2], [[0.04, 0], [0, 0.04]], 4)
    X_min = np.vstack([cluster, near_boundary])
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_maj), np.ones(len(X_min))])

    smote_pts = smote_samples(X, y)
    # min_samples=5 → 4-point near_boundary cloud cannot form a cluster
    agss_pts  = agss_samples_viz(X_min, eps=0.45, min_samples=5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.88, bottom=0.06)

    for ax, title, syn_pts, syn_color in [
        (axes[0], 'SMOTE — spreads into majority region', smote_pts, SMOTE_C),
        (axes[1], 'AGSS — stays in dense core',           agss_pts,  AGSS_C),
    ]:
        ax.set_facecolor(BG)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_edgecolor('#1e293b')
        ax.set_title(title, color='white', fontsize=11.5, fontweight='bold', pad=8)

        ax.scatter(X_maj[:, 0], X_maj[:, 1], c=MAJ_C, s=14, alpha=0.4, linewidths=0)
        if len(syn_pts):
            ax.scatter(syn_pts[:, 0], syn_pts[:, 1], c=syn_color, s=40,
                       alpha=0.85, zorder=4, marker='D', linewidths=0)
        ax.scatter(cluster[:, 0], cluster[:, 1], c=CLUSTER1, s=60, zorder=5, linewidths=0)
        ax.scatter(near_boundary[:, 0], near_boundary[:, 1], c=MIN_C, s=60,
                   zorder=6, marker='x', linewidths=2)

        if 'AGSS' in title:
            cx, cy, r = cluster_circle(cluster)
            circle = plt.Circle((cx, cy), r, color=CLUSTER1, fill=False,
                                 linewidth=1.5, linestyle='--', alpha=0.55, zorder=3)
            ax.add_patch(circle)

    out = os.path.join(OUT_DIR, 'compare_boundary_zoom.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    fig_three_panel()
    fig_two_panel()
    fig_boundary_zoom()
    print('All figures generated.')
