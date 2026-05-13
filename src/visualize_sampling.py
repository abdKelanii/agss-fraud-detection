"""
Generate visual comparisons of SMOTE vs AGSS for the presentation.
Saves PNG files to website/public/figures/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from agss import AGSS

# ── colour palette matching the dark website theme ──────────────────────────
BG       = '#0d1117'
MAJ_C    = '#334155'   # slate majority points
MAJ_SYN  = '#1e3a5f'
MIN_C    = '#f87171'   # red minority
SMOTE_C  = '#fbbf24'   # amber SMOTE synthetic
AGSS_C   = '#34d399'   # green AGSS synthetic
CLUSTER1 = '#818cf8'
CLUSTER2 = '#c084fc'

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'website', 'public', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

def make_dataset(seed=42):
    rng = np.random.default_rng(seed)
    # Majority class – large elliptical cloud
    n_maj = 500
    X_maj = rng.multivariate_normal([0, 0], [[3, 1.2], [1.2, 2]], n_maj)

    # Minority – two tight clusters + 3 noisy boundary points
    c1 = rng.multivariate_normal([-1.8, 1.8], [[0.07, 0.02], [0.02, 0.07]], 14)
    c2 = rng.multivariate_normal([ 1.5, -1.5], [[0.08, -0.02], [-0.02, 0.06]], 12)
    noise = np.array([[0.1, 0.3], [-0.3, 0.1], [0.2, -0.2]])   # near boundary
    X_min = np.vstack([c1, c2, noise])
    y = np.concatenate([np.zeros(n_maj), np.ones(len(X_min))])
    X = np.vstack([X_maj, X_min])
    return X, y, X_maj, X_min, c1, c2

def smote_samples(X, y):
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    mask = y_res == 1
    orig_minority = X[y == 1]
    # new samples = those not in the original minority set
    new_pts = []
    for pt in X_res[mask]:
        if not any(np.allclose(pt, o, atol=1e-8) for o in orig_minority):
            new_pts.append(pt)
    return np.array(new_pts)

def agss_samples(X, y):
    sampler = AGSS(eps=0.5, min_samples=2, n_neighbors=3, random_state=42,
                   adaptive_eps=True, density_weighted=True)
    X_res, y_res = sampler.fit_resample(X, y)
    orig_minority = X[y == 1]
    new_pts = []
    for pt in X_res[y_res == 1]:
        if not any(np.allclose(pt, o, atol=1e-8) for o in orig_minority):
            new_pts.append(pt)
    return np.array(new_pts)

# ── Figure 1: side-by-side 3-panel ──────────────────────────────────────────
def fig_three_panel():
    X, y, X_maj, X_min, c1, c2 = make_dataset()
    smote_pts = smote_samples(X, y)
    agss_pts  = agss_samples(X, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.88, bottom=0.08)

    titles = ['Original Data', 'SMOTE', 'AGSS (ours)']
    for ax, title in zip(axes, titles):
        ax.set_facecolor(BG)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=10)

        # Majority
        ax.scatter(X_maj[:, 0], X_maj[:, 1], c=MAJ_C, s=12, alpha=0.45, linewidths=0)

        if title == 'SMOTE' and len(smote_pts):
            ax.scatter(smote_pts[:, 0], smote_pts[:, 1], c=SMOTE_C, s=28,
                       alpha=0.85, linewidths=0, zorder=4, marker='D')
        if title == 'AGSS (ours)' and len(agss_pts):
            ax.scatter(agss_pts[:, 0], agss_pts[:, 1], c=AGSS_C, s=28,
                       alpha=0.85, linewidths=0, zorder=4, marker='D')

        # Minority clusters coloured
        ax.scatter(c1[:, 0], c1[:, 1], c=CLUSTER1, s=52, zorder=5, linewidths=0)
        ax.scatter(c2[:, 0], c2[:, 1], c=CLUSTER2, s=52, zorder=5, linewidths=0)
        # Noise points
        noise_idx = np.array([0.1, 0.3, -0.3])  # rough filter
        noise_pts = X_min[14 + 12:]
        ax.scatter(noise_pts[:, 0], noise_pts[:, 1], c=MIN_C, s=52,
                   zorder=5, marker='x', linewidths=1.5)

    # Legend
    legend_items = [
        mpatches.Patch(color=MAJ_C,    label='Majority class'),
        mpatches.Patch(color=CLUSTER1, label='Minority cluster 1'),
        mpatches.Patch(color=CLUSTER2, label='Minority cluster 2'),
        mpatches.Patch(color=MIN_C,    label='Noisy minority (boundary)'),
        mpatches.Patch(color=SMOTE_C,  label='SMOTE synthetic'),
        mpatches.Patch(color=AGSS_C,   label='AGSS synthetic'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=6,
               facecolor='#1e293b', edgecolor='#334155',
               labelcolor='white', fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    out = os.path.join(OUT_DIR, 'compare_smote_agss_3panel.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')

# ── Figure 2: tight 2-panel SMOTE vs AGSS ───────────────────────────────────
def fig_two_panel():
    X, y, X_maj, X_min, c1, c2 = make_dataset()
    smote_pts = smote_samples(X, y)
    agss_pts  = agss_samples(X, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.88, bottom=0.14)

    for ax, title, syn_pts, syn_color, problem_text in [
        (ax1, 'SMOTE', smote_pts, SMOTE_C,
         '✗ Generates samples near noisy\n   boundary & between outliers'),
        (ax2, 'AGSS', agss_pts,   AGSS_C,
         '✓ Samples stay within dense\n   cluster cores only'),
    ]:
        ax.set_facecolor(BG)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')

        # Title bar colour
        bar_c = '#fbbf2422' if 'SMOTE' in title else '#34d39922'
        ax.set_title(title, color=syn_color, fontsize=16, fontweight='bold', pad=10)

        ax.scatter(X_maj[:, 0], X_maj[:, 1], c=MAJ_C, s=14, alpha=0.4, linewidths=0)

        if len(syn_pts):
            ax.scatter(syn_pts[:, 0], syn_pts[:, 1], c=syn_color, s=35,
                       alpha=0.82, zorder=4, marker='D', linewidths=0,
                       label='Synthetic')

        ax.scatter(c1[:, 0], c1[:, 1], c=CLUSTER1, s=60, zorder=5, linewidths=0, label='Cluster 1')
        ax.scatter(c2[:, 0], c2[:, 1], c=CLUSTER2, s=60, zorder=5, linewidths=0, label='Cluster 2')
        noise_pts = X_min[26:]
        ax.scatter(noise_pts[:, 0], noise_pts[:, 1], c=MIN_C, s=60,
                   zorder=6, marker='x', linewidths=2, label='Noisy')

        # Draw convex-hull-like cluster circles for AGSS to show containment
        if 'AGSS' in title:
            for cpts, col in [(c1, CLUSTER1), (c2, CLUSTER2)]:
                cx, cy = cpts[:, 0].mean(), cpts[:, 1].mean()
                r = max(np.sqrt(((cpts - [cx, cy])**2).sum(axis=1))) + 0.25
                circle = plt.Circle((cx, cy), r, color=col, fill=False,
                                    linewidth=1.2, linestyle='--', alpha=0.5, zorder=3)
                ax.add_patch(circle)

        # Annotation
        t_col = '#fbbf24' if 'SMOTE' in title else '#34d399'
        ax.text(0.03, 0.04, problem_text, transform=ax.transAxes,
                color=t_col, fontsize=9.5, va='bottom', style='italic',
                bbox=dict(facecolor='#0d1117cc', edgecolor='none', pad=4))

    out = os.path.join(OUT_DIR, 'compare_smote_agss_2panel.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')

# ── Figure 3: boundary zoom – shows SMOTE polluting boundary ─────────────────
def fig_boundary_zoom():
    rng = np.random.default_rng(42)
    # Create a harder case: one minority cluster near boundary
    n_maj = 300
    X_maj = rng.multivariate_normal([0, 0], [[2.5, 0.8], [0.8, 2.5]], n_maj)
    cluster = rng.multivariate_normal([-0.4, 0.6], [[0.05, 0.01], [0.01, 0.05]], 12)
    near_boundary = rng.multivariate_normal([0.1, 0.2], [[0.04, 0], [0, 0.04]], 4)
    X_min = np.vstack([cluster, near_boundary])
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_maj), np.ones(len(X_min))])

    smote_pts = smote_samples(X, y)
    sampler = AGSS(eps=0.35, min_samples=2, n_neighbors=3, random_state=42,
                   adaptive_eps=True, density_weighted=True)
    X_r, y_r = sampler.fit_resample(X, y)
    agss_pts = np.array([pt for pt in X_r[y_r == 1]
                         if not any(np.allclose(pt, o, atol=1e-8) for o in X_min)])

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

    out = os.path.join(OUT_DIR, 'compare_boundary_zoom.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')

if __name__ == '__main__':
    fig_three_panel()
    fig_two_panel()
    fig_boundary_zoom()
    print('All figures generated.')
