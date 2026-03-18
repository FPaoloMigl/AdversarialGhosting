import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# === CONFIGURATION ===
OUTPUT_DIR = 'AdversarialGhosting/QualityAssessment/step3_angle_decompose/optimization/gmm/'

ALL_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'all_results.json')
TOP_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'top_results_diverse.json')

# Physical bounds
XY_MIN = -30.0
XY_MAX = 30.0

# Diversity filter (for validation candidates only)
DIVERSITY_MARGIN = 3.0
TOP_PERCENT = 0.20

# GMM Settings (fitted on ALL 50 results)
GMM_COMPONENTS = 3


def load_results():
    if not os.path.exists(ALL_RESULTS_PATH):
        print(f"Error: {ALL_RESULTS_PATH} not found. Run Phase 1+2 first.")
        return None

    with open(ALL_RESULTS_PATH, 'r') as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} total results.")
    return all_results


def filter_diverse_top(all_results, top_percent=0.20, margin=3.0):
    """Select top results by loss with minimum spatial distance between them."""
    sorted_results = sorted(all_results, key=lambda r: r['loss'], reverse=True)
    n_candidates = max(1, int(len(sorted_results) * top_percent))

    selected = []
    for result in sorted_results:
        is_diverse = True
        for s in selected:
            dist = np.sqrt((result['x'] - s['x'])**2 + (result['y'] - s['y'])**2)
            if dist < margin:
                is_diverse = False
                break
        if is_diverse:
            selected.append(result)
        if len(selected) >= n_candidates:
            break

    return selected


def fit_gmm(all_results, n_components):
    """Fit GMM on ALL results to capture natural convergence basins."""
    all_xy = np.array([[r['x'], r['y']] for r in all_results])
    n_components = min(n_components, len(all_results))
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(all_xy)
    return gmm


def visualize(all_results, diverse_top, gmm):
    all_xy = np.array([[r['x'], r['y']] for r in all_results])
    all_losses = np.array([r['loss'] for r in all_results])
    all_labels = gmm.predict(all_xy)

    top_xy = np.array([[r['x'], r['y']] for r in diverse_top])
    top_losses = np.array([r['loss'] for r in diverse_top])

    # Density grid
    x_grid = np.linspace(XY_MIN - 2, XY_MAX + 2, 200)
    y_grid = np.linspace(XY_MIN - 2, XY_MAX + 2, 200)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

    log_density = gmm.score_samples(grid_points)
    density = np.exp(log_density).reshape(X_grid.shape)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # --- Panel 1: All 50 runs colored by loss ---
    scatter = axes[0].scatter(all_xy[:, 0], all_xy[:, 1], c=all_losses,
                              cmap='RdYlGn', s=60, edgecolors='black', linewidth=0.5, zorder=5)
    axes[0].set_xlim(XY_MIN - 2, XY_MAX + 2)
    axes[0].set_ylim(XY_MIN - 2, XY_MAX + 2)
    axes[0].set_title(f'All {len(all_results)} Optimization Endpoints (by Loss)', fontsize=12)
    axes[0].set_xlabel('X (meters)')
    axes[0].set_ylabel('Y (meters)')
    plt.colorbar(scatter, ax=axes[0], label='Disruption Loss')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # --- Panel 2: GMM heatmap (fitted on all 50) + diverse top overlay ---
    contour = axes[1].contourf(X_grid, Y_grid, density, levels=30, cmap='hot_r', alpha=0.8)

    # Plot all 50 as small dots colored by GMM cluster
    cluster_colors = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3', '#FF7F00']
    for i in range(gmm.n_components):
        mask = all_labels == i
        axes[1].scatter(all_xy[mask, 0], all_xy[mask, 1], 
                       c=cluster_colors[i % len(cluster_colors)],
                       s=30, alpha=0.5, edgecolors='none', zorder=4)

    # Overlay diverse top as large numbered markers
    for i, r in enumerate(diverse_top):
        axes[1].scatter(r['x'], r['y'], c='lime', s=120,
                       edgecolors='black', linewidth=2, zorder=7)
        axes[1].annotate(f"#{i+1}", (r['x'] + 0.6, r['y'] + 0.6), fontsize=9,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))

    # Mark GMM centers
    for i, mean in enumerate(gmm.means_):
        axes[1].scatter(mean[0], mean[1], c='cyan', marker='X', s=250,
                       edgecolors='black', linewidth=2, zorder=10)
        axes[1].annotate(f'C{i+1}', (mean[0] + 0.8, mean[1] + 0.8), fontsize=11,
                        fontweight='bold', color='cyan')

    axes[1].set_xlim(XY_MIN - 2, XY_MAX + 2)
    axes[1].set_ylim(XY_MIN - 2, XY_MAX + 2)
    axes[1].set_title(f'GMM (all {len(all_results)}) + Top {len(diverse_top)} Diverse Candidates', fontsize=12)
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    plt.colorbar(contour, ax=axes[1], label='Placement Density')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    # --- Panel 3: Loss distribution per GMM cluster ---
    for i in range(gmm.n_components):
        mask = all_labels == i
        axes[2].hist(all_losses[mask], bins=15, alpha=0.5,
                    color=cluster_colors[i % len(cluster_colors)],
                    edgecolor='white', label=f'C{i+1} (n={mask.sum()})')

    # Mark diverse top losses
    for i, r in enumerate(diverse_top):
        axes[2].axvline(x=r['loss'], color='lime', linestyle='-', linewidth=1, alpha=0.5)

    axes[2].set_title('Loss Distribution by GMM Cluster', fontsize=12)
    axes[2].set_xlabel('Disruption Loss')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'gmm_placement_heatmap_v2.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to: {plot_path}")


def print_summary(gmm, all_results, diverse_top):
    all_xy = np.array([[r['x'], r['y']] for r in all_results])
    all_losses = np.array([r['loss'] for r in all_results])
    all_angles = np.array([r['angle'] for r in all_results])
    all_labels = gmm.predict(all_xy)

    top_xy = np.array([[r['x'], r['y']] for r in diverse_top])
    top_labels = gmm.predict(top_xy)

    # === GMM Cluster Summary (from all 50 results) ===
    print(f"\n{'='*60}")
    print(f"GMM CLUSTER SUMMARY (fitted on all {len(all_results)} results)")
    print(f"{'='*60}")

    for i in range(gmm.n_components):
        mask = all_labels == i
        cluster_losses = all_losses[mask]
        cluster_angles = all_angles[mask]

        print(f"\n  Cluster {i+1} ({mask.sum()} members, weight={gmm.weights_[i]:.1%}):")
        print(f"    Center:     X={gmm.means_[i][0]:+.2f}m, Y={gmm.means_[i][1]:+.2f}m")
        print(f"    Loss:       mean={np.mean(cluster_losses):.4f}, best={np.max(cluster_losses):.4f}")
        print(f"    Angle:      mean={np.mean(cluster_angles):.1f}, std={np.std(cluster_angles):.1f}")

    # === Diverse Validation Candidates ===
    print(f"\n{'='*60}")
    print(f"DIVERSE VALIDATION CANDIDATES ({len(diverse_top)} placements, {DIVERSITY_MARGIN}m margin)")
    print(f"{'='*60}")

    for i, r in enumerate(diverse_top):
        cluster = top_labels[i] + 1
        print(f"\n  #{i+1} [Cluster {cluster}]:")
        print(f"    Position: X={r['x']:+.2f}m, Y={r['y']:+.2f}m")
        print(f"    Angle:    {r['angle']:.1f}")
        print(f"    Loss:     {r['loss']:.4f}")

    # === Overall Stats ===
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"  Total optimization runs:  {len(all_results)}")
    print(f"  Diverse candidates:       {len(diverse_top)}")
    print(f"  Diversity margin:         {DIVERSITY_MARGIN}m")
    print(f"  GMM components:           {gmm.n_components}")
    print(f"  Loss range (all):         [{all_losses.min():.4f}, {all_losses.max():.4f}]")
    print(f"  Best placement:           X={diverse_top[0]['x']:+.2f}, Y={diverse_top[0]['y']:+.2f}, A={diverse_top[0]['angle']:.1f}")


def main():
    all_results = load_results()
    if all_results is None:
        return

    # === Diversity Filter (for validation candidates) ===
    print(f"\n{'='*60}")
    print(f"SELECTING DIVERSE VALIDATION CANDIDATES")
    print(f"  Top {int(TOP_PERCENT*100)}% by loss, {DIVERSITY_MARGIN}m minimum spacing")
    print(f"{'='*60}")

    diverse_top = filter_diverse_top(all_results, top_percent=TOP_PERCENT, margin=DIVERSITY_MARGIN)

    print(f"\nSelected {len(diverse_top)} diverse placements:")
    for i, r in enumerate(diverse_top):
        print(f"  #{i+1:02d} | X:{r['x']:+.2f}m | Y:{r['y']:+.2f}m | A:{r['angle']:.1f} | Loss:{r['loss']:.4f}")

    with open(TOP_RESULTS_PATH, 'w') as f:
        json.dump(diverse_top, f, indent=2)
    print(f"Saved to: {TOP_RESULTS_PATH}")

    # === Fit GMM on ALL 50 results ===
    print(f"\n{'='*60}")
    print(f"FITTING GMM ({GMM_COMPONENTS} components on all {len(all_results)} results)")
    print(f"{'='*60}")

    gmm = fit_gmm(all_results, GMM_COMPONENTS)

    # === Visualize & Summarize ===
    visualize(all_results, diverse_top, gmm)
    print_summary(gmm, all_results, diverse_top)


if __name__ == '__main__':
    main()