import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
CSV_PATH = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/metrics/adversarial_analysis_results.csv')
OUTPUT_PATH = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/metrics/')

# Layers and labels
LAYERS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
LAYER_LABELS = ['F1\nconv1', 'F2\nconv2', 'F3\nconv3', 'F4\nconv4', 'F5\nconv5', 'F6\nconv6', 'F7\nconv_out', 'F8\nshared']
INJECTION_LAYER_IDX = 5  # F6 = index 5 (0-based)
TOP_K = 50  # Number of top scenes for "top-K mean" variant


def load_and_extract(csv_path):
    """Load CSV and extract per-layer MSE and Cosine Distance columns."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} scenes from {csv_path.name}")
    
    mse_cols = [f"{layer}_MSE" for layer in LAYERS]
    cos_cols = [f"{layer}_CosDist" for layer in LAYERS]
    
    # Verify columns exist
    for col in mse_cols + cos_cols:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found in CSV!")
            return None, None, None
    
    return df, mse_cols, cos_cols


# ============================================================
# PLOT 1: MEAN (Honest baseline across all 1000 scenes)
# ============================================================
def plot_mean(df, mse_cols, cos_cols):
    """Dual-axis bar+line plot using mean across all scenes."""
    avg_mse = [df[col].mean() for col in mse_cols]
    avg_cos = [df[col].mean() for col in cos_cols]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LAYERS))
    
    # MSE bars with injection layer highlighted
    colors = ['#2166AC'] * len(LAYERS)
    colors[INJECTION_LAYER_IDX] = '#D6604D'  # Highlight injection point
    
    bars = ax1.bar(x, avg_mse, color=colors, alpha=0.7, width=0.6, 
                   edgecolor='white', linewidth=0.5, label='Feature MSE')
    ax1.set_xlabel('VoxelNeXt Backbone Layers', fontsize=11)
    ax1.set_ylabel('Mean Squared Error', color='#2166AC', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#2166AC')
    ax1.set_xticks(x)
    ax1.set_xticklabels(LAYER_LABELS, fontsize=9)
    
    # Cosine distance line on right axis
    ax2 = ax1.twinx()
    ax2.plot(x, avg_cos, color='#B2182B', marker='D', linewidth=2, 
             markersize=7, label='Cosine Distance', zorder=5)
    ax2.set_ylabel('Cosine Distance', color='#B2182B', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#B2182B')
    
    # Mark injection point
    ax1.axvline(x=INJECTION_LAYER_IDX, color='#D6604D', linestyle=':', 
                alpha=0.4, linewidth=1.5)
    ax1.annotate('Injection\nPoint', xy=(INJECTION_LAYER_IDX, max(avg_mse) * 0.95),
                fontsize=8, ha='center', color='#D6604D', fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    plt.title('Layer Sensitivity — Mean Across All Scenes (N=1000)', fontsize=13, pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    out = OUTPUT_PATH / 'layerwise_sensitivity_MEAN.png'
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")
    
    # Print values
    print("\n  Layer  |  MSE (mean)  |  CosDist (mean)")
    print("  " + "-"*42)
    for i, layer in enumerate(LAYERS):
        marker = " <-- injection" if i == INJECTION_LAYER_IDX else ""
        print(f"  {layer:5s}  |  {avg_mse[i]:.6f}   |  {avg_cos[i]:.6f}{marker}")


# ============================================================
# PLOT 2: TOP-K MEAN (Average of K most-affected scenes)
# ============================================================
def plot_top_k_mean(df, mse_cols, cos_cols, k=TOP_K):
    """For each layer, take the K scenes with highest cosine distance, then average."""
    top_mse = []
    top_cos = []
    
    for mse_col, cos_col in zip(mse_cols, cos_cols):
        # Select top-K scenes by cosine distance (most disrupted)
        top_scenes = df.nlargest(k, cos_col)
        top_mse.append(top_scenes[mse_col].mean())
        top_cos.append(top_scenes[cos_col].mean())
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LAYERS))
    
    colors = ['#2166AC'] * len(LAYERS)
    colors[INJECTION_LAYER_IDX] = '#D6604D'
    
    bars = ax1.bar(x, top_mse, color=colors, alpha=0.7, width=0.6,
                   edgecolor='white', linewidth=0.5, label='Feature MSE')
    ax1.set_xlabel('VoxelNeXt Backbone Layers', fontsize=11)
    ax1.set_ylabel('Mean Squared Error', color='#2166AC', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#2166AC')
    ax1.set_xticks(x)
    ax1.set_xticklabels(LAYER_LABELS, fontsize=9)
    
    ax2 = ax1.twinx()
    ax2.plot(x, top_cos, color='#B2182B', marker='D', linewidth=2,
             markersize=7, label='Cosine Distance', zorder=5)
    ax2.set_ylabel('Cosine Distance', color='#B2182B', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#B2182B')
    
    ax1.axvline(x=INJECTION_LAYER_IDX, color='#D6604D', linestyle=':',
                alpha=0.4, linewidth=1.5)
    ax1.annotate('Injection\nPoint', xy=(INJECTION_LAYER_IDX, max(top_mse) * 0.95),
                fontsize=8, ha='center', color='#D6604D', fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    plt.title(f'Layer Sensitivity — Top-{k} Most Disrupted Scenes', fontsize=13, pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    out = OUTPUT_PATH / f'layerwise_sensitivity_TOP{k}.png'
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")
    
    print(f"\n  Layer  |  MSE (top-{k})  |  CosDist (top-{k})")
    print("  " + "-"*46)
    for i, layer in enumerate(LAYERS):
        marker = " <-- injection" if i == INJECTION_LAYER_IDX else ""
        print(f"  {layer:5s}  |  {top_mse[i]:.6f}     |  {top_cos[i]:.6f}{marker}")


# ============================================================
# PLOT 3: BOX PLOT (Full distribution per layer)
# ============================================================
def plot_boxplot(df, mse_cols, cos_cols):
    """Side-by-side box plots showing full distribution per layer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- MSE Box Plot ---
    mse_data = [df[col].values for col in mse_cols]
    
    bp1 = ax1.boxplot(mse_data, positions=range(len(LAYERS)), widths=0.5,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='.', markersize=2, alpha=0.3),
                      medianprops=dict(color='black', linewidth=1.5))
    
    # Color boxes
    for i, box in enumerate(bp1['boxes']):
        if i == INJECTION_LAYER_IDX:
            box.set_facecolor('#D6604D')
            box.set_alpha(0.7)
        else:
            box.set_facecolor('#2166AC')
            box.set_alpha(0.5)
    
    ax1.set_xticks(range(len(LAYERS)))
    ax1.set_xticklabels(LAYER_LABELS, fontsize=9)
    ax1.set_xlabel('VoxelNeXt Backbone Layers', fontsize=11)
    ax1.set_ylabel('Mean Squared Error', fontsize=11)
    ax1.set_title('Feature MSE Distribution', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax1.axvline(x=INJECTION_LAYER_IDX, color='#D6604D', linestyle=':',
                alpha=0.3, linewidth=1.5)
    
    # --- Cosine Distance Box Plot ---
    cos_data = [df[col].values for col in cos_cols]
    
    bp2 = ax2.boxplot(cos_data, positions=range(len(LAYERS)), widths=0.5,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='.', markersize=2, alpha=0.3),
                      medianprops=dict(color='black', linewidth=1.5))
    
    for i, box in enumerate(bp2['boxes']):
        if i == INJECTION_LAYER_IDX:
            box.set_facecolor('#D6604D')
            box.set_alpha(0.7)
        else:
            box.set_facecolor('#B2182B')
            box.set_alpha(0.4)
    
    ax2.set_xticks(range(len(LAYERS)))
    ax2.set_xticklabels(LAYER_LABELS, fontsize=9)
    ax2.set_xlabel('VoxelNeXt Backbone Layers', fontsize=11)
    ax2.set_ylabel('Cosine Distance', fontsize=11)
    ax2.set_title('Cosine Distance Distribution', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax2.axvline(x=INJECTION_LAYER_IDX, color='#D6604D', linestyle=':',
                alpha=0.3, linewidth=1.5)
    
    fig.suptitle('Layer Sensitivity — Full Distribution (N=1000)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    out = OUTPUT_PATH / 'layerwise_sensitivity_BOXPLOT.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    
    # Print summary stats
    print(f"\n  Layer  |  MSE median  |  MSE IQR          |  CosDist median  |  CosDist IQR")
    print("  " + "-"*82)
    for i, layer in enumerate(LAYERS):
        mse_med = np.median(mse_data[i])
        mse_q1, mse_q3 = np.percentile(mse_data[i], [25, 75])
        cos_med = np.median(cos_data[i])
        cos_q1, cos_q3 = np.percentile(cos_data[i], [25, 75])
        marker = " <--" if i == INJECTION_LAYER_IDX else ""
        print(f"  {layer:5s}  |  {mse_med:.6f}  |  [{mse_q1:.4f}, {mse_q3:.4f}]  |"
              f"  {cos_med:.6f}      |  [{cos_q1:.4f}, {cos_q3:.4f}]{marker}")


# ============================================================
# MAIN
# ============================================================
def main():
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    df, mse_cols, cos_cols = load_and_extract(CSV_PATH)
    if df is None:
        return
    
    print("\n" + "="*60)
    print("PLOT 1: MEAN (all scenes)")
    print("="*60)
    plot_mean(df, mse_cols, cos_cols)
    
    print("\n" + "="*60)
    print(f"PLOT 2: TOP-{TOP_K} MEAN (most disrupted scenes)")
    print("="*60)
    plot_top_k_mean(df, mse_cols, cos_cols)
    
    print("\n" + "="*60)
    print("PLOT 3: BOX PLOT (full distribution)")
    print("="*60)
    plot_boxplot(df, mse_cols, cos_cols)
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED")
    print("="*60)


if __name__ == '__main__':
    main()