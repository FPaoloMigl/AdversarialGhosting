import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
CSV_PATH = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/adversarial_analysis_results.csv')
OUTPUT_DIR = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/plots/plots_output')
OUTPUT_FILE = OUTPUT_DIR / 'mse_vs_detection_boxplot.png'

def plot_mse_boxplots():
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. Compute Delta N
    #   Negative = Suppression (Loss)
    #   Positive = Ghosting (Gain)
    df['Delta_N'] = df['Ghosting'] - df['Suppression']

    # 2. Filter for Clean Plotting
    # Optional: If you have extreme outliers (e.g. +5 ghosts), you might want to clip or bin them
    # For now, we plot raw values.

    # 3. Setup Plot
    layers = ['F1', 'F4', 'F6', 'shared_conv']
    layer_labels = ['Layer F1', 'Layer F4', 'Layer F6', 'Layer F8 (BEV)']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=False)
    
    # Custom Palette: Red for Negatives, Gray for Zero, Blue for Positives
    unique_vals = sorted(df['Delta_N'].unique())
    palette = {val: 'crimson' if val < 0 else 'royalblue' if val > 0 else 'lightgray' for val in unique_vals}

    for i, layer in enumerate(layers):
        ax = axes[i]
        mse_col = f"{layer}_MSE"
        
        if mse_col not in df.columns: continue

        # --- THE BOX PLOT ---
        sns.boxplot(
            data=df, 
            x='Delta_N', 
            y=mse_col, 
            ax=ax, 
            palette=palette,
            showfliers=False, # Hides extreme outliers to keep scale readable
            width=0.5
        )
        
        # --- Optional: Overlay Scatter Points (Strip Plot) ---
        # This adds the raw data points on top of the box so you see the sample size
        sns.stripplot(
            data=df,
            x='Delta_N',
            y=mse_col,
            ax=ax,
            color='black',
            alpha=0.3,
            size=3,
            jitter=True
        )

        # Formatting
        ax.set_title(layer_labels[i], fontsize=14, fontweight='bold')
        ax.set_xlabel(r'Detection Change ($\Delta N$)', fontsize=12)
        
        if i == 0:
            ax.set_ylabel('Feature MSE (Magnitude)', fontsize=12)
        else:
            ax.set_ylabel('')
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Statistical Impact: Feature Error vs. Detection Failure', fontsize=16, y=1.02)
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    plot_mse_boxplots()