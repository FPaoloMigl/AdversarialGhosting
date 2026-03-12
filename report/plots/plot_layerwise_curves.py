import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# Path to the CSV you just generated
CSV_PATH = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/v2/adversarial_analysis_results.csv')
OUTPUT_PATH = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/plots_output')
OUTPUT_PLOT = OUTPUT_PATH / 'layerwise_sensitivity_v2.png'

def plot_layerwise_metrics():
    if not OUTPUT_PATH.exists():
        print(f"Creating directory: {OUTPUT_PATH}")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} scenes.")

    # 2. Define Layers and Metric Columns
    layers = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
    # Create readable labels for the X-axis
    layer_labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'BEV_Out', 'Shared']

    # 3. Calculate Averages per Layer
    avg_mse = []
    avg_cos = []

    print("\n--- Layerwise Averages ---")
    for layer in layers:
        # Construct column names based on your CSV format
        mse_col = f"{layer}_MSE"
        cos_col = f"{layer}_CosDist" # Note: Ensure this matches your CSV header exactly
        
        # Check if columns exist to avoid errors
        if mse_col in df.columns and cos_col in df.columns:
            m = df[mse_col].mean()
            c = df[cos_col].mean()
            avg_mse.append(m)
            avg_cos.append(c)
            print(f"{layer}: MSE={m:.4f}, Cos={c:.4f}")
        else:
            print(f"Warning: Columns for {layer} not found. Filling with 0.")
            avg_mse.append(0)
            avg_cos.append(0)

    # 4. Plotting (Dual Axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X-Axis Setup
    x = np.arange(len(layers))
    
    # Plot MSE (Left Axis) - Blue Bars
    bars = ax1.bar(x, avg_mse, color='royalblue', alpha=0.6, label='Feature MSE (Magnitude)')
    ax1.set_xlabel('VoxelNeXt Layers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Squared Error (MSE)', color='royalblue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels, rotation=0) # No rotation needed usually
    
    # Create Right Axis for Cosine Distance
    ax2 = ax1.twinx()
    
    # Plot Cosine Distance (Right Axis) - Red Line
    line = ax2.plot(x, avg_cos, color='crimson', marker='o', linewidth=2, markersize=8, label='Cosine Distance (Semantic Shift)')
    ax2.set_ylabel('Cosine Distance', color='crimson', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, 1.0) # Cosine distance is always between 0 and 1 (or 2)

    # 5. Finishing Touches
    plt.title('Adversarial Layer Sensitivity: Magnitude vs. Semantic Shift', fontsize=14, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Combine Legends
    # (Matplotlib dual-axis legends are tricky, this manually combines them)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\nPlot saved to: {OUTPUT_PLOT}")
    plt.show()

if __name__ == '__main__':
    plot_layerwise_metrics()