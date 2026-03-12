import torch
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIG ---
RECORDED_DIR = '/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/recorded_features/'
BASELINE_FILE = 'recorded_sample_bg.pth' 

# UPDATED: Added F7 (BEV) and F8 (Shared/Decision)
LAYERS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
LAYER_LABELS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'BEV_Out', 'Shared']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def get_cosine_dist(feat1, feat2):
    n1 = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
    n2 = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)
    return 1 - np.sum(n1 * n2, axis=1)

def align_and_compare(base_layer, curr_layer, k=1000):
    """Handles both sparse 3D voxels and dense 2D feature maps."""
    # print(f"k = {k}")
    # --- DENSE LAYER FALLBACK (F7/F8) ---
    if 'indices' not in base_layer:
        # Expected shape: (Batch, Channels, Height, Width)
        # Flatten (C, H, W) into (Pixels, Channels)
        f_base = base_layer['features'].transpose(0, 2, 3, 1).reshape(-1, base_layer['features'].shape[1])
        f_curr = curr_layer['features'].transpose(0, 2, 3, 1).reshape(-1, curr_layer['features'].shape[1])
        
        dist = get_cosine_dist(f_base, f_curr)
        
    # --- SPARSE LAYER LOGIC (F1-F6) ---
    else:
        def fast_hash(idx):
            idx = idx.astype(np.int64) # Prevent overflow error
            if idx.shape[1] == 4:
                return idx[:, 1] * 10**10 + idx[:, 2] * 10**5 + idx[:, 3]
            return idx[:, 1] * 10**5 + idx[:, 2]

        base_h = fast_hash(base_layer['indices'])
        curr_h = fast_hash(curr_layer['indices'])
        common, b_mask, c_mask = np.intersect1d(base_h, curr_h, return_indices=True)
        
        if len(common) == 0: return 0.0
        dist = get_cosine_dist(base_layer['features'][b_mask], curr_layer['features'][c_mask])
    
    return np.mean(dist)

# --- DATA COLLECTION ---
baseline_path = os.path.join(RECORDED_DIR, BASELINE_FILE)
baseline_data = torch.load(baseline_path, weights_only=False)

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

files = sorted([f for f in glob.glob(os.path.join(RECORDED_DIR, "recorded_*.pth")) 
                if BASELINE_FILE not in f], key=natural_key)

results = {layer: [] for layer in LAYERS}

print(f"Analyzing {len(files)} randomized scenes...")
for f_path in tqdm(files):
    data = torch.load(f_path, weights_only=False)
    for layer in LAYERS:
        if layer in data['layers']:
            val = align_and_compare(baseline_data['layers'][layer], data['layers'][layer])
            results[layer].append(val)


# --- PLOTTING ---
plt.figure(figsize=(15, 8))
x_axis = np.arange(len(files))

for i, layer in enumerate(LAYERS):
    # Use Scatter for independent samples
    # plt.scatter(x_axis, results[layer], label=LAYER_LABELS[i], color=COLORS[i], s=20, alpha=0.5)
    # Faint trend line to guide the eye
    plt.plot(x_axis, results[layer], label=layer, color=COLORS[i], alpha=0.9, linewidth=2)

plt.title('Adversarial Perturbation Sensitivity per Backbone Layer', fontsize=16)
plt.xlabel('Adversarial Position (Scene Index)', fontsize=12)
plt.ylabel('Mean Cosine Distance (Perturbation Shift)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Network Layers", loc='upper left', ncol=2)

# Save high-res version for your paper
plt.savefig('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/plots/plots_output/adversarial_line_chart_v1.png', dpi=300)
print("Line chart saved as adversarial_line_chart.png")