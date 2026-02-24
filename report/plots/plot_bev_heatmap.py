import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
# 1. Path to Baseline
BASELINE_PATH = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/recorded_features/recorded_sample_bg.pth')
# 2. Path to ONE Perturbed Scene (Pick a specific file you want to visualize)
PERTURBED_PATH = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/recorded_features/recorded_sample_0005.pth')
# 3. Output
OUTPUT_DIR = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/plots/plots_output')

def sparse_to_dense_bev(indices, features, grid_size=(200, 200)):
    """
    Converts sparse (N, 3) indices [batch, y, x] into a dense (H, W, C) grid.
    """

    # 1. Ensure inputs are Numpy
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    # Indices: [batch_idx, y, x]
    # We assume batch_size=1, so we take columns 1 (y) and 2 (x)
    ys = indices[:, 1].astype(np.int32)
    xs = indices[:, 2].astype(np.int32)
    
    # Auto-expand grid if indices go out of bounds (Safety Check)
    max_y, max_x = np.max(ys), np.max(xs)
    H = max(grid_size[0], max_y + 1)
    W = max(grid_size[1], max_x + 1)
    
    # Initialize Dense Grid (H, W, Channels)
    channels = features.shape[1]
    dense_grid = np.zeros((H, W, channels), dtype=np.float32)
    
    # Fill values
    # Note: If multiple sparse points hit the same dense pixel, last one wins.
    # (In VoxelNeXt BEV, indices are usually unique per pixel)
    dense_grid[ys, xs] = features
    
    return dense_grid

def generate_bev_heatmap():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Baseline: {BASELINE_PATH.name}")
    data_bg = torch.load(BASELINE_PATH, weights_only=False)
    
    print(f"Loading Perturbed: {PERTURBED_PATH.name}")
    data_pt = torch.load(PERTURBED_PATH, weights_only=False)

    # --- 1. Extract Shared Conv (The BEV Layer) ---
    layer_name = 'F8' # The final feature map before detection heads

    # Check if F8 exists in the dictionary keys
    if layer_name not in data_bg['layers']:
        print(f"Error: Later '{layer_name}' not found. Available layers: {list(data_bg['layers'].keys())}")
        return
        
    # Get Baseline Sparse Data
    idx_bg = data_bg['layers'][layer_name]['indices']
    feat_bg = data_bg['layers'][layer_name]['features']
    
    # Get Perturbed Sparse Data
    idx_pt = data_pt['layers'][layer_name]['indices']
    feat_pt = data_pt['layers'][layer_name]['features']

    # --- 2. Convert to Dense Grids ---
    # We determine the grid size dynamically based on the max index found in BG
    # (VoxelNeXt usually uses 1x downsampling or 8x. Let's assume max found)
    def get_max_coord(idx_data, col):
        if isinstance(idx_data, torch.Tensor):
            return torch.max(idx_data[:, col]).item()
        else: # It's numpy
            return np.max(idx_data[:, col])
    
    max_y = max(get_max_coord(idx_bg, 1), get_max_coord(idx_pt, 1))
    max_x = max(get_max_coord(idx_bg, 2), get_max_coord(idx_pt, 2))

    # Ensure grid shape is at least big enough (and convert to int)
    grid_shape = (max_y + 1, max_x + 1)

    print(f"Reconstructing Dense BEV Grid: {grid_shape}")
    
    dense_bg = sparse_to_dense_bev(idx_bg, feat_bg, grid_shape)
    dense_pt = sparse_to_dense_bev(idx_pt, feat_pt, grid_shape)

    # --- 3. Compute Feature Difference (Heatmap) ---
    # Formula: || phi(p) - phi(bg) ||^2
    
    # Ensure shapes match (pad if necessary)
    # Sometimes perturbed grid expands slightly if object pushes boundaries
    h_bg, w_bg, _ = dense_bg.shape
    h_pt, w_pt, _ = dense_pt.shape
    
    common_h = max(h_bg, h_pt)
    common_w = max(w_bg, w_pt)
    
    # Pad both to common size
    padded_bg = np.zeros((common_h, common_w, dense_bg.shape[2]))
    padded_pt = np.zeros((common_h, common_w, dense_pt.shape[2]))
    
    padded_bg[:h_bg, :w_bg, :] = dense_bg
    padded_pt[:h_pt, :w_pt, :] = dense_pt

    # Difference tensor
    diff_tensor = dense_pt - dense_bg 
    
    # Square differences
    squared_diff = np.square(diff_tensor)
    
    # Sum over channels to get a single scalar intensity per pixel
    # Shape becomes (H, W)
    heatmap = np.sum(squared_diff, axis=2)

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 10))
    
    # Use 'inferno' or 'hot' colormap for intensity
    # vmin=0 helps anchor the background to black
    plt.imshow(heatmap, cmap='inferno', origin='lower')
    
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('Feature Distortion Energy ($|| \Delta \phi ||^2$)', rotation=270, labelpad=20)
    
    plt.title(f'BEV Feature Difference Heatmap\nLayer: {layer_name}', fontsize=14)
    plt.xlabel('X Grid Coordinate')
    plt.ylabel('Y Grid Coordinate')
    
    save_file = OUTPUT_DIR / f'bev_heatmap_{PERTURBED_PATH.stem}.png'
    plt.savefig(save_file, dpi=300)
    print(f"Heatmap saved to: {save_file}")
    plt.show()

if __name__ == '__main__':
    generate_bev_heatmap()