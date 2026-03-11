import torch
import numpy as np
import glob
import re
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- CONFIGURATION ---
DATA_DIR = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/recorded_features')
RAW_DATA_DIR = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/dataset_out_xy')
OUTPUT_DIR = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/model')

# Ensure we are targeting the 3D layer
LAYER_NAME = 'F4' 

# Grid Constants for NuScenes / VoxelNeXt
STRIDE = 8
VOXEL_SIZE = 0.075
PC_RANGE = np.array([-51.2, -51.2]) # NuScenes X and Y start

def get_theta_from_pkl(scene_id):
    pkl_path = RAW_DATA_DIR / f"sample_{scene_id:04d}.pkl"
    if not pkl_path.exists(): return None
    try:
        with open(pkl_path, 'rb') as f: data = pickle.load(f)
        pos = data['object_position']
        return np.array([pos[0], pos[1]], dtype=np.float32)
    except: return None
    
def to_numpy(data):
    if isinstance(data, torch.Tensor): return data.cpu().numpy()
    return data

def gaussian(d, A, sigma):    
    return A * np.exp(-d**2 / (2 * sigma**2))

def fit_spatial_decay():
    print(f"--- FITTING SPATIAL DECAY FOR LAYER {LAYER_NAME} ---")
    
    baseline_path = DATA_DIR / "recorded_sample_bg.pth"
    data_bg = torch.load(baseline_path, weights_only=False)
    
    bg_idx = to_numpy(data_bg['layers'][LAYER_NAME]['indices'])
    bg_feat = to_numpy(data_bg['layers'][LAYER_NAME]['features'])
    
    # Map for conv4: key is (z, y, x)
    bg_map = {tuple(row[1:]): feat for row, feat in zip(bg_idx, bg_feat)}
    
    files = sorted([f for f in glob.glob(str(DATA_DIR / "recorded_*.pth")) if 'sample_bg' not in f])
    print(f"Found {len(files)} perturbed scenes.")

    all_distances = []
    all_magnitudes = []

    # --- PROCESS EACH SCENE ---
    for f_path in tqdm(files, desc="Measuring Shockwaves"):
        try:
            match = re.search(r'(\d+)', Path(f_path).stem)
            if not match: continue
            scene_id = int(match.group(1))

            theta = get_theta_from_pkl(scene_id)
            if theta is None: continue
            
            data_pt = torch.load(f_path, weights_only=False)
            if LAYER_NAME not in data_pt['layers']: continue

            pt_idx = to_numpy(data_pt['layers'][LAYER_NAME]['indices'])
            pt_feat = to_numpy(data_pt['layers'][LAYER_NAME]['features'])
            pt_map = {tuple(row[1:]): feat for row, feat in zip(pt_idx, pt_feat)}

            common_coords = set(bg_map.keys()).intersection(set(pt_map.keys()))
            if len(common_coords) == 0: continue

            vec_bg = np.stack([bg_map[c] for c in common_coords])
            vec_pt = np.stack([pt_map[c] for c in common_coords])
            diffs = vec_pt - vec_bg
            
            # --- NEW SPATIAL DECAY MEASUREMENT ---
            shared_coords = np.array([list(c) for c in common_coords])  # [N, 3] -> [z, y, x]
            
            # 1. Convert physical pole position to Grid Coordinates
            theta_vx = (theta[0] - PC_RANGE[0]) / (VOXEL_SIZE * STRIDE)
            theta_vy = (theta[1] - PC_RANGE[1]) / (VOXEL_SIZE * STRIDE)
            
            # 2. Calculate Distance from pole to every voxel
            dx = shared_coords[:, 2] - theta_vx   # x is column 2
            dy = shared_coords[:, 1] - theta_vy   # y is column 1
            distances = np.sqrt(dx**2 + dy**2)
            
            # 3. Calculate Feature Shock Magnitude
            magnitudes = np.linalg.norm(diffs, axis=1)
            
            all_distances.append(distances)
            all_magnitudes.append(magnitudes)
            # -------------------------------------

        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    # --- CURVE FITTING & PLOTTING ---
    print("\n--- Fitting Gaussian Curve ---")
    all_dist = np.concatenate(all_distances)
    all_mag = np.concatenate(all_magnitudes)

    # Bin the data to smooth out the noise
    bins = np.linspace(0, 150, 50)
    bin_indices = np.digitize(all_dist, bins)
    
    bin_means = [all_mag[bin_indices == i].mean() for i in range(1, len(bins)) if (bin_indices == i).sum() > 0]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1) if (bin_indices == i+1).sum() > 0]

    # Fit the Gaussian
    try:
        popt, _ = curve_fit(gaussian, bin_centers, bin_means, p0=[5.0, 20.0])
        fitted_A = popt[0]
        fitted_sigma = popt[1]
        physical_meters = fitted_sigma * STRIDE * VOXEL_SIZE
        
        print(f"✅ FIT SUCCESSFUL!")
        print(f"Peak Amplitude (A): {fitted_A:.2f}")
        print(f"Grid Sigma:         {fitted_sigma:.2f} pixels")
        print(f"Physical Radius:    {physical_meters:.2f} meters")
        
        # Plotting 
        plt.figure(figsize=(8,5))
        plt.scatter(bin_centers, bin_means, s=40, alpha=0.7, color='blue', label='Empirical Isaac Sim Data')
        d_fit = np.linspace(0, max(bin_centers), 200)
        plt.plot(d_fit, gaussian(d_fit, *popt), 'r-', linewidth=3, label=f'Fitted Curve (σ={fitted_sigma:.1f})')
        
        plt.title('Physical Reach of Adversarial LiDAR Multipath')
        plt.xlabel('Distance from Reflector (Grid Pixels)')
        plt.ylabel('Mean Feature Perturbation Magnitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = OUTPUT_DIR / 'spatial_decay_fit.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Curve fitting failed. The data might be too noisy. Error: {e}")

if __name__ == '__main__':
    fit_spatial_decay()