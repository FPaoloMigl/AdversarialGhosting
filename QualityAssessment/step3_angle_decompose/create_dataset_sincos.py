import torch
import numpy as np
import glob
import re
from pathlib import Path
from tqdm import tqdm
import pickle

# --- CONFIGURATION ---
DATA_DIR = Path('AdversarialGhosting/QualityAssessment/step3/feature_extraction')
RAW_DATA_DIR = Path('AdversarialGhosting/QualityAssessment/step3/isaacsim/data')
OUTPUT_DIR = Path('AdversarialGhosting/QualityAssessment/step3_angle_decompose/model')
OUTPUT_FILE = OUTPUT_DIR / 'surrogate_training_data_F6_v3_sincos.pt'
LAYER_NAME = 'F6'
TOP_K = 100  # Number of most-affected voxels to average over


def get_theta_from_pkl(scene_id):
    """
    Reads the .pkl file and returns [x, y, sin(angle), cos(angle)].
    The sin/cos encoding gives the MLP a periodic, bounded representation
    of the angle instead of a raw scalar that invites monotonic fits.
    """
    pkl_path = RAW_DATA_DIR / f"sample_{scene_id:04d}.pkl"
    
    if not pkl_path.exists():
        return None
        
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        pos = data['object_position']
        angle_deg = data.get('object_z_rotation_deg', 0.0)
        
        # === SIN/COS ENCODING ===
        angle_rad = np.radians(angle_deg)
        
        # Return [x, y, sin(angle), cos(angle)] — 4D input
        return np.array([pos[0], pos[1], np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)

    except Exception as e:
        print(f"Error reading pickle {pkl_path}: {e}")
        return None


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data


def prepare_data():
    print(f"--- PREPARING SURROGATE DATA FOR LAYER {LAYER_NAME} ---")
    print(f"    Angle encoding: sin/cos (4D input)")
    print(f"    Aggregation: Top-{TOP_K} most affected voxels")
    
    # 1. Load Baseline
    baseline_path = DATA_DIR / "recorded_sample_bg.pth"
    if not baseline_path.exists():
        print(f"CRITICAL ERROR: Baseline file not found at {baseline_path}")
        return
    
    print(f"Loading Baseline: {baseline_path.name}...")
    data_bg = torch.load(baseline_path, weights_only=False)
    
    bg_idx = to_numpy(data_bg['layers'][LAYER_NAME]['indices'])
    bg_feat = to_numpy(data_bg['layers'][LAYER_NAME]['features'])
    bg_map = {tuple(row[1:]): feat for row, feat in zip(bg_idx, bg_feat)}
    
    # 2. Find Perturbed Files
    files = sorted([f for f in glob.glob(str(DATA_DIR / "recorded_*.pth")) if 'sample_bg' not in f])
    print(f"Found {len(files)} perturbed scenes.")

    dataset_X = []
    dataset_Y = []

    # 3. Process Each Scene
    for f_path in tqdm(files, desc="Processing Scenes"):
        try:
            match = re.search(r'(\d+)', Path(f_path).stem)
            if not match:
                continue
            scene_id = int(match.group(1))

            # --- Input: Theta [x, y, sin(a), cos(a)] ---
            theta = get_theta_from_pkl(scene_id)
            if theta is None:
                continue

            # --- Target: Delta Phi ---
            data_pt = torch.load(f_path, weights_only=False)
            
            if LAYER_NAME not in data_pt['layers']:
                continue

            pt_idx = to_numpy(data_pt['layers'][LAYER_NAME]['indices'])
            pt_feat = to_numpy(data_pt['layers'][LAYER_NAME]['features'])
            pt_map = {tuple(row[1:]): feat for row, feat in zip(pt_idx, pt_feat)}

            # Alignment: shared voxels
            common_coords = set(bg_map.keys()).intersection(set(pt_map.keys()))
            
            if len(common_coords) == 0:
                continue

            vec_bg = np.stack([bg_map[c] for c in common_coords])
            vec_pt = np.stack([pt_map[c] for c in common_coords])

            # Delta = Perturbed - Baseline
            diffs = vec_pt - vec_bg
            
            # === TOP-K DILUTION FIX ===
            # Instead of averaging ALL voxels (which dilutes the signal),
            # only average the K most-affected voxels (by L2 norm of their delta)
            magnitudes = np.linalg.norm(diffs, axis=1)
            k = min(TOP_K, len(magnitudes))
            top_k_indices = np.argpartition(magnitudes, -k)[-k:]
            
            mean_delta = np.mean(diffs[top_k_indices], axis=0)

            dataset_X.append(theta)
            dataset_Y.append(mean_delta)

        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    # 4. Save Dataset
    if len(dataset_X) > 0:
        X_tensor = torch.tensor(dataset_X, dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(dataset_Y), dtype=torch.float32)

        # --- NORMALIZATION ---
        # X is now [x, y, sin(a), cos(a)]
        # x and y need min-max normalization
        # sin and cos are already in [-1, 1] — no normalization needed
        
        # Normalize x and y to [-1, 1]
        xy_bounds_min = torch.tensor([-30.0, -30.0], dtype=torch.float32)
        xy_bounds_max = torch.tensor([30.0, 30.0], dtype=torch.float32)
        
        X_xy = X_tensor[:, :2]
        X_sincos = X_tensor[:, 2:]  # Already [-1, 1]
        
        X_xy_norm = (X_xy - xy_bounds_min) / (xy_bounds_max - xy_bounds_min)  # [0, 1]
        X_xy_norm = X_xy_norm * 2.0 - 1.0  # [-1, 1]
        
        X_norm = torch.cat([X_xy_norm, X_sincos], dim=1)  # [N, 4] all in [-1, 1]

        save_dict = {
            'X': X_norm,
            'Y': Y_tensor,
            'layer': LAYER_NAME,
            'description': 'X is [x_norm, y_norm, sin(angle), cos(angle)] all in [-1,1]. Y is Top-K Mean Delta Phi.',
            'xy_bounds_min': xy_bounds_min,
            'xy_bounds_max': xy_bounds_max,
            'top_k': TOP_K,
            'encoding': 'sincos'
        }

        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        torch.save(save_dict, OUTPUT_FILE)
        
        print(f"\nSUCCESS!")
        print(f"Saved {len(dataset_X)} samples to {OUTPUT_FILE}")
        print(f"Input Shape:  {X_norm.shape} — [x, y, sin(a), cos(a)]")
        print(f"Target Shape: {Y_tensor.shape} — Delta Phi (Top-{TOP_K})")
        
        # Sanity check
        print(f"\nSanity Check (first sample):")
        print(f"  X_norm: {X_norm[0]}")
        print(f"  Y norm: {Y_tensor[0].norm():.4f}")
    else:
        print("No valid samples found.")


if __name__ == '__main__':
    prepare_data()