import torch
import numpy as np
import glob
import re
from pathlib import Path
from tqdm import tqdm
import pickle

# --- CONFIGURATION ---
# Folder containing your .pth files
DATA_DIR = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/recorded_features')
# 2. Path to Simulation Metadata (.pkl files) - Contains the Theta (x, y)
RAW_DATA_DIR = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/isaacsim/test/dataset_out_xy')
# Where to save the processed training data
OUTPUT_DIR = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/model')
OUTPUT_FILE = OUTPUT_DIR / 'surrogate_training_data_F4.pt'
# The specific layer we are targeting
LAYER_NAME = 'F4' 

def get_theta_from_pkl(scene_id):
    """
    Reads the original .pkl file to get the object position [x, y].
    Returns numpy array [x, y] or None if file missing.
    """
    # Matches format: sample_0005.pkl
    pkl_path = RAW_DATA_DIR / f"sample_{scene_id:04d}.pkl"
    
    if not pkl_path.exists():
        return None
        
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract X and Y (The physical parameters of the attack)
        # data['object_position'] is [x, y, z]
        pos = data['object_position']
        return np.array([pos[0], pos[1]], dtype=np.float32)
    except Exception as e:
        print(f"Error reading pickle {pkl_path}: {e}")
        return None
    
def to_numpy(data):
    """
    Safely converts Tensor or Numpy array to Numpy.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data # It's already numpy

def prepare_data():
    print(f"--- PREPARING SURROGATE DATA FOR LAYER {LAYER_NAME} ---")
    
    # 1. Load Baseline (Clean Scene)
    baseline_path = DATA_DIR / "recorded_sample_bg.pth"
    if not baseline_path.exists():
        print(f"CRITICAL ERROR: Baseline file not found at {baseline_path}")
        return
    
    print(f"Loading Baseline: {baseline_path.name}...")
    data_bg = torch.load(baseline_path, weights_only=False)
    
    # Extract Baseline Features & Indices (Move to CPU/Numpy)
    # We create a lookup map: coordinate tuple -> feature vector
    # bg_idx = data_bg['layers'][LAYER_NAME]['indices'].cpu().numpy()
    bg_idx = to_numpy(data_bg['layers'][LAYER_NAME]['indices'])
    # bg_feat = data_bg['layers'][LAYER_NAME]['features'].cpu().numpy()
    bg_feat = to_numpy(data_bg['layers'][LAYER_NAME]['features'])
    # Key is (y, x) or (z, y, x). We exclude batch dim (col 0).
    bg_map = {tuple(row[1:]): feat for row, feat in zip(bg_idx, bg_feat)}
    
    # 2. Find Perturbed Files
    files = sorted([f for f in glob.glob(str(DATA_DIR / "recorded_*.pth")) if 'sample_bg' not in f])
    print(f"Found {len(files)} perturbed scenes.")

    dataset_X = [] # Inputs (Theta)
    dataset_Y = [] # Targets (Delta Phi Vectors)

    # 3. Process Each Scene
    for f_path in tqdm(files, desc="Processing Scenes"):
        try:
            # Extract ID from filename (e.g., recorded_sample_0005.pth -> 5)
            match = re.search(r'(\d+)', Path(f_path).stem)
            if not match: continue
            scene_id = int(match.group(1))

            # --- Input: Theta ---
            # Extract angle/position from filename
            theta = get_theta_from_pkl(scene_id)
            
            # --- Target: Delta Phi ---
            data_pt = torch.load(f_path, weights_only=False)
            
            if LAYER_NAME not in data_pt['layers']:
                continue

            pt_idx = to_numpy(data_pt['layers'][LAYER_NAME]['indices'])
            pt_feat = to_numpy(data_pt['layers'][LAYER_NAME]['features'])
            
            # Create Lookup Map
            pt_map = {tuple(row[1:]): feat for row, feat in zip(pt_idx, pt_feat)}

            # ALIGNMENT: Find shared voxels (Intersection)
            common_coords = set(bg_map.keys()).intersection(set(pt_map.keys()))
            
            if len(common_coords) == 0:
                continue

            # Extract Vectors for Shared Voxels
            # shape: (N_shared, Channels)
            vec_bg = np.stack([bg_map[c] for c in common_coords])
            vec_pt = np.stack([pt_map[c] for c in common_coords])

            # COMPUTE DIFFERENCE (The "Feature Shock")
            # Delta = Perturbed - Baseline
            diffs = vec_pt - vec_bg
            
            # AGGREGATE: Mean Difference Vector
            # We condense the whole scene into one "average perturbation vector"
            # Shape: (Channels,) e.g., (128,)
            mean_delta = np.mean(diffs, axis=0)

            dataset_X.append(theta)
            dataset_Y.append(mean_delta)

        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    # 4. Save Dataset
    if len(dataset_X) > 0:
        # Convert list to Tensor
        # X shape: (N, 1) -> [[theta1], [theta2], ...]
        X_tensor = torch.tensor(dataset_X, dtype=torch.float32)
        # Y shape: (N, C) -> [[delta1...], [delta2...], ...]
        Y_tensor = torch.tensor(np.array(dataset_Y), dtype=torch.float32)

        save_dict = {
            'X': X_tensor, 
            'Y': Y_tensor, 
            'layer': LAYER_NAME,
            'description': 'X is (x,y) position, Y is Mean Delta Phi Vector'
        }

        # Create the folder if it doesn't exist
        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        torch.save(save_dict, OUTPUT_FILE)
        
        print(f"\nSUCCESS!")
        print(f"Saved {len(dataset_X)} samples to {OUTPUT_FILE}")
        print(f"Input Shape: {X_tensor.shape} (Theta)")
        print(f"Target Shape: {Y_tensor.shape} (Delta Phi)")
    else:
        print("No valid samples found.")

if __name__ == '__main__':
    prepare_data()