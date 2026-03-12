import torch
import numpy as np
import glob
import os
from pathlib import Path
from tqdm import tqdm # Optional: for a nice progress bar
import pandas as pd # Use pandas for easy CSV saving

# --- CONFIGURATION ---
IOU_THRESH = 0.1  # Threshold to define 'existence' or 'match'
RECORD_DIR = Path('/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/recorded_features/v2')
OUTPUT_CSV = '/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/report/v2/adversarial_analysis_results.csv'

# ==========================================
# PART 1: GEOMETRIC UTILS (IoU & Box Matching)
# ==========================================
def calculate_iou_3d(box_a, box_b):
    """
    Computes 3D IoU (AABB approximation).
    Args:
        box_a, box_b: [x, y, z, dx, dy, dz, heading]
    """
    # Helper to get min/max corners
    def get_min_max(b):
        return [b[0]-b[3]/2, b[1]-b[4]/2, b[2]-b[5]/2, 
                b[0]+b[3]/2, b[1]+b[4]/2, b[2]+b[5]/2]

    a, b = get_min_max(box_a), get_min_max(box_b)
    
    # Intersection
    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:], b[3:])
    dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(dims)
    
    if inter_vol == 0: return 0.0

    # Union
    vol_a, vol_b = np.prod(box_a[3:6]), np.prod(box_b[3:6])
    return inter_vol / (vol_a + vol_b - inter_vol)

def compute_detection_metrics(c_boxes, c_scores, p_boxes, p_scores):
    """
    Compares Baseline (c) vs Perturbed (p) detections.
    Returns: Suppression, Ghosting, BoxShift, ConfChange
    """

    sup_count = 0
    ghost_count = 0
    shifts = []
    conf_changes = []

    # 1. Check Baseline -> Perturbed (Did objects disappear or move?)
    for i, c_box in enumerate(c_boxes):
        max_iou = 0
        match_idx = -1
        
        # Find best match in perturbed scene
        for j, p_box in enumerate(p_boxes):
            iou = calculate_iou_3d(c_box, p_box)
            if iou > max_iou:
                max_iou = iou
                match_idx = j
        
        if max_iou < IOU_THRESH:
            sup_count += 1  # Erasure
        else:
            # Box Shift (Euclidean distance of centers)
            dist = np.linalg.norm(c_box[:3] - p_boxes[match_idx][:3])
            shifts.append(dist)
            
            # Confidence Change (Perturbed - Clean)
            conf_diff = p_scores[match_idx] - c_scores[i]
            conf_changes.append(conf_diff)

    # 2. Check Perturbed -> Baseline (Did new objects appear?)
    for p_box in p_boxes:
        # If p_box matches NOTHING in c_boxes, it's a Ghost
        has_match = False
        for c_box in c_boxes:
            if calculate_iou_3d(p_box, c_box) >= IOU_THRESH:
                has_match = True
                break
        if not has_match:
            ghost_count += 1

    return {
        'Suppression': sup_count,
        'Ghosting': ghost_count,
        'BoxShift': np.mean(shifts) if shifts else 0.0,
        'ConfChange': np.mean(conf_changes) if conf_changes else 0.0
    }

# ==========================================
# PART 2: NEURAL UTILS (Alignment & Distance)
# ==========================================
def compute_layer_metrics(bg_data, pert_data, layer_name):
    """
    Aligns sparse voxels and computes MSE & Cosine Distance.
    Handles shape mismatch (e.g., 137k vs 132k).
    """
    # 1. Extract Indices (coordinates) and Features
    # Convert to CPU numpy for processing
    bg_idx = bg_data['indices']
    bg_feat = bg_data['features']
    pt_idx = pert_data['indices']
    pt_feat = pert_data['features']

    # 2. Map Coordinates to Features
    # Key: tuple(z, y, x) or tuple(y, x). We skip col 0 (batch_idx).
    bg_map = {tuple(idx[1:]): feat for idx, feat in zip(bg_idx, bg_feat)}
    pt_map = {tuple(idx[1:]): feat for idx, feat in zip(pt_idx, pt_feat)}

    # 3. Find Shared Voxels (Intersection)
    common_coords = set(bg_map.keys()).intersection(set(pt_map.keys()))
    
    if len(common_coords) == 0:
        return 0.0, 0.0  # No overlap = cannot compute

    # 4. Create Aligned Arrays (N_shared, C)
    # Stack features in the same order
    vec_a = np.stack([bg_map[c] for c in common_coords])
    vec_b = np.stack([pt_map[c] for c in common_coords])

    # --- METRIC A: Feature MSE ---
    # MSE = (1/Channels) * mean(||A - B||^2)
    d_l = vec_a.shape[1]
    diff = vec_a - vec_b
    mse = np.mean(np.sum(np.square(diff), axis=1) / d_l)

    # --- METRIC B: Cosine Distance ---
    # Dist = 1 - (A . B) / (||A|| * ||B||)
    dot = np.sum(vec_a * vec_b, axis=1)
    norms = np.linalg.norm(vec_a, axis=1) * np.linalg.norm(vec_b, axis=1)
    
    # Add epsilon to prevent div/0
    similarity = dot / (norms + 1e-8)
    similarity = np.clip(similarity, -1.0, 1.0)
    cos_dist = np.mean(1.0 - similarity)

    return mse, cos_dist

# ==========================================
# PART 3: MAIN EXECUTION LOOP
# ==========================================
def main():
    # 1. Load Baseline
    baseline_path = RECORD_DIR / "recorded_sample_bg.pth"
    if not baseline_path.exists():
        print(f"CRITICAL ERROR: Baseline file not found at {baseline_path}")
        return
    
    print(f"Loading Baseline: {baseline_path.name}...")
    clean_data = torch.load(baseline_path, weights_only=False) # Fix for torch 2.6

    # Move baseline data to Numpy/CPU once to save time
    c_boxes = clean_data['detections']['boxes']
    c_scores = clean_data['detections']['scores']

    # Convert baseline layers to CPU numpy (for alignment speed)
    clean_layers_cpu = {}
    for layer_key, layer_val in clean_data['layers'].items():
        clean_layers_cpu[layer_key] = {
            'indices': layer_val['indices'],
            'features': layer_val['features']
        }

    # 2. Find Perturbed Files
    files = sorted([f for f in glob.glob(str(RECORD_DIR / "recorded_*.pth")) if 'sample_bg' not in f])
    print(f"Found {len(files)} adversarial scenes to analyze.")

    results = []

    # 3. Iterate
    for f_path in tqdm(files, desc="Analyzing Scenes"):
        try:
            # Load specific scene
            data = torch.load(f_path, weights_only=False)
            # Extract Scene ID / Theta (Position) from filename
            # Assuming filename is like "recorded_000005.pth" or "recorded_theta_0.5.pth"
            filename = Path(f_path).stem

            # --- A. Detection Metrics ---
            p_boxes = data['detections']['boxes']
            p_scores = data['detections']['scores']

            det_metrics = compute_detection_metrics(c_boxes, c_scores, p_boxes, p_scores)

            # --- B. Neural Metrics (Per Layer) ---
            neural_metrics = {}
            for layer_name in clean_layers_cpu.keys():
                # Get the perturbed layer data
                if layer_name in data['layers']:
                    pert_layer = {
                        'indices': data['layers'][layer_name]['indices'],
                        'features': data['layers'][layer_name]['features']
                    }

                    mse, cos = compute_layer_metrics(
                        clean_layers_cpu[layer_name], 
                        pert_layer, 
                        layer_name
                    )

                    neural_metrics[f"{layer_name}_MSE"] = mse
                    neural_metrics[f"{layer_name}_CosDist"] = cos

            # --- C. Aggregate Result Row ---
            row = {
                'Scene_ID': filename,
                **det_metrics,       # Unpack Sup, Ghost, Shift, Conf
                **neural_metrics     # Unpack F1_MSE, F1_Cos, F6_MSE...
            }
            results.append(row)

        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    # 4. Save Final Report
    df = pd.DataFrame(results)
    save_path = OUTPUT_CSV
    df.to_csv(save_path, index=False)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {save_path}")
    print("="*50)
    print(df.head()) # Show preview

if __name__ == '__main__':
    main()