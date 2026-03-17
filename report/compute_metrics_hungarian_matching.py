import torch
import numpy as np
import glob
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- CONFIGURATION ---
IOU_THRESH = 0.5  # Standard threshold for vehicles (NuScenes/KITTI default)
RECORD_DIR = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/feature_extraction')
OUTPUT_DIR = '/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/feature_extraction/metrics'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ==========================================
# PART 1: GEOMETRIC UTILS (IoU & Hungarian Matching)
# ==========================================
def calculate_iou_3d(box_a, box_b):
    """
    Computes 3D IoU (AABB approximation).
    Args:
        box_a, box_b: [x, y, z, dx, dy, dz, heading]
    """
    def get_min_max(b):
        return [b[0]-b[3]/2, b[1]-b[4]/2, b[2]-b[5]/2, 
                b[0]+b[3]/2, b[1]+b[4]/2, b[2]+b[5]/2]

    a, b = get_min_max(box_a), get_min_max(box_b)
    
    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:], b[3:])
    dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(dims)
    
    if inter_vol == 0: 
        return 0.0

    vol_a, vol_b = np.prod(box_a[3:6]), np.prod(box_b[3:6])
    return inter_vol / (vol_a + vol_b - inter_vol)


def build_iou_matrix(c_boxes, p_boxes):
    """
    Builds a full IoU matrix between clean and perturbed boxes.
    Shape: (N_clean, N_perturbed)
    """
    n_clean = len(c_boxes)
    n_pert = len(p_boxes)
    iou_matrix = np.zeros((n_clean, n_pert))
    
    for i in range(n_clean):
        for j in range(n_pert):
            iou_matrix[i, j] = calculate_iou_3d(c_boxes[i], p_boxes[j])
    
    return iou_matrix


def compute_detection_metrics(c_boxes, c_scores, p_boxes, p_scores, iou_thresh=0.5):
    """
    Compares Baseline (c) vs Perturbed (p) detections using Hungarian matching.
    
    Returns:
        Suppression: clean detections with no match (missing)
        Ghosting:    perturbed detections with no match (hallucinated)
        BoxShift:    mean L2 center distance for matched pairs
        ConfChange:  mean confidence change (perturbed - clean) for matched pairs
    """
    n_clean = len(c_boxes)
    n_pert = len(p_boxes)
    
    # Edge cases: one side is empty
    if n_clean == 0 and n_pert == 0:
        return {'Suppression': 0, 'Ghosting': 0, 'BoxShift': 0.0, 'ConfChange': 0.0}
    if n_clean == 0:
        return {'Suppression': 0, 'Ghosting': n_pert, 'BoxShift': 0.0, 'ConfChange': 0.0}
    if n_pert == 0:
        return {'Suppression': n_clean, 'Ghosting': 0, 'BoxShift': 0.0, 'ConfChange': 0.0}

    # 1. Build full IoU matrix (N_clean x N_perturbed)
    iou_matrix = build_iou_matrix(c_boxes, p_boxes)
    
    # 2. Hungarian matching (finds optimal one-to-one assignment)
    #    linear_sum_assignment minimizes cost, so we pass negative IoU
    cost_matrix = -iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 3. Filter matches by IoU threshold
    sup_count = 0
    ghost_count = 0
    shifts = []
    conf_changes = []
    
    matched_clean = set()
    matched_pert = set()
    
    for r, c in zip(row_indices, col_indices):
        if iou_matrix[r, c] >= iou_thresh:
            # Valid match
            matched_clean.add(r)
            matched_pert.add(c)
            
            # Box Shift: L2 distance between centers
            dist = np.linalg.norm(c_boxes[r][:3] - p_boxes[c][:3])
            shifts.append(dist)
            
            # Confidence Change: perturbed - clean
            conf_diff = p_scores[c] - c_scores[r]
            conf_changes.append(conf_diff)
    
    # 4. Count unmatched
    sup_count = n_clean - len(matched_clean)    # Missing detections
    ghost_count = n_pert - len(matched_pert)     # Ghost detections
    
    return {
        'Suppression': sup_count,
        'Ghosting': ghost_count,
        'BoxShift': np.mean(shifts) if shifts else 0.0,
        'ConfChange': np.mean(conf_changes) if conf_changes else 0.0
    }


# ==========================================
# PART 2: NEURAL UTILS (Alignment & Distance)
# ==========================================
def compute_layer_metrics(bg_map, pert_data, layer_name):
    """
    Aligns sparse voxels and computes MSE & Cosine Distance.
    """
    pt_idx = pert_data['indices']
    pt_feat = pert_data['features']

    pt_map = {tuple(idx[1:]): feat for idx, feat in zip(pt_idx, pt_feat)}

    common_coords = set(bg_map.keys()).intersection(set(pt_map.keys()))
    
    if len(common_coords) == 0:
        return 0.0, 0.0

    vec_a = np.stack([bg_map[c] for c in common_coords])
    vec_b = np.stack([pt_map[c] for c in common_coords])

    # Feature MSE
    d_l = vec_a.shape[1]
    diff = vec_a - vec_b
    mse = np.mean(np.sum(np.square(diff), axis=1) / d_l)

    # Cosine Distance
    dot = np.sum(vec_a * vec_b, axis=1)
    norms = np.linalg.norm(vec_a, axis=1) * np.linalg.norm(vec_b, axis=1)
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
    clean_data = torch.load(baseline_path, weights_only=False)

    c_boxes = clean_data['detections']['boxes']
    c_scores = clean_data['detections']['scores']

    # Pre-build baseline layer maps (coordinate -> feature vector)
    clean_layers_maps = {}
    for layer_key, layer_val in clean_data['layers'].items():
        clean_layers_maps[layer_key] = {
            tuple(idx[1:]): feat 
            for idx, feat in zip(layer_val['indices'], layer_val['features'])
        }

    # 2. Find Perturbed Files
    files = sorted([f for f in glob.glob(str(RECORD_DIR / "recorded_*.pth")) if 'sample_bg' not in f])
    print(f"Found {len(files)} adversarial scenes to analyze.")
    print(f"Using IoU threshold: {IOU_THRESH}")
    print(f"Matching method: Hungarian (optimal one-to-one)\n")

    results = []

    # 3. Iterate
    for f_path in tqdm(files, desc="Analyzing Scenes"):
        try:
            data = torch.load(f_path, weights_only=False)
            filename = Path(f_path).stem

            # --- A. Detection Metrics (Hungarian Matching) ---
            p_boxes = data['detections']['boxes']
            p_scores = data['detections']['scores']

            det_metrics = compute_detection_metrics(
                c_boxes, c_scores, p_boxes, p_scores, 
                iou_thresh=IOU_THRESH
            )

            # --- B. Neural Metrics (Per Layer) ---
            neural_metrics = {}
            for layer_name in clean_layers_maps.keys():
                if layer_name in data['layers']:
                    pert_layer = {
                        'indices': data['layers'][layer_name]['indices'],
                        'features': data['layers'][layer_name]['features']
                    }

                    mse, cos = compute_layer_metrics(
                        clean_layers_maps[layer_name], 
                        pert_layer, 
                        layer_name
                    )

                    neural_metrics[f"{layer_name}_MSE"] = mse
                    neural_metrics[f"{layer_name}_CosDist"] = cos

            # --- C. Aggregate Result Row ---
            row = {
                'Scene_ID': filename,
                **det_metrics,
                **neural_metrics
            }
            results.append(row)

        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    # 4. Save Final Report
    df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, "adversarial_analysis_results.csv") 
    df.to_csv(save_path, index=False)

    # 5. Print Summary Statistics
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {save_path}")
    print(f"Total scenes analyzed: {len(df)}")
    print(f"\n--- Detection Metric Averages (IoU threshold = {IOU_THRESH}) ---")
    print(f"  Avg Suppressions:     {df['Suppression'].mean():.2f}")
    print(f"  Avg Ghosts:           {df['Ghosting'].mean():.2f}")
    print(f"  Avg Box Shift:        {df['BoxShift'].mean():.4f} m")
    print(f"  Avg Confidence Change: {df['ConfChange'].mean():.4f}")
    print(f"\nPreview:")
    print(df.head())


if __name__ == '__main__':
    main()