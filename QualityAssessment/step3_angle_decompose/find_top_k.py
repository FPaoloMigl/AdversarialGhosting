import torch
import numpy as np
import glob
import re
from pathlib import Path
import pickle

# --- CONFIGURATION ---
DATA_DIR = Path('AdversarialGhosting/QualityAssessment/step3/feature_extraction')
RAW_DATA_DIR = Path('AdversarialGhosting/QualityAssessment/step3/isaacsim/data')
LAYER_NAME = 'F6'
NUM_SCENES_TO_SAMPLE = 10  # How many scenes to inspect


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data


def get_position_from_pkl(scene_id):
    pkl_path = RAW_DATA_DIR / f"sample_{scene_id:04d}.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        pos = data['object_position']
        angle = data.get('object_z_rotation_deg', 0.0)
        return pos[0], pos[1], angle
    except:
        return None


def main():
    # 1. Load Baseline
    baseline_path = DATA_DIR / "recorded_sample_bg.pth"
    if not baseline_path.exists():
        print(f"Error: Baseline not found at {baseline_path}")
        return

    print(f"Loading Baseline...")
    data_bg = torch.load(baseline_path, weights_only=False)
    bg_idx = to_numpy(data_bg['layers'][LAYER_NAME]['indices'])
    bg_feat = to_numpy(data_bg['layers'][LAYER_NAME]['features'])
    bg_map = {tuple(row[1:]): feat for row, feat in zip(bg_idx, bg_feat)}

    # 2. Find perturbed files and sample evenly
    files = sorted([f for f in glob.glob(str(DATA_DIR / "recorded_*.pth")) if 'sample_bg' not in f])
    
    # Sample evenly spaced scenes
    step = max(1, len(files) // NUM_SCENES_TO_SAMPLE)
    sampled_files = files[::step][:NUM_SCENES_TO_SAMPLE]
    
    print(f"Analyzing {len(sampled_files)} scenes out of {len(files)} total\n")

    all_tail_counts = []

    for f_path in sampled_files:
        try:
            match = re.search(r'(\d+)', Path(f_path).stem)
            if not match:
                continue
            scene_id = int(match.group(1))
            
            # Get wall position for context
            pos_info = get_position_from_pkl(scene_id)
            pos_str = f"X:{pos_info[0]:.1f}, Y:{pos_info[1]:.1f}, A:{pos_info[2]:.0f}" if pos_info else "unknown"

            # Load perturbed features
            data_pt = torch.load(f_path, weights_only=False)
            if LAYER_NAME not in data_pt['layers']:
                continue

            pt_idx = to_numpy(data_pt['layers'][LAYER_NAME]['indices'])
            pt_feat = to_numpy(data_pt['layers'][LAYER_NAME]['features'])
            pt_map = {tuple(row[1:]): feat for row, feat in zip(pt_idx, pt_feat)}

            # Shared voxels
            common_coords = set(bg_map.keys()).intersection(set(pt_map.keys()))
            if len(common_coords) == 0:
                continue

            vec_bg = np.stack([bg_map[c] for c in common_coords])
            vec_pt = np.stack([pt_map[c] for c in common_coords])
            diffs = vec_pt - vec_bg
            magnitudes = np.linalg.norm(diffs, axis=1)

            median_mag = np.median(magnitudes)
            above_2x = (magnitudes > 2 * median_mag).sum()
            above_5x = (magnitudes > 5 * median_mag).sum()
            above_10x = (magnitudes > 10 * median_mag).sum()
            all_tail_counts.append(above_5x)

            print(f"Scene {scene_id:04d} | Wall: [{pos_str}]")
            print(f"  Total shared voxels: {len(magnitudes)}")
            print(f"  Mean:   {np.mean(magnitudes):.6f}")
            print(f"  Median: {median_mag:.6f}")
            print(f"  90th percentile: {np.percentile(magnitudes, 90):.6f}")
            print(f"  95th percentile: {np.percentile(magnitudes, 95):.6f}")
            print(f"  99th percentile: {np.percentile(magnitudes, 99):.6f}")
            print(f"  Max:    {np.max(magnitudes):.6f}")
            print(f"  Voxels above 2x median: {above_2x}")
            print(f"  Voxels above 5x median: {above_5x}")
            print(f"  Voxels above 10x median: {above_10x}")
            print()

        except Exception as e:
            print(f"Error processing {f_path}: {e}\n")

    # Summary
    if all_tail_counts:
        print("=" * 60)
        print("SUMMARY: Voxels above 5x median across sampled scenes")
        print("=" * 60)
        print(f"  Min:    {min(all_tail_counts)}")
        print(f"  Max:    {max(all_tail_counts)}")
        print(f"  Mean:   {np.mean(all_tail_counts):.0f}")
        print(f"  Median: {np.median(all_tail_counts):.0f}")
        print(f"\nSuggested K: {int(np.median(all_tail_counts))}")
        print(f"  (median of 'above 5x median' counts)")
        print(f"  This captures the natural attack footprint per scene.")


if __name__ == '__main__':
    main()