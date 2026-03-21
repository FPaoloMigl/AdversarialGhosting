import pickle
import glob
import os
from pathlib import Path

# --- CONFIGURATION ---
# You can put a single file path here, or a folder path to read multiple!
DATA_PATH = '/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/isaacsim/data/sample_0000.pkl'

def extract_wall_info(pkl_path):
    """Reads a single .pkl file and extracts the wall's physical parameters."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract the saved variables
        pos = data['object_position']
        # Using .get() as a safety net in case an older file doesn't have rotation
        rot = data.get('object_z_rotation_deg', 0.0) 
        
        print(f"--- File: {Path(pkl_path).name} ---")
        print(f"Position  (X, Y, Z) : [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] meters")
        print(f"Rotation  (Z-Yaw)   : {rot:.2f}°")
        print("-" * 35)

        print(data)
        
        return pos, rot
        
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}")
        return None, None

def main():
    path = Path(DATA_PATH)
    
    if not path.exists():
        print(f"Path does not exist: {path}")
        return
        
    # If the path is a single file
    if path.is_file() and path.suffix == '.pkl':
        extract_wall_info(path)
        
    # If the path is a folder, read the first 5 files as a test
    elif path.is_dir():
        pkl_files = sorted(glob.glob(str(path / "*.pkl")))
        print(f"Found {len(pkl_files)} .pkl files in directory.\n")
        
        # Just process the first 5 to avoid flooding the terminal
        for pkl_file in pkl_files[:5]: 
            extract_wall_info(pkl_file)
            
if __name__ == '__main__':
    main()