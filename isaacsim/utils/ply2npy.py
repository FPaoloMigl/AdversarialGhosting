# import open3d as o3d
# import numpy as np
# import argparse
# import sys

# def convert_ply_to_npy(ply_path, output_path):
#     print(f"Loading {ply_path}...")
    
#     # Load the PLY file
#     try:
#         pcd = o3d.io.read_point_cloud(ply_path)
#     except Exception as e:
#         print(f"Error reading PLY: {e}")
#         sys.exit(1)

#     # 1. Get Geometry (X, Y, Z)
#     # Shape: (N, 3)
#     points = np.asarray(pcd.points)
    
#     # 2. Get Intensity
#     # OpenPCDet requires 4 channels: [x, y, z, intensity]
#     # Isaac Sim PLY files sometimes hide intensity in 'colors' or custom fields.
#     # We will try to find it, or default to 0.5 if missing.
    
#     # Check if intensity is stored in colors (common in some exports)
#     # If colors exist, often R=G=B=Intensity
#     if pcd.has_colors():
#         colors = np.asarray(pcd.intensity)
#         # Take the first channel (R) as intensity
#         intensity = colors[:, 0].reshape(-1, 1)
#     else:
#         print("Warning: No intensity/color found. padding with default intensity 0.")
#         intensity = np.zeros((points.shape[0], 1))

#     # 3. Concatenate to make (N, 4)
#     # [x, y, z] + [intensity]
#     point_cloud_array = np.hstack((points, intensity)).astype(np.float32)

#     # 4. Save as .npy
#     np.save(output_path, point_cloud_array)
#     print(f"Success! Saved {point_cloud_array.shape[0]} points to:")
#     print(f" -> {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert PLY to NPY for OpenPCDet")
#     parser.add_argument("ply_file", type=str, help="Path to input .ply file")
#     args = parser.parse_args()

#     # Generate output filename
#     output_filename = args.ply_file.replace(".ply", ".npy")
#     convert_ply_to_npy(args.ply_file, output_filename)


# ============== UNCOMMENT ABOVE IF YOU WANNA USE JUST ONE FILE VERSION =======
import open3d as o3d
import numpy as np
import argparse
import sys
from pathlib import Path

def convert_ply_to_npy(ply_path, output_path):
    # Load the PLY file
    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
    except Exception as e:
        print(f"Error reading {ply_path.name}: {e}")
        return False

    if not pcd.has_points():
        print(f"Warning: {ply_path.name} is empty. Skipping.")
        return False

    # 1. Get Geometry (X, Y, Z)
    points = np.asarray(pcd.points)
    
    # 2. Get Intensity / Color
    if pcd.has_colors():
        # Open3D colors are 0.0-1.0; OpenPCDet usually expects normalized intensity
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0].reshape(-1, 1)
    else:
        intensity = np.zeros((points.shape[0], 1))

    # 3. Concatenate to make (N, 4) -> [x, y, z, intensity]
    point_cloud_array = np.hstack((points, intensity)).astype(np.float32)

    # 4. Save as .npy
    np.save(output_path, point_cloud_array)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directory of PLY to NPY")
    parser.add_argument("input_path", type=str, help="Path to input .ply file OR directory")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    # Determine if we are processing one file or a directory
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = list(input_path.glob("*.ply"))
        print(f"Found {len(files_to_process)} .ply files in {input_path}")
    else:
        print(f"Error: {args.input_path} is not a valid file or directory.")
        sys.exit(1)

    # Create an output directory if processing a folder (optional, here saving in-place)
    success_count = 0
    for ply_file in files_to_process:
        output_file = ply_file.with_suffix(".npy")
        
        if convert_ply_to_npy(ply_file, output_file):
            print(f"Converted: {ply_file.name} -> {output_file.name}")
            success_count += 1
        else:
            print(f"Failed: {ply_file.name}")

    print(f"\nFinished! Successfully converted {success_count}/{len(files_to_process)} files.")