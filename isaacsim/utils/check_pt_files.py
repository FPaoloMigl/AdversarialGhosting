import torch
from pathlib import Path

# --- Load the file ---
path = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3_angle_decompose/step3_small_wall/model/surrogate_training_data_F6_v3_sincos.pt')
data = torch.load(path, weights_only=False)

# --- Inspect top-level keys ---
print("=== Top-level keys ===")
print(list(data.keys()))

# --- Inspect detections ---
print("\n=== Detections ===")
print(f"  X: {data['X'][0]}")
print(f"  Y : {data['Y'][0]}")
print(f"  Description : {data['description']}")
print(f"  xy_bounds_min : {data['xy_bounds_min']}")
print(f"  xy_bounds_max    : {data['xy_bounds_max']}")

# # --- Inspect layers ---
# print("\n=== Layers ===")
# for layer_name, layer_val in data['layers'].items():
#     idx  = layer_val['indices']
#     feat = layer_val['features']
#     print(f"  [{layer_name}]")
#     print(f"    indices  shape : {idx.shape}  dtype: {idx.dtype}")
#     print(f"    features shape : {feat.shape}  dtype: {feat.dtype}")
#     print(f"    indices  sample: {idx[0]}")
#     print(f"    features sample: {feat[0][:5]}...")  # first 5 channels
# ```

# This will print something like:
# ```
# === Top-level keys ===
# ['detections', 'layers']

# === Detections ===
#   boxes  shape : (12, 7)
#   scores shape : (12,)
#   boxes  dtype : float32
#   First box    : [ 3.14  1.02  0.5   4.2   1.8   1.5   0.01]

# === Layers ===
#   [layer_1]
#     indices  shape : (137244, 4)  dtype: int32
#     features shape : (137244, 16)  dtype: float32
#     indices  sample: [0 42 317 128]   ← [batch, z, y, x]
#     features sample: [0.12 -0.03 0.55 0.21 0.08]...