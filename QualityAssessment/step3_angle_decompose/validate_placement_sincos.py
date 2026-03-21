import torch
import numpy as np
from pathlib import Path
import torch.nn as nn

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === 1. CONFIGURATION ===
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'

# The physical point clouds from Isaac Sim
CLEAN_PC_PATH = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3_angle_decompose/step3_small_wall/ply/sample_bg.npy')
ADV_PC_PATH   = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3_angle_decompose/step3_small_wall/optimization/sample_test.npy')

# Surrogate Model Details (updated to sin/cos version)
SURROGATE_WEIGHTS = '/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3_angle_decompose/step3_small_wall/model/surrogate_model_F6_v3_sincos.pth'
LAYER_TO_ATTACK   = 'backbone_3d.conv6'
THETA_STAR        = [2.39, 5.87, 28.2]  # The optimized coordinate from best run
# DELTA_SCALE       = 100.0 / 1800.0
DELTA_SCALE = 1.0
SCORE_THRESH      = 0.28

# Physical bounds for x, y (must match training)
XY_BOUNDS_MIN = torch.tensor([-30.0, -30.0], device='cuda')
XY_BOUNDS_MAX = torch.tensor([30.0, 30.0], device='cuda')


# === 2. SURROGATE MODEL DEFINITION (input_dim=4: x, y, sin(a), cos(a)) ===
class MLP_Surrogate(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)


def encode_theta(theta_physical):
    """
    Converts physical theta [x, y, angle_deg] to surrogate input [x_norm, y_norm, sin(a), cos(a)].
    """
    x = theta_physical[0]
    y = theta_physical[1]
    angle_deg = theta_physical[2]
    
    # Normalize x, y to [-1, 1]
    x_norm = (x - XY_BOUNDS_MIN[0]) / (XY_BOUNDS_MAX[0] - XY_BOUNDS_MIN[0])
    x_norm = x_norm * 2.0 - 1.0
    
    y_norm = (y - XY_BOUNDS_MIN[1]) / (XY_BOUNDS_MAX[1] - XY_BOUNDS_MIN[1])
    y_norm = y_norm * 2.0 - 1.0
    
    # Encode angle as sin/cos
    angle_rad = angle_deg * (3.14159265358979 / 180.0)
    sin_a = torch.sin(angle_rad)
    cos_a = torch.cos(angle_rad)
    
    # Shape: [1, 4]
    return torch.tensor([[x_norm.item(), y_norm.item(), sin_a.item(), cos_a.item()]], 
                        dtype=torch.float32, device='cuda')


# === 3. DATASET LOADER ===
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, pc_path, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=pc_path, logger=logger)
        self.pc_path = pc_path

    def __len__(self): return 1

    def __getitem__(self, index):
        points = np.load(self.pc_path)
        required_dim = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        if points.shape[-1] < required_dim:
            padding = np.zeros((points.shape[0], required_dim - points.shape[-1]), dtype=points.dtype)
            points = np.hstack((points, padding))
        elif points.shape[-1] > required_dim:
            points = points[:, :required_dim]
        return self.prepare_data(data_dict={'points': points, 'frame_id': index})


# === 4. INFERENCE & FORMATTING ===
def run_inference(model, pc_path, logger):
    dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, pc_path=pc_path, logger=logger)
    data_dict = dataset.collate_batch([dataset[0]])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        preds, _ = model.forward(data_dict)
    return preds


def print_summary_table(title, pred_dicts, class_names, threshold=0.05):
    boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    labels = pred_dicts[0]['pred_labels'].cpu().numpy()
    
    mask = scores > threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    if len(labels) == 0:
        print(f"  [No objects detected above {threshold*100:.0f}% threshold]")
        return

    print(f"{'ID':<4} | {'Class':<12} | {'Score':<7} | {'Position (X, Y, Z)'}")
    print("-" * 80)
    for i in range(len(labels)):
        c_name = class_names[labels[i] - 1]
        score_str = f"{scores[i]:.2%}"
        pos_str = f"X: {boxes[i][0]:>6.2f}, Y: {boxes[i][1]:>6.2f}, Z: {boxes[i][2]:>5.2f}"
        print(f"#{i:<2} | {c_name:<12} | {score_str:<7} | {pos_str}")


# === 5. THE GLOBAL INJECTION HOOK ===
def get_injection_hook(delta_phi, scale):
    def hook(module, inp, output):
        delta = delta_phi.expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + scale * delta)
    return hook


# === 6. MAIN EXECUTION ===
def main():
    logger = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    
    print("\n--- Loading VoxelNeXt ---")
    init_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, pc_path=CLEAN_PC_PATH, logger=logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), init_dataset)
    model.load_params_from_file(CKPT_FILE, logger=logger, to_cpu=True)
    model.cuda().eval()
    
    # Fix the shared indice key bug for sequential runs
    for i, head in enumerate(model.dense_head.heads_list):
        for cur_name in head.sep_head_dict:
            for module in head.__getattr__(cur_name).modules():
                if hasattr(module, 'indice_key') and module.indice_key is not None:
                    module.indice_key = f"{module.indice_key}_head{i}"

    # --- RUN 1: CLEAN BASELINE (y_0) ---
    clean_preds = run_inference(model, CLEAN_PC_PATH, logger)
    print_summary_table("RUN 1: CLEAN BASELINE (y_0)", clean_preds, cfg.CLASS_NAMES, SCORE_THRESH)

    # --- RUN 2: PHYSICAL REALITY (y_theta*) ---
    if ADV_PC_PATH.exists():
        phys_preds = run_inference(model, ADV_PC_PATH, logger)
        print_summary_table("RUN 2: PHYSICAL REALITY - y(theta*)", phys_preds, cfg.CLASS_NAMES, SCORE_THRESH)
    else:
        print(f"\nWARNING: Adversarial point cloud not found at {ADV_PC_PATH}")
        print("Skipping RUN 2. Generate the point cloud in Isaac Sim first.")

    # --- RUN 3: SURROGATE PREDICTION (y_hat_theta*) ---
    print(f"\n--- Loading Surrogate Model & Computing delta_phi for {THETA_STAR} ---")
    surrogate = MLP_Surrogate().cuda()
    surrogate.load_state_dict(torch.load(SURROGATE_WEIGHTS))
    surrogate.eval()
    
    # Encode theta with sin/cos for angle
    theta_tensor = torch.tensor(THETA_STAR, dtype=torch.float32, device='cuda')
    theta_encoded = encode_theta(theta_tensor)
    
    print(f"Physical theta: X={THETA_STAR[0]}, Y={THETA_STAR[1]}, A={THETA_STAR[2]}")
    print(f"Encoded input:  {theta_encoded[0].tolist()}")
    
    with torch.no_grad():
        delta_phi = surrogate(theta_encoded)
        
    print(f"Delta Phi Norm: {torch.norm(delta_phi).item():.4f}")
    print(f"Delta Phi Range: [{delta_phi.min().item():.4f}, {delta_phi.max().item():.4f}]")

    # Attach hook
    modules_dict = dict(model.named_modules())
    hook_handle = modules_dict[LAYER_TO_ATTACK].register_forward_hook(get_injection_hook(delta_phi, DELTA_SCALE))
    
    # Run inference on CLEAN data, but with hook active
    surr_preds = run_inference(model, CLEAN_PC_PATH, logger)
    print_summary_table("RUN 3: SURROGATE PREDICTION - y_hat(theta*)", surr_preds, cfg.CLASS_NAMES, SCORE_THRESH)
    
    # Remove hook
    hook_handle.remove()


if __name__ == '__main__':
    main()