import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/rivermark/model/surrogate_model_F6_v3_sincos.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
LAYER_TO_ATTACK = 'backbone_3d.conv6'

# Physical bounds for x and y
XY_BOUNDS_MIN = torch.tensor([-65.5, -8.0], device='cuda')
XY_BOUNDS_MAX = torch.tensor([5.5, 52.0], device='cuda')

# Angle bounds (physical degrees)
ANGLE_MIN = 0.0
ANGLE_MAX = 360.0

OUTPUT_PLOT_DIR = '/isaacsim_projects/AdversarialGhosting/QualityAssessment/rivermark/optimization'
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. SURROGATE MODEL (input_dim=4: x, y, sin(a), cos(a))
# ---------------------------------------------------------
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),  nn.BatchNorm1d(32),  nn.GELU(),
            nn.Linear(32, 128),        nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): 
        return self.net(x)


def encode_theta_for_surrogate(theta):
    """
    Converts physical theta [x, y, angle_deg] to surrogate input [x_norm, y_norm, sin(a), cos(a)].
    
    - x, y: min-max normalized to [-1, 1]
    - angle: encoded as sin/cos (already in [-1, 1], periodic, bounded)
    
    This is differentiable — gradients flow back through sin/cos to the raw angle.
    """
    # Normalize x, y to [-1, 1]
    xy_norm = (theta[:, :2] - XY_BOUNDS_MIN) / (XY_BOUNDS_MAX - XY_BOUNDS_MIN)  # [0, 1]
    xy_norm = xy_norm * 2.0 - 1.0  # [-1, 1]
    
    # Encode angle as sin/cos (differentiable)
    angle_rad = theta[:, 2:3] * (3.14159265358979 / 180.0)  # deg → rad
    sin_a = torch.sin(angle_rad)
    cos_a = torch.cos(angle_rad)
    
    # Concatenate: [x_norm, y_norm, sin(a), cos(a)] — shape [1, 4]
    return torch.cat([xy_norm, sin_a, cos_a], dim=-1)


# ---------------------------------------------------------
# 2. DATASET
# ---------------------------------------------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True,
                 root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        self.sample_file_list = (
            glob.glob(str(root_path / f'*{self.ext}'))
            if self.root_path.is_dir()
            else [self.root_path]
        )
        self.sample_file_list.sort()

    def __len__(self): return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32)
            points = (points.reshape(-1, 5) if points.shape[0] % 5 == 0 else points.reshape(-1, 4))
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])

        required_dim = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        if points.shape[-1] < required_dim:
            padding = np.zeros((points.shape[0], required_dim - points.shape[-1]))
            points  = np.hstack((points, padding))
        elif points.shape[-1] > required_dim:
            points  = points[:, :required_dim]

        return self.prepare_data({'points': points, 'frame_id': index})


# ---------------------------------------------------------
# 3. INDICE_KEY FIX
# ---------------------------------------------------------
def fix_head_indice_keys(model):
    print("--- Fixing indice_key collision in dense_head ---")
    for i, head in enumerate(model.dense_head.heads_list):
        for cur_name in head.sep_head_dict:
            seq = head.__getattr__(cur_name)
            for module in seq.modules():
                if hasattr(module, 'indice_key') and module.indice_key is not None:
                    old_key = module.indice_key
                    module.indice_key = f"{old_key}_head{i}"
    print("--- indice_key fix applied ---\n")


# ---------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------
def plot_theta_trajectory(x_hist, y_hist, a_hist, loss_hist, run_id):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    epochs = range(len(x_hist))

    # Plot 1: Loss
    axes[0].plot(epochs, loss_hist, color='black', linewidth=2)
    axes[0].set_title(f'Disruption Score (Run {run_id})')
    axes[0].set_xlabel('Optimization Step')
    axes[0].set_ylabel('Score (Higher = Better)')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # Plot 2: X and Y
    axes[1].plot(epochs, x_hist, label='Theta X', color='blue', linewidth=2)
    axes[1].plot(epochs, y_hist, label='Theta Y', color='red', linewidth=2)
    axes[1].set_title('Coordinates over Time')
    axes[1].set_xlabel('Optimization Step')
    axes[1].set_ylabel('Position (m)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)

    # Plot 3: Angle
    axes[2].plot(epochs, a_hist, label='Angle', color='orange', linewidth=2)
    axes[2].set_title('Wall Angle over Time')
    axes[2].set_xlabel('Optimization Step')
    axes[2].set_ylabel('Angle (Degrees)')
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.7)

    # Plot 4: 2D Trajectory
    axes[3].plot(x_hist, y_hist, color='purple', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[3].scatter(x_hist[0], y_hist[0], color='green', marker='o', s=100, label='Start')
    axes[3].scatter(x_hist[-1], y_hist[-1], color='red', marker='*', s=200,
                    label=f'End ({x_hist[-1]:.2f}, {y_hist[-1]:.2f})')
    axes[3].set_xlim(XY_BOUNDS_MIN[0].item() - 2, XY_BOUNDS_MAX[0].item() + 2)
    axes[3].set_ylim(XY_BOUNDS_MIN[1].item() - 2, XY_BOUNDS_MAX[1].item() + 2)
    rect = plt.Rectangle((XY_BOUNDS_MIN[0].item(), XY_BOUNDS_MIN[1].item()),
                         XY_BOUNDS_MAX[0].item() - XY_BOUNDS_MIN[0].item(),
                         XY_BOUNDS_MAX[1].item() - XY_BOUNDS_MIN[1].item(),
                         fill=False, color='black', linestyle='--', linewidth=2, label='Boundary')
    axes[3].add_patch(rect)
    axes[3].set_title('2D Search Trajectory')
    axes[3].set_xlabel('X Coordinate')
    axes[3].set_ylabel('Y Coordinate')
    axes[3].legend()
    axes[3].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_PLOT_DIR, f'theta_run_{run_id}.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_file}")


# ---------------------------------------------------------
# 5. ADVERSARIAL OPTIMIZATION LOOP
# ---------------------------------------------------------
def optimize_placement_j_theta(model, surrogate, data_dict, layer_to_attack, init_val, run_id,
                               head_idx=0, num_iters=500, lr=0.05, alpha=10.0, beta=5.0):
    
    # 1. Prepare Model State
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    for name, module in model.named_modules():
        if 'backbone_3d' in name: module.train()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): module.eval()

    # 2. Clean Baseline
    with torch.no_grad():
        _ = model(data_dict)
        clean_hm_logits = model.dense_head.forward_ret_dict['pred_dicts'][head_idx]['hm']
        H_0 = torch.sigmoid(clean_hm_logits).detach()

    mask_obj = (H_0 > 0.1)
    mask_bg  = (H_0 < 0.05)

    # 3. Initialize Theta in PHYSICAL space [x, y, angle_deg]
    theta = nn.Parameter(torch.tensor([init_val], dtype=torch.float32, device='cuda'), requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)

    theta_x_history, theta_y_history, theta_a_history, loss_history = [], [], [], []
    tracked_hm, handles = {}, []

    # Hooks
    def hm_hook(module, inp, output):
        tracked_hm['adv_hm'] = output.features if hasattr(output, 'features') else output

    def injection_hook(module, inp, output):
        # Encode physical theta → [x_norm, y_norm, sin(a), cos(a)]
        # This is fully differentiable: gradients flow through sin/cos back to the angle
        theta_encoded = encode_theta_for_surrogate(theta)
        
        # Surrogate outputs real-scale delta (Y was NOT normalized)
        delta = surrogate(theta_encoded)
        delta = delta.expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + delta)

    modules_dict = dict(model.named_modules())
    handles.append(model.dense_head.heads_list[head_idx].hm.register_forward_hook(hm_hook))
    handles.append(modules_dict[layer_to_attack].register_forward_hook(injection_hook))

    # 4. Optimization
    for step in range(num_iters):
        optimizer.zero_grad()
        tracked_hm.clear()

        if 'voxel_features' in data_dict:
            data_dict['voxel_features'].requires_grad_(True)

        _ = model(data_dict)
        H_adv = torch.sigmoid(tracked_hm['adv_hm'])

        missing_proxy = ((H_0 - H_adv)[mask_obj].mean() if mask_obj.sum() > 0 else torch.tensor(0.0, device='cuda'))
        new_proxy = H_adv[mask_bg].mean() if mask_bg.sum() > 0 else torch.tensor(0.0, device='cuda')
        delta_N_proxy = (H_adv - H_0).abs().mean()

        loss = delta_N_proxy + alpha * missing_proxy + beta * new_proxy
        (-loss).backward()
        optimizer.step()

        # Clamp to physical bounds
        with torch.no_grad():
            theta[0, 0].clamp_(XY_BOUNDS_MIN[0].item(), XY_BOUNDS_MAX[0].item())
            theta[0, 1].clamp_(XY_BOUNDS_MIN[1].item(), XY_BOUNDS_MAX[1].item())
            theta[0, 2].clamp_(ANGLE_MIN, ANGLE_MAX)

        # Logging
        theta_x_history.append(theta[0, 0].item())
        theta_y_history.append(theta[0, 1].item())
        theta_a_history.append(theta[0, 2].item())
        loss_history.append(loss.item())

        if step % 50 == 0:
            grad_str = ("None" if theta.grad is None
                       else f"[{theta.grad[0,0].item():.6e}, {theta.grad[0,1].item():.6e}, {theta.grad[0,2].item():+.6e}]")
            print(f"  Step {step:03d} | "
                  f"theta=[X:{theta[0,0].item():+.2f}, Y:{theta[0,1].item():+.2f}, A:{theta[0,2].item():.1f}] | "
                  f"grad={grad_str} | "
                  f"loss={loss.item():.6f}")

    for h in handles: h.remove()
    plot_theta_trajectory(theta_x_history, theta_y_history, theta_a_history, loss_history, run_id)
    return theta.detach(), loss_history[-1]


# ---------------------------------------------------------
# 6. MAIN MULTI-RUN LOOP
# ---------------------------------------------------------
def main():
    DATA_PATH = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/rivermark/ply/sample_bg.npy')
    logger = common_utils.create_logger()

    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("--- Loading VoxelNeXt ---")
    cfg_from_yaml_file(CFG_FILE, cfg)
    demo_dataset = DemoDataset(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES,
        training=False, root_path=DATA_PATH, ext='.npy', logger=logger
    )
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), demo_dataset)
    model.load_params_from_file(CKPT_FILE, logger=logger, to_cpu=True)
    model.cuda()
    fix_head_indice_keys(model)

    data_dict = demo_dataset.collate_batch([demo_dataset[0]])
    load_data_to_gpu(data_dict)

    print("--- Loading Surrogate Model (sin/cos) ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()

    # Sanity check
    print("\n--- Surrogate Sanity Check ---")
    with torch.no_grad():
        for x, y, a in [(-5, 8, 90), (-5, 8, 180), (-5, 8, 270)]:
            t = torch.tensor([[x, y, a]], device='cuda', dtype=torch.float32)
            t_enc = encode_theta_for_surrogate(t)
            delta = surrogate(t_enc)
            print(f"  theta=({x}, {y}, {a}) -> encoded={t_enc[0].tolist()} -> "
                  f"delta_norm={delta.norm():.4f}")

    NUM_RUNS = 10
    results = []

    print("\n" + "="*50)
    print(f"STARTING OPTIMIZATION: {NUM_RUNS} RANDOM INITIALIZATIONS")
    print("="*50)

    for i in range(NUM_RUNS):
        start_x = np.random.uniform(XY_BOUNDS_MIN[0].item(), XY_BOUNDS_MAX[0].item())
        start_y = np.random.uniform(XY_BOUNDS_MIN[1].item(), XY_BOUNDS_MAX[1].item())
        start_angle = np.random.uniform(ANGLE_MIN, ANGLE_MAX)
        start_coords = [start_x, start_y, start_angle]

        print(f"\n>>> RUN {i+1}/{NUM_RUNS} | Init: [X:{start_x:.2f}, Y:{start_y:.2f}, A:{start_angle:.1f}]")

        best_theta, final_loss = optimize_placement_j_theta(
            model=model,
            surrogate=surrogate,
            data_dict=data_dict,
            layer_to_attack=LAYER_TO_ATTACK,
            init_val=start_coords,
            run_id=i,
            num_iters=1000
        )

        results.append((final_loss, best_theta.cpu().numpy()[0]))

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY (Top 3 Runs)")
    print("="*50)
    for rank in range(min(3, len(results))):
        r = results[rank]
        print(f"#{rank+1} | Loss: {r[0]:.6f} | X: {r[1][0]:+.2f}m | Y: {r[1][1]:+.2f}m | Angle: {r[1][2]:.1f}")


if __name__ == '__main__':
    main()