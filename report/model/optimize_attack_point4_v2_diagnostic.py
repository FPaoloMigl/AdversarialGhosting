import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/model/surrogate_model_F6_v2.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
LAYER_TO_ATTACK = 'backbone_3d.conv6'

# === FIX 1: Per-dimension physical bounds (matching training normalization) ===
BOUNDS_MIN = torch.tensor([-30.0, -30.0, 0.0], device='cuda')
BOUNDS_MAX = torch.tensor([30.0,  30.0, 360.0], device='cuda')


# ---------------------------------------------------------
# 1. SURROGATE MODEL (FIX 2: matches Training_model exactly — BatchNorm, not Dropout)
# ---------------------------------------------------------
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),  nn.BatchNorm1d(32),  nn.GELU(),
            nn.Linear(32, 128),        nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)


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
            points = (points.reshape(-1, 5)
                      if points.shape[0] % 5 == 0
                      else points.reshape(-1, 4))
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
# 4. ADVERSARIAL OPTIMIZATION LOOP
# ---------------------------------------------------------
def optimize_placement_j_theta(model, surrogate, data_dict, layer_to_attack,
                               init_theta=None,
                               head_idx=0, num_iters=500, lr=0.05, alpha=10.0, beta=5.0):
    """
    Optimizes theta = [x, y, angle] to maximize heatmap disruption.
    Theta lives in physical coordinates; min-max normalized to [0,1] before surrogate.
    """
    print("\n" + "="*50)
    print("PHASE 1: COMPUTING CLEAN BASELINE (H_0)")
    print("="*50)

    # 1. Prepare Model State
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    for name, module in model.named_modules():
        if 'backbone_3d' in name:
            module.train()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()

    # 2. Clean Forward Pass
    with torch.no_grad():
        _ = model(data_dict)
        clean_hm_logits = model.dense_head.forward_ret_dict['pred_dicts'][head_idx]['hm']
        H_0 = torch.sigmoid(clean_hm_logits).detach()

    # 3. Spatial Masks
    TAU_OBJ = 0.1
    TAU_BG  = 0.05
    mask_obj = (H_0 > TAU_OBJ)
    mask_bg  = (H_0 < TAU_BG)
    print(f"Object features: {mask_obj.sum().item()}")
    print(f"Background features: {mask_bg.sum().item()}")

    print("\n" + "="*50)
    print("PHASE 2: OPTIMIZING THETA (Gradient Ascent on J(theta))")
    print("="*50)

    # === FIX 3: Default initialization within valid physical bounds ===
    if init_theta is None:
        init_theta = [0.0, 0.0, 90.0]  # X, Y, Angle 
    
    theta = nn.Parameter(
        torch.tensor([init_theta], dtype=torch.float32, device='cuda'),
        requires_grad=True
    )
    optimizer = optim.Adam([theta], lr=lr)

    theta_x_history, theta_y_history, theta_a_history, loss_history = [], [], [], []

    # 5. Hooks
    tracked_hm = {}
    handles = []

    def hm_hook(module, inp, output):
        tracked_hm['adv_hm'] = output.features if hasattr(output, 'features') else output

    # === FIX 4: Min-max normalization to [0,1], no output denormalization ===
    def injection_hook(module, inp, output):
        # Normalize physical theta to [0, 1] (matching training)
        theta_norm = (theta - BOUNDS_MIN) / (BOUNDS_MAX - BOUNDS_MIN)
        theta_norm = theta_norm * 2.0 - 1.0
        
        # Surrogate output is already real-scale (Y was NOT normalized)
        delta = surrogate(theta_norm)
        delta = delta.expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + delta)

    modules_dict = dict(model.named_modules())
    handles.append(model.dense_head.heads_list[head_idx].hm.register_forward_hook(hm_hook))
    handles.append(modules_dict[layer_to_attack].register_forward_hook(injection_hook))

    # 6. Optimization Loop
    for step in range(num_iters):
        optimizer.zero_grad()
        tracked_hm.clear()

        if 'voxel_features' in data_dict:
            data_dict['voxel_features'].requires_grad_(True)

        _ = model(data_dict)
        H_adv = torch.sigmoid(tracked_hm['adv_hm'])

        missing_proxy = ((H_0 - H_adv)[mask_obj].mean()
                         if mask_obj.sum() > 0
                         else torch.tensor(0.0, device='cuda'))
        new_proxy = (H_adv[mask_bg].mean()
                     if mask_bg.sum() > 0
                     else torch.tensor(0.0, device='cuda'))
        delta_N_proxy = (H_adv - H_0).abs().mean()

        loss = delta_N_proxy + alpha * missing_proxy + beta * new_proxy
        (-loss).backward()
        optimizer.step()

        # === FIX 5: Clamp to physical bounds, not z-score ranges ===
        with torch.no_grad():
            theta[0, 0].clamp_(BOUNDS_MIN[0].item(), BOUNDS_MAX[0].item())  # X: [-30, 30]
            theta[0, 1].clamp_(BOUNDS_MIN[1].item(), BOUNDS_MAX[1].item())  # Y: [-30, 30]
            theta[0, 2].clamp_(BOUNDS_MIN[2].item(), BOUNDS_MAX[2].item())  # Angle: [0, 360]

        # Logging
        theta_x_history.append(theta[0, 0].item())
        theta_y_history.append(theta[0, 1].item())
        theta_a_history.append(theta[0, 2].item())
        loss_history.append(loss.item())

        grad_str = ("None" if theta.grad is None 
                    else f"[{theta.grad[0,0].item():.6e}, {theta.grad[0,1].item():.6e}, {theta.grad[0,2].item():+.6e}]")

        if step % 50 == 0:
            print(f"  Step {step:03d} | "
                  f"theta=[X:{theta[0,0].item():+.2f}, Y:{theta[0,1].item():+.2f}, A:{theta[0,2].item():.1f}] | "
                  f"grad={grad_str} | "
                  f"loss={loss.item():.6f} | "
                  f"missing={missing_proxy.item():.6f} | "
                  f"new={new_proxy.item():.6f}")

    for h in handles:
        h.remove()

    print("\nOptimization Complete.")
    plot_theta_trajectory(theta_x_history, theta_y_history, theta_a_history, loss_history)
    return theta.detach()


# ---------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------
def plot_theta_trajectory(x_hist, y_hist, a_hist, loss_hist):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    epochs = range(len(x_hist))

    # Plot 1: Loss
    axes[0].plot(epochs, loss_hist, color='black', linewidth=2)
    axes[0].set_title('Disruption Value over Time')
    axes[0].set_xlabel('Optimization Step')
    axes[0].set_ylabel('Disruption Score (Higher = Better)')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # Plot 2: X and Y
    axes[1].plot(epochs, x_hist, label='Theta X', color='blue', linewidth=2)
    axes[1].plot(epochs, y_hist, label='Theta Y', color='red', linewidth=2)
    axes[1].set_title('Theta Coordinates over Time')
    axes[1].set_xlabel('Optimization Step')
    axes[1].set_ylabel('Physical Distance (m)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)

    # Plot 3: Angle
    axes[2].plot(epochs, a_hist, label='Theta Angle', color='orange', linewidth=2)
    axes[2].set_title('Wall Angle over Time')
    axes[2].set_xlabel('Optimization Step')
    axes[2].set_ylabel('Angle (Degrees)')
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.7)

    # Plot 4: 2D Trajectory
    axes[3].plot(x_hist, y_hist, color='purple', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[3].scatter(x_hist[0], y_hist[0], color='green', marker='o', s=100, 
                    label=f'Start ({x_hist[0]:.1f}, {y_hist[0]:.1f})')
    axes[3].scatter(x_hist[-1], y_hist[-1], color='red', marker='*', s=200, 
                    label=f'End ({x_hist[-1]:.2f}, {y_hist[-1]:.2f})')
    
    axes[3].set_xlim(BOUNDS_MIN[0].item() - 2, BOUNDS_MAX[0].item() + 2)
    axes[3].set_ylim(BOUNDS_MIN[1].item() - 2, BOUNDS_MAX[1].item() + 2)
    
    rect = plt.Rectangle((BOUNDS_MIN[0].item(), BOUNDS_MIN[1].item()),
                         BOUNDS_MAX[0].item() - BOUNDS_MIN[0].item(),
                         BOUNDS_MAX[1].item() - BOUNDS_MIN[1].item(),
                         fill=False, color='black', linestyle='--', linewidth=2, label='Boundary')
    axes[3].add_patch(rect)

    axes[3].set_title('2D Search Trajectory')
    axes[3].set_xlabel('X Coordinate')
    axes[3].set_ylabel('Y Coordinate')
    axes[3].legend()
    axes[3].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = '/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/optimization/theta_convergence_plot_v3.png'
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"\nConvergence plot saved to: {out_file}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    DATA_PATH = Path(
        '/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/isaacsim/data/ply'
    )
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

    # Prepare Data
    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    # Point cloud sanity check
    points = np.load('/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/isaacsim/data/ply/sample_bg.npy')
    print(f"Point cloud shape: {points.shape}")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # Load Surrogate
    print("--- Loading Surrogate Model ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()

    # === Sanity check: verify surrogate outputs at known points ===
    # print("\n--- Surrogate Sanity Check ---")
    # with torch.no_grad():
    #     for x, y, a in [(-11, 10, 90), (0, 0, 90), (1.7, 8.9, 95)]:
    #         t = torch.tensor([[x, y, a]], device='cuda', dtype=torch.float32)
    #         t_norm = (t - BOUNDS_MIN) / (BOUNDS_MAX - BOUNDS_MIN)  # [0, 1]
    #         delta = surrogate(t_norm)
    #         print(f"  theta=({x}, {y}, {a}) -> norm={delta.norm():.4f}, "
    #               f"range=[{delta.min():.4f}, {delta.max():.4f}]")

    # Execute Optimization
    optimal_theta = optimize_placement_j_theta(
        model=model,
        surrogate=surrogate,
        data_dict=data_dict,
        layer_to_attack=LAYER_TO_ATTACK,
        init_theta=[0.0, 0.0, 90.0],  # Physical: X(m), Y(m), Angle(deg)
        head_idx=0,
        num_iters=2000,
        lr=0.05,
        alpha=10.0,
        beta=5.0
    )

    print(f"\nFinal Optimal Reflector Placement:")
    print(f"  X:     {optimal_theta[0, 0].item():+.2f} m")
    print(f"  Y:     {optimal_theta[0, 1].item():+.2f} m")
    print(f"  Angle: {optimal_theta[0, 2].item():.1f}°")


if __name__ == '__main__':
    main()