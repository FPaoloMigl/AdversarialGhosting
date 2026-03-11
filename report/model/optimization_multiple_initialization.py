import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import glob
import os
import matplotlib.pyplot as plt

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/report/model/surrogate_model_F4_top1000.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
LAYER_TO_ATTACK = 'backbone_3d.conv4'

# The range your surrogate was trained on (Must match Isaac Sim scene generation bounds)
TRAIN_RANGE_MIN = -3.5
TRAIN_RANGE_MAX = 3.5

# Ensure output directory exists for plots
OUTPUT_PLOT_DIR = '/isaacsim_projects/AdversarialGhosting/report/model/'
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. SURROGATE MODEL
# ---------------------------------------------------------
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256),       nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): 
        return self.net(x)


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
# 4. PLOTTING FUNCTION
# ---------------------------------------------------------
def plot_theta_trajectory(x_hist, y_hist, loss_hist, run_id):
    """Generates a 3-part plot showing loss, coordinate convergence, and 2D trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(len(x_hist))

    # Plot 1: Loss
    axes[0].plot(epochs, loss_hist, color='black', linewidth=2)
    axes[0].set_title(f'Loss Value over Time (Run {run_id})')
    axes[0].set_xlabel('Optimization Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # Plot 2: X and Y over Epochs
    axes[1].plot(epochs, x_hist, label='Theta X', color='blue', linewidth=2)
    axes[1].plot(epochs, y_hist, label='Theta Y', color='red', linewidth=2)
    axes[1].set_title('Coordinates over Time')
    axes[1].set_xlabel('Optimization Step')
    axes[1].set_ylabel('Position')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)

    # Plot 3: 2D Spatial Trajectory
    axes[2].plot(x_hist, y_hist, color='purple', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[2].scatter(x_hist[0], y_hist[0], color='green', marker='o', s=100, label='Start')
    axes[2].scatter(x_hist[-1], y_hist[-1], color='red', marker='*', s=200, label=f'End ({x_hist[-1]:.2f}, {y_hist[-1]:.2f})')
    
    # Show the "Safety Box" limits
    axes[2].set_xlim(TRAIN_RANGE_MIN - 0.5, TRAIN_RANGE_MAX + 0.5)
    axes[2].set_ylim(TRAIN_RANGE_MIN - 0.5, TRAIN_RANGE_MAX + 0.5)
    
    # Draw a box representing the allowed boundaries
    rect = plt.Rectangle((TRAIN_RANGE_MIN, TRAIN_RANGE_MIN), 
                         TRAIN_RANGE_MAX - TRAIN_RANGE_MIN, 
                         TRAIN_RANGE_MAX - TRAIN_RANGE_MIN, 
                         fill=False, color='black', linestyle='--', linewidth=2, label='Boundary')
    axes[2].add_patch(rect)

    axes[2].set_title('2D Search Trajectory')
    axes[2].set_xlabel('X Coordinate')
    axes[2].set_ylabel('Y Coordinate')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_PLOT_DIR, f'theta_run_{run_id}.png')
    plt.savefig(out_file, dpi=150)
    plt.close() # Free memory
    print(f"✅ Plot saved to: {out_file}")


# ---------------------------------------------------------
# 5. ADVERSARIAL OPTIMIZATION LOOP
# ---------------------------------------------------------
def optimize_placement_j_theta(model, surrogate, data_dict, layer_to_attack, init_val, run_id, 
                               head_idx=0, num_iters=500, lr=0.1, alpha=10.0, beta=5.0):
    
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

    # 3. Initialize Theta
    theta = nn.Parameter(torch.tensor([init_val], dtype=torch.float32, device='cuda'), requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)

    theta_x_history, theta_y_history, loss_history = [], [], []
    tracked_hm, handles = {}, []

    # Hooks
    def hm_hook(module, inp, output): tracked_hm['adv_hm'] = output.features if hasattr(output, 'features') else output
    def injection_hook(module, inp, output):
        delta = surrogate(theta).expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + delta)

    modules_dict = dict(model.named_modules())
    handles.append(model.dense_head.heads_list[head_idx].hm.register_forward_hook(hm_hook))
    handles.append(modules_dict[layer_to_attack].register_forward_hook(injection_hook))

    # 4. Optimization
    for step in range(num_iters):
        optimizer.zero_grad()
        tracked_hm.clear()

        if 'voxel_features' in data_dict: data_dict['voxel_features'].requires_grad_(True)

        _ = model(data_dict)
        H_adv = torch.sigmoid(tracked_hm['adv_hm'])

        missing_proxy = ((H_0 - H_adv)[mask_obj].mean() if mask_obj.sum() > 0 else torch.tensor(0.0, device='cuda'))
        new_proxy = H_adv[mask_bg].mean() if mask_bg.sum() > 0 else torch.tensor(0.0, device='cuda')
        delta_N_proxy = (H_adv - H_0).abs().mean()

        # Target: Maximize the disruption (we minimize negative loss)
        loss = delta_N_proxy + alpha * missing_proxy + beta * new_proxy
        (-loss).backward()
        
        optimizer.step()

        # --- BOX CONSTRAINT CLAMPING ---
        with torch.no_grad():
            theta[0, 0].clamp_(TRAIN_RANGE_MIN, TRAIN_RANGE_MAX)  # X
            theta[0, 1].clamp_(TRAIN_RANGE_MIN, TRAIN_RANGE_MAX)  # Y

        # Logging
        theta_x_history.append(theta[0,0].item())
        theta_y_history.append(theta[0,1].item())
        loss_history.append(loss.item())

        if step % 100 == 0:
            print(f"  Step {step:03d} | theta=[{theta[0,0].item():+.4f}, {theta[0,1].item():+.4f}] | loss={loss.item():.4f}")

    for h in handles: h.remove()
    plot_theta_trajectory(theta_x_history, theta_y_history, loss_history, run_id)
    return theta.detach(), loss_history[-1]


# ---------------------------------------------------------
# 6. MAIN MULTI-RUN LOOP
# ---------------------------------------------------------
def main():
    DATA_PATH = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy')
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

    print("--- Loading Surrogate Model ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()

    NUM_RUNS = 10 
    results = []

    print("\n" + "="*50)
    print(f"STARTING OPTIMIZATION: {NUM_RUNS} RANDOM INITIALIZATIONS")
    print("="*50)

    for i in range(NUM_RUNS):
        start_x = np.random.uniform(TRAIN_RANGE_MIN, TRAIN_RANGE_MAX)
        start_y = np.random.uniform(TRAIN_RANGE_MIN, TRAIN_RANGE_MAX)
        start_coords = [start_x, start_y]
        
        print(f"\n>>> RUN {i+1}/{NUM_RUNS} | Initial Point: [{start_x:.3f}, {start_y:.3f}]")
        
        best_theta, final_loss = optimize_placement_j_theta(
            model=model, 
            surrogate=surrogate, 
            data_dict=data_dict, 
            layer_to_attack=LAYER_TO_ATTACK, 
            init_val=start_coords, 
            run_id=i,
            num_iters=500
        )
        
        results.append((final_loss, best_theta.cpu().numpy()[0]))

    # Rank results to find the global optimum
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY (Top 3 Runs)")
    print("="*50)
    for rank in range(min(3, len(results))):
        print(f"#{rank+1} | Loss: {results[rank][0]:.6f} | Coords: [{results[rank][1][0]:.4f}, {results[rank][1][1]:.4f}]")

if __name__ == '__main__':
    main()