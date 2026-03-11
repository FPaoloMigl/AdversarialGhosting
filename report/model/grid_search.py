import torch
import torch.nn as nn
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
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/report/model/surrogate_model_F4_top1000.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
OUTPUT_DIR = Path('/isaacsim_projects/AdversarialGhosting/report/model')

LAYER_TO_ATTACK = 'backbone_3d.conv4'
DELTA_SCALE = 1.0

# Grid resolution — 50x50 = 2500 evaluations
GRID_SIZE = 20
THETA_MIN = -3.5
THETA_MAX = 3.5


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
# 4. EVALUATE J(theta) AT ONE POINT
# ---------------------------------------------------------
def evaluate_theta(model, surrogate, data_dict, theta_val, H_0, mask_obj, mask_bg,
                   layer_to_attack, head_idx=0, alpha=1.0, beta=3.0):
    """
    Runs one forward pass with the surrogate injecting at theta_val.
    Returns J(theta) = |delta_N| + alpha * missing + beta * new
    """
    # Create theta tensor (no grad needed — forward only)
    theta = torch.tensor([theta_val], dtype=torch.float32, device='cuda')

    # Capture heatmap
    tracked_hm = {}

    def capture_hook(module, inp, output):
        tracked_hm['hm'] = output.features if hasattr(output, 'features') else output

    def injection_hook(module, inp, output):
        delta_phi = surrogate(theta)
        delta = delta_phi.expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + DELTA_SCALE * delta)

    # Register hooks
    modules_dict = dict(model.named_modules())
    h1 = model.dense_head.heads_list[head_idx].hm.register_forward_hook(capture_hook)
    h2 = modules_dict[layer_to_attack].register_forward_hook(injection_hook)

    # Forward pass (no gradients)
    with torch.no_grad():
        _ = model(data_dict)

    # Remove hooks
    h1.remove()
    h2.remove()

    # Compute J(theta)
    H_adv = torch.sigmoid(tracked_hm['hm'])

    missing_proxy = ((H_0 - H_adv)[mask_obj].mean().item()
                     if mask_obj.sum() > 0 else 0.0)

    new_proxy = (H_adv[mask_bg].mean().item()
                 if mask_bg.sum() > 0 else 0.0)

    delta_N_proxy = (H_adv - H_0).abs().mean().item()

    J = delta_N_proxy + alpha * missing_proxy + beta * new_proxy

    return J, missing_proxy, new_proxy, delta_N_proxy


# ---------------------------------------------------------
# 5. GRID SEARCH
# ---------------------------------------------------------
def grid_search(model, surrogate, data_dict, head_idx=0, alpha=1.0, beta=3.0):
    print("\n" + "=" * 60)
    print("PHASE 1: COMPUTING CLEAN BASELINE")
    print("=" * 60)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Clean forward pass
    with torch.no_grad():
        _ = model(data_dict)
        clean_hm_logits = model.dense_head.forward_ret_dict['pred_dicts'][head_idx]['hm']
        H_0 = torch.sigmoid(clean_hm_logits).detach()

    TAU_OBJ = 0.1
    TAU_BG  = 0.05
    mask_obj = (H_0 > TAU_OBJ)
    mask_bg  = (H_0 < TAU_BG)
    print(f"  Object features (H > {TAU_OBJ}): {mask_obj.sum().item()}")
    print(f"  Background features (H < {TAU_BG}): {mask_bg.sum().item()}")

    print("\n" + "=" * 60)
    print(f"PHASE 2: GRID SEARCH ({GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE**2} evaluations)")
    print(f"  Range: [{THETA_MIN}, {THETA_MAX}] x [{THETA_MIN}, {THETA_MAX}]")
    print("=" * 60)

    # Create grid
    x_vals = np.linspace(THETA_MIN, THETA_MAX, GRID_SIZE)
    y_vals = np.linspace(THETA_MIN, THETA_MAX, GRID_SIZE)

    # Storage
    J_grid       = np.zeros((GRID_SIZE, GRID_SIZE))
    missing_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    new_grid     = np.zeros((GRID_SIZE, GRID_SIZE))
    deltaN_grid  = np.zeros((GRID_SIZE, GRID_SIZE))

    best_J = -np.inf
    best_theta = None
    best_missing = 0
    best_new = 0

    total = GRID_SIZE * GRID_SIZE
    count = 0

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            J, miss, new, dN = evaluate_theta(
                model, surrogate, data_dict,
                [x, y], H_0, mask_obj, mask_bg,
                LAYER_TO_ATTACK, head_idx, alpha, beta
            )

            J_grid[j, i]       = J       # j=row(y), i=col(x)
            missing_grid[j, i] = miss
            new_grid[j, i]     = new
            deltaN_grid[j, i]  = dN

            if J > best_J:
                best_J = J
                best_theta = [x, y]
                best_missing = miss
                best_new = new

            count += 1
            if count % 100 == 0 or count == total:
                print(f"  [{count:4d}/{total}] Current best: "
                      f"J={best_J:.6f} at theta=[{best_theta[0]:.3f}, {best_theta[1]:.3f}] | "
                      f"missing={best_missing:.6f} | new={best_new:.6f}")

    print(f"\n{'=' * 60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"  Global optimum: theta = [{best_theta[0]:.4f}, {best_theta[1]:.4f}]")
    print(f"  J(theta)       = {best_J:.6f}")
    print(f"  Missing proxy  = {best_missing:.6f}")
    print(f"  New proxy      = {best_new:.6f}")
    print(f"{'=' * 60}")

    # Save results
    np.savez(
        OUTPUT_DIR / 'grid_search_results.npz',
        x_vals=x_vals, y_vals=y_vals,
        J_grid=J_grid, missing_grid=missing_grid,
        new_grid=new_grid, deltaN_grid=deltaN_grid,
        best_theta=best_theta, best_J=best_J
    )
    print(f"  Results saved to: {OUTPUT_DIR / 'grid_search_results.npz'}")

    # Plot
    plot_landscape(x_vals, y_vals, J_grid, missing_grid, new_grid, best_theta)

    return best_theta, best_J


# ---------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------
def plot_landscape(x_vals, y_vals, J_grid, missing_grid, new_grid, best_theta):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: J(theta) landscape
    im0 = axes[0].imshow(
        J_grid, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower', aspect='auto', cmap='hot'
    )
    axes[0].plot(best_theta[0], best_theta[1], 'c*', markersize=15, label=f'Best ({best_theta[0]:.2f}, {best_theta[1]:.2f})')
    axes[0].set_title('J(θ) — Total Objective')
    axes[0].set_xlabel('θ_x')
    axes[0].set_ylabel('θ_y')
    axes[0].legend(loc='upper right')
    plt.colorbar(im0, ax=axes[0])

    # Plot 2: Missing proxy landscape
    im1 = axes[1].imshow(
        missing_grid, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower', aspect='auto', cmap='Reds'
    )
    axes[1].plot(best_theta[0], best_theta[1], 'c*', markersize=15)
    axes[1].set_title('Missing Proxy (suppress objects)')
    axes[1].set_xlabel('θ_x')
    axes[1].set_ylabel('θ_y')
    plt.colorbar(im1, ax=axes[1])

    # Plot 3: New proxy landscape
    im2 = axes[2].imshow(
        new_grid, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower', aspect='auto', cmap='Blues'
    )
    axes[2].plot(best_theta[0], best_theta[1], 'c*', markersize=15)
    axes[2].set_title('New Proxy (create false detections)')
    axes[2].set_xlabel('θ_x')
    axes[2].set_ylabel('θ_y')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'grid_search_landscape.png'
    plt.savefig(out_path, dpi=150)
    print(f"\n  Landscape plot saved to: {out_path}")


# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------
def main():
    DATA_PATH = Path(
        '/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy'
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

    # Prepare data
    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    # Load surrogate
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()

    # Run grid search
    best_theta, best_J = grid_search(
        model=model,
        surrogate=surrogate,
        data_dict=data_dict,
        head_idx=0,
        alpha=10.0,
        beta=2.0
    )

    print(f"\nFinal: Place reflector at x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
    print(f"       Expected disruption J = {best_J:.6f}")


if __name__ == '__main__':
    main()