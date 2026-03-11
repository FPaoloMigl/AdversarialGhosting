import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import glob

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
# 4. ADVERSARIAL OPTIMIZATION LOOP
# ---------------------------------------------------------
def optimize_placement_j_theta(model, surrogate, data_dict, layer_to_attack, head_idx=0, num_iters=100, lr=0.1, alpha=1.0, beta=0.5):
    """
    Executes Part III: Optimizes theta based on Missing and New detection proxies.
    """
    print("\n" + "="*50)
    print("PHASE 1: COMPUTING CLEAN BASELINE (y_0)")
    print("="*50)

    # 1. Ensure model is in the correct Pseudo-Train state
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    for name, module in model.named_modules():
        if 'backbone_3d' in name:
            module.train() # Keep spconv graph alive
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()

    # 2. Run Clean Forward Pass
    with torch.no_grad():
        _ = model(data_dict)
        # Extract the clean heatmap for the target head (e.g., Cars)
        clean_hm_logits = model.dense_head.forward_ret_dict['pred_dicts'][head_idx]['hm']
        H_0 = torch.sigmoid(clean_hm_logits).detach()

    # 3. Define the Spatial Masks (M_obj and M_bg)
    TAU_OBJ = 0.1  # Threshold for existing objects
    TAU_BG  = 0.05 # Threshold for background
    
    mask_obj = (H_0 > TAU_OBJ)
    mask_bg  = (H_0 < TAU_BG)
    
    print(f"Detected {mask_obj.sum().item()} features belonging to real objects.")
    print(f"Detected {mask_bg.sum().item()} features belonging to background.")

    print("\n" + "="*50)
    print(f"PHASE 2: OPTIMIZING THETA (Gradient Ascent on J(theta))")
    print("="*50)

    # 4. Initialize Theta (x, y placement)
    theta = nn.Parameter(
        torch.tensor([[-1.1, -0.4]], dtype=torch.float32, device='cuda'),
        requires_grad=True
    )
    optimizer = optim.Adam([theta], lr=lr)

    # --- ADD TRACKING LISTS HERE ---
    theta_x_history = []
    theta_y_history = []
    loss_history = [] # <-- NEW: Tracking the loss

    # 5. Setup the Differentiable Pipeline (Hooks)
    tracked_hm = {}
    handles = []

    def make_hm_hook():
        def hook(module, inp, output):
            tracked_hm['adv_hm'] = output.features if hasattr(output, 'features') else output
        return hook

    # Hook the target head to get \hat{y}(\theta)
    modules_dict = dict(model.named_modules())
    target_head = model.dense_head.heads_list[head_idx].hm
    handles.append(target_head.register_forward_hook(make_hm_hook()))

    DELTA_SCALE = 10.0

    # Hook the chosen layer \ell to inject g_\ell(\theta)
    def injection_hook(module, inp, output):
        delta_phi = surrogate(theta)                                   
        delta = delta_phi.expand(output.features.shape[0], -1)         
        # new_output = output.replace_feature(output.features + DELTA_SCALE*delta)
        new_output = output.replace_feature(output.features + delta)
        return new_output
    
    handles.append(modules_dict[layer_to_attack].register_forward_hook(injection_hook))

    # 6. Optimization Loop
    for step in range(num_iters):
        optimizer.zero_grad()
        tracked_hm.clear()

        # Re-initialize grad tape for the backbone features
        if 'voxel_features' in data_dict:
            data_dict['voxel_features'].requires_grad_(True)

        # Forward Pass through differentiable pipeline
        _ = model(data_dict)
        
        # Get \hat{H}_\theta (yhat(theta))
        H_adv = torch.sigmoid(tracked_hm['adv_hm'])

        # Calculate Differentiable Proxies
        # 1. Missing Proxy: Suppress H_adv where mask_obj is True
        # Missing: mean suppression of real-object cells
        #   H_0 is detached baseline; (H_0 - H_adv) > 0 when we push detections down
        missing_proxy = ((H_0 - H_adv)[mask_obj].mean()
                         if mask_obj.sum() > 0
                         else torch.tensor(0.0, device='cuda'))
        
        # 2. New Proxy: Excite H_adv where mask_bg is True
        new_proxy = H_adv[mask_bg].mean() if mask_bg.sum() > 0 else torch.tensor(0.0, device='cuda')

        # |∆N|: total heatmap change (captures both suppression and hallucination)
        delta_N_proxy = (H_adv - H_0).abs().mean()

        # J(theta) = |∆N| + alpha * Missing + beta * New
        loss = delta_N_proxy + alpha * missing_proxy + beta * new_proxy

        # # Combined Objective (Minimize this to Maximize Disruption)
        # # We subtract loss_new because we want the optimizer to push fake detections UP
        # loss = (alpha * loss_missing) - (beta * loss_new)

        (-loss).backward()

        # --- THETA DIAGNOSTIC: print every step ---
        grad_str = "None" if theta.grad is None else f"[{theta.grad[0,0].item():.6e}, {theta.grad[0,1].item():.6e}]"

        print(f"  Step {step:03d} | "
              f"theta=[{theta[0,0].item():+.6f}, {theta[0,1].item():+.6f}] | "
              f"theta.grad={grad_str} | "
              f"loss={loss.item():.6f} | "
              f"missing={missing_proxy.item():.6f} | "
              f"new={new_proxy.item():.6f}")

        optimizer.step()

        # --- SAVE THETA AT EACH STEP ---
        current_x = theta[0,0].item()
        current_y = theta[0,1].item()
        theta_x_history.append(current_x)
        theta_y_history.append(current_y)
        loss_history.append(loss.item()) # <-- NEW: Saving the loss

        # === ADD THIS ===
        # Y range is [-37.99, 39.85], X range is [-57.64, 45.68]
        with torch.no_grad():
            theta[0, 0].clamp_(-3.5, 3.5)  # X
            theta[0, 1].clamp_(-3.5, 3.5)  # Y
        # ================

    # Cleanup
    for h in handles:
        h.remove()

    print("\nOptimization Complete.")

    # --- PLOT THE RESULTS ---
    plot_theta_trajectory(theta_x_history, theta_y_history, loss_history) # <-- UPDATED
    return theta.detach()

# ---------------------------------------------------------
# NEW 3-PANEL PLOTTING FUNCTION
# ---------------------------------------------------------
def plot_theta_trajectory(x_hist, y_hist, loss_hist):
    import matplotlib.pyplot as plt
    """Generates a 3-part plot showing loss, coordinate convergence, and 2D trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Changed to 3 subplots
    epochs = range(len(x_hist))

    # Plot 1: Loss Convergence (NEW)
    axes[0].plot(epochs, loss_hist, color='black', linewidth=2)
    axes[0].set_title('Loss Value over Time')
    axes[0].set_xlabel('Optimization Step (Epoch)')
    axes[0].set_ylabel('Loss (Lower is Better)')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # Plot 2: X and Y over Epochs
    axes[1].plot(epochs, x_hist, label='Theta X', color='blue', linewidth=2)
    axes[1].plot(epochs, y_hist, label='Theta Y', color='red', linewidth=2)
    axes[1].set_title('Theta Coordinates over Time')
    axes[1].set_xlabel('Optimization Step (Epoch)')
    axes[1].set_ylabel('Physical Distance (Meters)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)

    # Plot 3: 2D Spatial Trajectory (Bird's Eye View)
    axes[2].plot(x_hist, y_hist, color='purple', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[2].scatter(x_hist[0], y_hist[0], color='green', marker='o', s=100, label='Start (0,0)')
    axes[2].scatter(x_hist[-1], y_hist[-1], color='red', marker='*', s=200, label=f'End ({x_hist[-1]:.2f}, {y_hist[-1]:.2f})')
    
    axes[2].set_xlim(min(x_hist) - 0.5, max(x_hist) + 0.5)
    axes[2].set_ylim(min(y_hist) - 0.5, max(y_hist) + 0.5)
    
    axes[2].set_title('2D Search Trajectory')
    axes[2].set_xlabel('X Coordinate')
    axes[2].set_ylabel('Y Coordinate')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = '/isaacsim_projects/AdversarialGhosting/report/model/theta_convergence_plot.png'
    plt.savefig(out_file, dpi=150)
    print(f"\n✅ Convergence plot saved to: {out_file}")


# ---------------------------------------------------------
# 5. MAIN
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

    # ========================== NEW PART ==========================
    # Prepare Data
    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    # === POINT CLOUD RANGE CHECK ===
    points = np.load('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy')
    print(f"Point cloud shape: {points.shape}")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    # ================================

    # This can be deleted not useful for the actual optimization
    def check_activation_magnitude(model, data_dict):
        mags = {}
        def hook(name):
            def fn(module, inp, out):
                mags[name] = out.features.norm(dim=1).mean().item()
            return fn
        handles = []
        for name, module in model.named_modules():
            if name in ['backbone_3d.conv4', 'backbone_3d.shared_conv']:
                handles.append(module.register_forward_hook(hook(name)))
        
        model.eval()  # ← THIS IS MISSING IN YOUR VERSION

        with torch.no_grad():
            model(data_dict)
        for h in handles: h.remove()
        print(f"\nActivation norms: {mags}")
    # =============================== END ==========================

    # Load Surrogate
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()

    # ===================================== NEW ==================================
    with torch.no_grad():
        test_theta = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device='cuda')
        delta = surrogate(test_theta)
        print(f"Surrogate output norm:   {delta.norm().item():.4f}")
        print(f"Surrogate output range:  [{delta.min().item():.4f}, {delta.max().item():.4f}]")
        print(f"Surrogate output sample: {delta[0, :8]}")

    check_activation_magnitude(model, data_dict)
    # ===================================== END ===================================

    # Execute Part C: Optimize Placement
    optimal_theta = optimize_placement_j_theta(
        model=model, 
        surrogate=surrogate, 
        data_dict=data_dict, 
        layer_to_attack=LAYER_TO_ATTACK,
        head_idx=0,         # Change if targeting a different class head
        num_iters=500,      # Adjust iterations as needed
        lr=0.1,            # Learning rate for position updates
        alpha=10.0,         # Weight for suppressing existing objects
        beta=5.0            # Weight for hallucinating new objects
    )

    print(f"\nFinal Optimal Reflector Placement (Theta): {optimal_theta.cpu().numpy()}")


if __name__ == '__main__':
    main()