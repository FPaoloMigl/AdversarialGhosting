import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils  # <--- ADDED THIS IMPORT
import glob

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/report/model/surrogate_model_F4.pth')
CFG_FILE = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
OUTPUT_DIR = SURROGATE_PATH.parent
LAYER_TO_ATTACK = 'backbone_3d.conv4' # The layer name in VoxelNeXt to inject noise (F4)

# Weights for the Multi-Objective Loss
ALPHA = 1.0  # Weight for Suppression (Missing Detections)
BETA = 1.0   # Weight for Ghosting (New Detections)

# ---------------------------------------------------------
# 1. DEFINE THE SURROGATE MODEL (Must match training arch!)
# ---------------------------------------------------------
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=128):
        super(SurrogateModel, self).__init__()
        # Enhanced MLP Architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, output_dim) 
        )
        
    def forward(self, x):
        return self.net(x)
    
# ---------------------------------------------------------
# 2. THE ADVERSARIAL PIPELINE (Connecting Theta -> Output)
# ---------------------------------------------------------
class AdversarialPipeline(nn.Module):
    def __init__(self, detector_model, surrogate_model, layer_name):
        super().__init__()
        self.detector = detector_model
        self.surrogate = surrogate_model
        self.layer_name = layer_name

        # Freeze EVERYTHING. We only optimize 'self.theta'
        for param in self.detector.parameters(): param.requires_grad = False
        for param in self.surrogate.parameters(): param.requires_grad = False

        self.detector.eval()
        self.surrogate.eval()

        # THE ATTACK PARAMETER (Theta: x, y)
        # We start at (0,0) or a random guess
        self.theta = nn.Parameter(torch.tensor([[0.5, 0.5]], dtype=torch.float32), requires_grad=True)

        self.hook_handle = None

    def _injection_hook(self, module, input, output):
        """
        Injects the surrogate noise into the feature stream.
        """
        # 1. Get Perturbation from Surrogate (Differentiable!)
        delta_phi = self.surrogate(self.theta) # Shape (1, 128)

        # 2. Add to VoxelNeXt Features
        # Handle SparseConvTensor (pcdet uses spconv)
        if hasattr(output, 'features'):
            # Broadcast add: (N_voxels, 128) + (1, 128)
            perturbed_feats = output.features + delta_phi
            # Replace features but keep indices
            output = output.replace_feature(perturbed_feats)
        elif isinstance(output, torch.Tensor):
            # Fallback for dense tensors (N, C, H, W)
            # Add to channel dim (assuming N, C, H, W)
            delta_reshaped = delta_phi.unsqueeze(-1).unsqueeze(-1)
            output = output + delta_reshaped
            
        return output
    
    def forward(self, batch_dict):
        # 1. Locate the target layer
        # Note: In pcdet, layers are often nested. We assume 'backbone_3d' is accessible.
        # You might need to adjust this depending on how you load VoxelNeXt.
        try:
            target_module = dict(self.detector.named_modules())[self.layer_name]
        except KeyError:
            print(f"Error: Layer '{self.layer_name}' not found. Available layers:")
            print(list(self.detector.named_modules()).keys()[:5])
            return None

        # 2. Register Hook
        self.hook_handle = target_module.register_forward_hook(self._injection_hook)
        
        try:
            # 3. Run Detector (The hook will trigger inside)
            preds = self.detector(batch_dict)
            return preds
        finally:
            # 4. Cleanup (Critical!)
            if self.hook_handle:
                self.hook_handle.remove()

# ---------------------------------------------------------
# 3. THE OPTIMIZATION OBJECTIVE J(theta)
# ---------------------------------------------------------
def compute_loss(pred_dict, clean_scores_constant, alpha=1.0, beta=1.0):
    """
    Computes J(theta) = |Delta N| + alpha * Missing + beta * New
    
    Args:
        pred_dict: Output from VoxelNeXt (containing 'cls_preds')
        clean_scores_constant: The frozen scores from the clean run (y0). 
                               Must be DETACHED and same shape as pred_scores.
        alpha: Weight for Suppression (Missing detections)
        beta: Weight for Ghosting (New detections)
    """

    # 1. Get Current Scores (Probabilities)
    # Shape: (Batch, N_anchors, N_classes)
    curr_scores = torch.sigmoid(pred_dict['cls_preds'])

    # Ensure shapes match (Flattening helps sum over all anchors easily)
    curr_flat = curr_scores.view(-1)
    clean_flat = clean_scores_constant.view(-1)

    # -------------------------------------------------------------
    # Term 1: |Delta N| (Absolute change in total number of detections)
    # -------------------------------------------------------------
    N_current = torch.sum(curr_flat)
    N_clean = torch.sum(clean_flat)
    term_delta_N = torch.abs(N_current - N_clean)

    # -------------------------------------------------------------
    # Term 2: Missing (Suppression)
    # "Where did score drop?" -> Clean > Current
    # ReLU(Clean - Current) captures ONLY the drops
    # -------------------------------------------------------------
    # We sum the drops to quantify "how much" was lost
    term_missing = torch.sum(torch.relu(clean_flat - curr_flat))

    # -------------------------------------------------------------
    # Term 3: New (Ghosting)
    # "Where did score rise?" -> Current > Clean
    # ReLU(Current - Clean) captures ONLY the gains
    # -------------------------------------------------------------
    term_new = torch.sum(torch.relu(curr_flat - clean_flat))

    # -------------------------------------------------------------
    # Final Objective J(theta)
    # -------------------------------------------------------------
    J_theta = term_delta_N + (alpha * term_missing) + (beta * term_new)

    # We want to MAXIMIZE J (cause maximum chaos), so we MINIMIZE negative J
    loss = -J_theta
    
    return loss, {
        "J": J_theta.item(),
        "Delta_N": term_delta_N.item(),
        "Missing": term_missing.item(),
        "New": term_new.item()
   }

# ---------------------------------------------------------
# 4. CUSTOM DATASET LOADER (Simple NPY/BIN Reader)
# ---------------------------------------------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            # 1. Load Flat Array first (Don't reshape yet!)
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32)
            
            # 2. Smart Reshape based on total size
            if points.shape[0] % 5 == 0:
                # It's a 5-column file (Waymo style: x, y, z, intensity, elongation)
                points = points.reshape(-1, 5)
            elif points.shape[0] % 4 == 0:
                # It's a 4-column file (KITTI style: x, y, z, intensity)
                points = points.reshape(-1, 4)
            else:
                # Fallback or Error
                print(f"Error: Point cloud size {points.shape[0]} is not divisible by 4 or 5.")
                raise ValueError("Invalid Point Cloud Dimension")

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        # --- FIX: Auto-Match Config ---
        # This part ensures the data matches what the model expects (from your config)
        try:
            src_feature_list = self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list
            required_dim = len(src_feature_list)  # Likely 5 for your config

            if points.shape[-1] < required_dim:
                diff = required_dim - points.shape[-1]
                padding = np.zeros((points.shape[0], diff), dtype=points.dtype)
                points = np.hstack((points, padding))
            
            elif points.shape[-1] > required_dim:
                points = points[:, :required_dim]
                
        except Exception as e:
            pass
        # -------------------------------------------------------------

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


# ---------------------------------------------------------
# 4. MAIN OPTIMIZATION LOOP
# ---------------------------------------------------------
def main():
    # 1. Load Surrogate
    # --- Load Clean Data (Background) ---
    # Update this to point to a valid CLEAN scan (background only)
    DATA_PATH = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy') # or .bin

    logger = common_utils.create_logger()

    if not DATA_PATH.exists():
        print(f"Error: Surrogate not found at {DATA_PATH}")
        return

    print("--- Loading VoxelNeXt ---")
    cfg_from_yaml_file(CFG_FILE, cfg)

    # Build Dataset wrapper
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=DATA_PATH, ext='.npy', logger=None
    )

    # Build Model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Get Batch
    with torch.no_grad():
        data_dict = demo_dataset[0]
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

    print("--- Preparing Attack ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()

    pipeline = AdversarialPipeline(model, surrogate, LAYER_TO_ATTACK)
    pipeline.cuda()

    # Get Baseline
    with torch.no_grad():
        clean_preds = model(data_dict)
        clean_scores_const = torch.sigmoid(clean_preds['cls_preds']).detach()
        print(f"Clean Baseline Activity: {clean_scores_const.sum().item():.4f}")

    # Optimization Loop
    optimizer = optim.Adam([pipeline.theta], lr=0.1)
    print("\n--- STARTING OPTIMIZATION ---")

    best_J = -1
    best_theta = None

    print("\n--- STARTING ATTACK OPTIMIZATION ---")
    print(f"Initial Theta: {pipeline.theta.data[0].numpy()}")

    steps = 100
    history = [] # <--- FIX: Initialize history list
    best_J = -1
    # best_loss = float('inf')
    best_theta = None

    for i in range(steps):
        optimizer.zero_grad()

        # Forward Pass (Differentiable)
        preds = pipeline(data_dict)

        # Compute Loss
        loss, J_val = compute_loss(preds, clean_scores_const)
        
        # Backward Pass
        loss.backward()
        
        # Update Theta
        optimizer.step()

        # Constraint: Clip to physical limits (-3.5m to 3.5m)
        with torch.no_grad():
            pipeline.theta.clamp_(-3.5, 3.5)

        if J_val > best_J:
            best_J = J_val
            best_theta = pipeline.theta.detach().cpu().numpy()[0].copy()

        history.append(J_val)

        if i % 10 == 0:
             theta_curr = pipeline.theta.detach().cpu().numpy()[0]
             print(f"Step {i}: Theta=({theta_curr[0]:.2f}, {theta_curr[1]:.2f}) | Objective J={J_val:.4f}")

    print(f"\n✅ OPTIMAL ATTACK FOUND!")
    print(f"Reflector Position: x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
    print(f"Max Objective J: {best_J:.4f}")
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(history)
    plt.title("Attack Optimization: J(theta)")
    plt.xlabel("Step")
    plt.ylabel("Objective Value (Higher is Better)")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "attack_optimization_J_theta.png")
    print(f"Plot saved to {OUTPUT_DIR}/attack_optimization_J_theta.png")
    
    return best_theta

# # =========================================================
# # MOCK MODEL (FOR TESTING) - REPLACE WITH REAL ONE
# # =========================================================
# class MockVoxelNeXt(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Mock layer structure
#         self.backbone_3d = nn.Module()
#         self.backbone_3d.conv4 = nn.Linear(128, 128) # Placeholder
        
#     def forward(self, batch_dict):
#         # Fake output: (Batch, Anchors, Classes)
#         # Returns random logits
#         return {'cls_preds': torch.randn(1, 100, 3, requires_grad=True)}

if __name__ == '__main__':
    # # --- TO RUN WITH REAL DATA ---
    # # 1. Load VoxelNeXt:
    # #    detector = ... (load your pcdet model)
    # #    detector.cuda()
    
    # # 2. Load One Sample:
    # #    data_dict = ... (load from pkl or dataset)
    # #    data_dict = model.load_data_to_gpu(data_dict) (if needed)

    # # --- DEMO MODE ---
    # print("⚠️  Using Mock Model. Replace with real VoxelNeXt for actual results.")
    # detector = MockVoxelNeXt()
    # # Dummy batch
    # clean_batch = {'points': torch.randn(10, 4)} 
    
    # run_attack(detector, clean_batch)

    main()