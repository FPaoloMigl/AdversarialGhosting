import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/report/model/surrogate_model_F4.pth')
CFG_FILE = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml' 
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
OUTPUT_DIR = SURROGATE_PATH.parent

# --- THE FIX: TARGET THE SPECIFIC BLOCK ---
LAYER_TO_ATTACK = 'backbone_3d.conv4' 

# Weights
ALPHA = 1.0  
BETA = 1.0   

# ---------------------------------------------------------
# 1. SURROGATE MODEL
# ---------------------------------------------------------
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=128):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, output_dim) 
        )
    def forward(self, x): return self.net(x)
    
# ---------------------------------------------------------
# 2. ADVERSARIAL PIPELINE
# ---------------------------------------------------------
class AdversarialPipeline(nn.Module):
    def __init__(self, detector_model, surrogate_model, layer_name):
        super().__init__()
        self.detector = detector_model
        self.surrogate = surrogate_model
        self.layer_name = layer_name

        for param in self.detector.parameters(): param.requires_grad = False
        for param in self.surrogate.parameters(): param.requires_grad = False

        self.detector.eval()
        self.surrogate.eval()

        # Theta is the ONLY leaf variable with grad
        self.theta = nn.Parameter(torch.tensor([[0.0, 0.0]], dtype=torch.float32), requires_grad=True)
        
        self.capture_handles = []
        self.injection_handle = None
        self.captured_heatmaps = []

    def _injection_hook(self, module, input, output):

        # 1. Calculate Noise (Gradient flows from Theta)
        delta_phi = self.surrogate(self.theta) 
        
        # 2. Inject
        if hasattr(output, 'features'):
            # DIAGNOSTIC 1: Check if delta_phi has a gread_fn
            # print(f"DEBUG [Injection]: delta_phi grad_fn: {delta_phi.grad_fn}")

            # The addition creates a new tensor connected to the graph
            perturbed_feats = output.features + delta_phi
            
            # replace_feature keeps the structure but swaps the data
            output = output.replace_feature(perturbed_feats)

            # # DIAGNOSTIC 2: Check if output.features now has a grad_fn
            # print(f"DEBUG [Injection]: features grad_fn: {output.features.grad_fn}")
            # DEBUG: Verify the link is still alive here
            if output.features.requires_grad:
                print(f"DEBUG: Layer 4 injection SUCCESS. Grad_fn: {output.features.grad_fn}")
            else:
                print("DEBUG: WARNING! Layer 4 injection LOST GRADIENT immediately.")

        return output

    def _capture_hook(self, module, input, output):
        # Capture features (output.features is a tensor)
        if hasattr(output, 'features'):
            self.captured_heatmaps.append(output.features)
        elif isinstance(output, torch.Tensor):
            self.captured_heatmaps.append(output)

    def forward(self, batch_dict):
        self.captured_heatmaps = []

        # 1. Hook Injection (The specific block)
        if self.injection_handle is None:
            # We look up the module using the string name
            modules = dict(self.detector.named_modules())
            if self.layer_name in modules:
                target_module = modules[self.layer_name]
                self.injection_handle = target_module.register_forward_hook(self._injection_hook)
            else:
                print(f"CRITICAL ERROR: Layer {self.layer_name} not found!")
                return None

        # 2. Hook All Heads
        if not self.capture_handles:
            try:
                heads = self.detector.dense_head.heads_list
                for head in heads:
                    self.capture_handles.append(head.hm.register_forward_hook(self._capture_hook))
            except AttributeError:
                print("Error: Could not find heads_list.")
                return None

        # 3. Run Model
        _ = self.detector(batch_dict)
        
        if not self.captured_heatmaps:
            return None
            
        return self.captured_heatmaps 

    def cleanup(self):
        if self.injection_handle: self.injection_handle.remove()
        for h in self.capture_handles: h.remove()

# ---------------------------------------------------------
# 3. LOSS
# ---------------------------------------------------------
def compute_loss(curr_maps_list, clean_maps_list):
    total_J = 0.0
    
    for curr_map, clean_map in zip(curr_maps_list, clean_maps_list):
        curr_flat = torch.sigmoid(curr_map).view(-1)
        clean_flat = torch.sigmoid(clean_map).view(-1)

        # Objective J(theta)
        delta_N = torch.abs(torch.sum(curr_flat) - torch.sum(clean_flat))
        missing = torch.sum(torch.relu(clean_flat - curr_flat))
        new = torch.sum(torch.relu(curr_flat - clean_flat))

        total_J += delta_N + (ALPHA * missing) + (BETA * new)

    return -total_J, total_J.item()

# ---------------------------------------------------------
# 4. DATASET
# ---------------------------------------------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        self.sample_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        self.sample_file_list.sort()

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32)
            points = points.reshape(-1, 5) if points.shape[0] % 5 == 0 else points.reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        
        required_dim = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        if points.shape[-1] < required_dim:
            padding = np.zeros((points.shape[0], required_dim - points.shape[-1]))
            points = np.hstack((points, padding))
        elif points.shape[-1] > required_dim:
            points = points[:, :required_dim]
            
        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------
def main():
    DATA_PATH = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy') 
    logger = common_utils.create_logger()

    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("--- Loading VoxelNeXt ---")
    cfg_from_yaml_file(CFG_FILE, cfg)
    demo_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False, root_path=DATA_PATH, ext='.npy', logger=logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), demo_dataset)
    model.load_params_from_file(CKPT_FILE, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        data_dict = demo_dataset[0]
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

    print(f"--- Preparing Attack on {LAYER_TO_ATTACK} ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()

    pipeline = AdversarialPipeline(model, surrogate, LAYER_TO_ATTACK)
    pipeline.cuda()

    print("--- Computing Clean Baseline ---")
    with torch.no_grad():
        clean_maps_list = pipeline(data_dict) 
        if clean_maps_list is None:
            print("Failed to capture baseline. Exiting.")
            return
        clean_maps_const = [t.detach() for t in clean_maps_list]
        act = sum([torch.sigmoid(t).sum().item() for t in clean_maps_const])
        print(f"Clean Baseline Activity: {act:.4f}")

    optimizer = optim.Adam([pipeline.theta], lr=0.1)
    print("\n--- STARTING OPTIMIZATION ---")

    history = [] 
    best_J = -1
    best_theta = None

    try:
        for i in range(50):
            optimizer.zero_grad()

            curr_maps_list = pipeline(data_dict)
            loss, J_val = compute_loss(curr_maps_list, clean_maps_const)
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pipeline.theta.clamp_(-3.5, 3.5)

            if J_val > best_J:
                best_J = J_val
                best_theta = pipeline.theta.detach().cpu().numpy()[0].copy()
            
            history.append(J_val) 

            if i % 10 == 0:
                 theta_curr = pipeline.theta.detach().cpu().numpy()[0]
                 print(f"Step {i}: Theta=({theta_curr[0]:.2f}, {theta_curr[1]:.2f}) | Objective J={J_val:.4f}")

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        pipeline.cleanup()

    if best_theta is not None:
        print(f"\n✅ OPTIMAL ATTACK FOUND!")
        print(f"Reflector Position: x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
        print(f"Max Objective J: {best_J:.4f}")
        
        plt.figure(figsize=(8,5))
        plt.plot(history)
        plt.title("Attack Optimization")
        plt.savefig(OUTPUT_DIR / "attack_optimization_J_theta.png")
        print(f"Plot saved to {OUTPUT_DIR}/attack_optimization_J_theta.png")
    else:
        print("\nOptimization Failed.")

if __name__ == '__main__':
    main()