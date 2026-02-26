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
# 2. ADVERSARIAL PIPELINE (With DEBUGGING)
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

        self.theta = nn.Parameter(torch.tensor([[0.0, 0.0]], dtype=torch.float32), requires_grad=True)
        self.injection_handle = None
        self.capture_handle = None
        self.latest_cls_preds = None

    def _injection_hook(self, module, input, output):
        delta_phi = self.surrogate(self.theta) 
        if hasattr(output, 'features'):
            perturbed_feats = output.features + delta_phi
            output = output.replace_feature(perturbed_feats)
        return output

    def _capture_hook(self, module, input, output):
        if hasattr(output, 'features'):
            self.latest_cls_preds = output.features
        elif isinstance(output, torch.Tensor):
            self.latest_cls_preds = output

    def forward(self, batch_dict):
        # 1. Hook Injection (F4)
        target_module = dict(self.detector.named_modules())[self.layer_name]
        self.injection_handle = target_module.register_forward_hook(self._injection_hook)

        # 2. Hook Capture (DEBUGGING MODE)
        try:
            # Try the standard name first
            cls_layer = self.detector.dense_head.conv_cls
            self.capture_handle = cls_layer.register_forward_hook(self._capture_hook)
        except AttributeError:
             print("\n" + "="*50)
             print("!!! ERROR: 'dense_head.conv_cls' NOT FOUND !!!")
             print("Printing available layers in 'dense_head' to find the right one:")
             print("="*50)
             print(self.detector.dense_head)
             print("="*50 + "\n")
             return None

        try:
            # 3. Run Model
            _ = self.detector(batch_dict)
            
            if self.latest_cls_preds is None:
                # If we hooked successfully but got no data, print why
                print("Warning: Hook attached but 'latest_cls_preds' is None.")
                
            return self.latest_cls_preds
        finally:
            if self.injection_handle: self.injection_handle.remove()
            if self.capture_handle: self.capture_handle.remove()

# ---------------------------------------------------------
# 3. OPTIMIZATION OBJECTIVE
# ---------------------------------------------------------
def compute_loss(curr_preds, clean_preds_const, alpha=1.0, beta=1.0):
    curr_flat = torch.sigmoid(curr_preds).view(-1)
    clean_flat = torch.sigmoid(clean_preds_const).view(-1)

    # 1. Delta N
    term_delta_N = torch.abs(torch.sum(curr_flat) - torch.sum(clean_flat))
    # 2. Missing
    term_missing = torch.sum(torch.relu(clean_flat - curr_flat))
    # 3. New
    term_new = torch.sum(torch.relu(curr_flat - clean_flat))

    J_theta = term_delta_N + (alpha * term_missing) + (beta * term_new)
    return -J_theta, J_theta.item()

# ---------------------------------------------------------
# 4. DATASET LOADER
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
# 5. MAIN EXECUTION
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
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=DATA_PATH, ext='.npy', logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

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

    # E. BASELINE
    print("--- Computing Clean Baseline ---")
    with torch.no_grad():
        clean_logits = pipeline(data_dict) 
        
        # --- DEBUG CHECK ---
        if clean_logits is None:
            print("Stopping execution due to missing layer.")
            return

        clean_logits_const = clean_logits.detach()
        activity = torch.sigmoid(clean_logits_const).sum().item()
        print(f"Clean Baseline Activity: {activity:.4f}")

    # F. OPTIMIZATION
    optimizer = optim.Adam([pipeline.theta], lr=0.1)
    print("\n--- STARTING OPTIMIZATION ---")

    history = [] 
    best_J = -1
    best_theta = None

    for i in range(50):
        optimizer.zero_grad()

        curr_logits = pipeline(data_dict)
        loss, J_val = compute_loss(curr_logits, clean_logits_const)
        
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

    print(f"\n✅ OPTIMAL ATTACK FOUND!")
    print(f"Reflector Position: x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
    
    plt.figure(figsize=(8,5))
    plt.plot(history)
    plt.savefig(OUTPUT_DIR / "attack_optimization_J_theta.png")
    print(f"Plot saved to {OUTPUT_DIR}/attack_optimization_J_theta.png")

if __name__ == '__main__':
    main()