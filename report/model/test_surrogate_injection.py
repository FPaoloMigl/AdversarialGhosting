import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import glob

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/report/model/surrogate_model_F4.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
DATA_PATH = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_test.npy')

LAYER_TO_ATTACK = 'backbone_3d.conv4'
DELTA_SCALE = 10.0  # Must match what you used in the optimization!

# The specific coordinate you want to test
TEST_X = 1.02
TEST_Y = 15.89

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
# 2. DATASET LOADER
# ---------------------------------------------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.sample_file_list = [self.root_path]

    def __len__(self): return len(self.sample_file_list)

    def __getitem__(self, index):
        points = np.load(self.sample_file_list[index])
        required_dim = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        if points.shape[-1] < required_dim:
            padding = np.zeros((points.shape[0], required_dim - points.shape[-1]), dtype=points.dtype)
            points = np.hstack((points, padding))
        elif points.shape[-1] > required_dim:
            points = points[:, :required_dim]

        return self.prepare_data(data_dict={'points': points, 'frame_id': index})

def fix_head_indice_keys(model):
    for i, head in enumerate(model.dense_head.heads_list):
        for cur_name in head.sep_head_dict:
            for module in head.__getattr__(cur_name).modules():
                if hasattr(module, 'indice_key') and module.indice_key is not None:
                    module.indice_key = f"{module.indice_key}_head{i}"

# ---------------------------------------------------------
# 3. EVALUATION LOGIC
# ---------------------------------------------------------
def print_detections(pred_dicts, class_names, title="Detections"):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    boxes = pred_dicts[0]['pred_boxes']
    scores = pred_dicts[0]['pred_scores']
    labels = pred_dicts[0]['pred_labels']
    
    mask = scores > 0.01
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    
    if len(labels) == 0:
        print("No objects detected above 28% confidence.")
    else:
        for i in range(len(labels)):
            class_name = class_names[labels[i].item() - 1]
            x, y, z = boxes[i][:3].cpu().numpy()
            print(f"Object {i}: {class_name:10} | Score: {scores[i].item():.2%} | Pos: [X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f}]")

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    logger = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    
    # Load Data
    dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, root_path=DATA_PATH, logger=logger)
    data_dict = dataset[0]
    data_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    
    # Load VoxelNeXt
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    checkpoint = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.cuda()
    model.eval()
    fix_head_indice_keys(model)
    
    # Load Surrogate
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()

    # --- RUN 1: CLEAN BASELINE ---
    with torch.no_grad():
        clean_preds, _ = model.forward(data_dict)
    print_detections(clean_preds, cfg.CLASS_NAMES, title="RUN 1: CLEAN BASELINE (No Perturbation)")

    # --- SETUP INJECTION ---
    theta = torch.tensor([[TEST_X, TEST_Y]], dtype=torch.float32, device='cuda')
    
    with torch.no_grad():
        delta_phi = surrogate(theta) * DELTA_SCALE

    def injection_hook(module, inp, output):
        delta_expanded = delta_phi.expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + delta_expanded)

    # Attach the hook to conv4
    modules_dict = dict(model.named_modules())
    hook_handle = modules_dict[LAYER_TO_ATTACK].register_forward_hook(injection_hook)

    # --- RUN 2: ADVERSARIAL INJECTION ---
    # We must pass the data_dict through the model again, this time the hook will trigger!
    with torch.no_grad():
        adv_preds, _ = model.forward(data_dict)
    
    print_detections(adv_preds, cfg.CLASS_NAMES, title=f"RUN 2: ADVERSARIAL INJECTION (Theta X={TEST_X}, Y={TEST_Y})")

    # Cleanup
    hook_handle.remove()

if __name__ == '__main__':
    main()