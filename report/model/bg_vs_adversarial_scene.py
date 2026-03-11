import torch
import numpy as np
from pathlib import Path

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'

# The two physical point clouds from Isaac Sim
CLEAN_PC_PATH = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy')
ADV_PC_PATH   = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_test_F8_1000.npy')

SCORE_THRESH  = 0.28  # Keep this low to see degraded objects and faint ghosts

# ---------------------------------------------------------
# 1. DATASET LOADER
# ---------------------------------------------------------
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

def fix_head_indice_keys(model):
    for i, head in enumerate(model.dense_head.heads_list):
        for cur_name in head.sep_head_dict:
            for module in head.__getattr__(cur_name).modules():
                if hasattr(module, 'indice_key') and module.indice_key is not None:
                    module.indice_key = f"{module.indice_key}_head{i}"

# ---------------------------------------------------------
# 2. EVALUATION & SUMMARY LOGIC
# ---------------------------------------------------------
def get_detections_summary(pred_dicts, class_names, threshold=0.05):
    boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    labels = pred_dicts[0]['pred_labels'].cpu().numpy()
    
    mask = scores > threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    
    detections = []
    for i in range(len(labels)):
        detections.append({
            'class': class_names[labels[i] - 1],
            'score': scores[i],
            'x': boxes[i][0], 'y': boxes[i][1], 'z': boxes[i][2]
        })
    return detections

def print_summary_table(title, detections):
    print(f"\n{'='*75}")
    print(f"{title}")
    print(f"{'='*75}")
    
    if not detections:
        print(f"  [No objects detected above {SCORE_THRESH*100}% threshold]")
        return

    print(f"{'ID':<4} | {'Class':<12} | {'Score':<7} | {'Position (X, Y, Z)'}")
    print("-" * 75)
    for idx, det in enumerate(detections):
        score_str = f"{det['score']:.1%}"
        pos_str = f"X: {det['x']:>6.2f}, Y: {det['y']:>6.2f}, Z: {det['z']:>5.2f}"
        print(f"#{idx:<2} | {det['class']:<12} | {score_str:<7} | {pos_str}")

# ---------------------------------------------------------
# 3. INFERENCE WRAPPER
# ---------------------------------------------------------
def run_inference(model, pc_path, logger):
    dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, pc_path=pc_path, logger=logger)
    data_dict = dataset[0]
    data_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    
    with torch.no_grad():
        preds, _ = model.forward(data_dict)
    return preds

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    logger = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    
    print("--- Loading VoxelNeXt ---")
    # We use the clean path just to initialize the architecture properly
    init_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, pc_path=CLEAN_PC_PATH, logger=logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), init_dataset)
    
    checkpoint = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.cuda()
    model.eval()
    fix_head_indice_keys(model)
    
    # --- RUN 1: CLEAN BASELINE ---
    if CLEAN_PC_PATH.exists():
        clean_preds = run_inference(model, CLEAN_PC_PATH, logger)
        clean_summary = get_detections_summary(clean_preds, cfg.CLASS_NAMES, threshold=SCORE_THRESH)
        print_summary_table("RUN 1: CLEAN BASELINE (sample_bg.npy)", clean_summary)
    else:
        print(f"\nERROR: Clean file not found at {CLEAN_PC_PATH}")

    # --- RUN 2: PHYSICAL ATTACK ---
    if ADV_PC_PATH.exists():
        adv_preds = run_inference(model, ADV_PC_PATH, logger)
        adv_summary = get_detections_summary(adv_preds, cfg.CLASS_NAMES, threshold=SCORE_THRESH)
        print_summary_table("RUN 2: PHYSICAL ADVERSARIAL SCENE (sample_test.npy)", adv_summary)
    else:
        print(f"\nERROR: Adversarial file not found at {ADV_PC_PATH}")

if __name__ == '__main__':
    main()