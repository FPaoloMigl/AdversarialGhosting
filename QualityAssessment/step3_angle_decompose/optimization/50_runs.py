import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import glob
import os
import json

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3_angle_decompose/model/surrogate_model_F6_v3_sincos.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
LAYER_TO_ATTACK = 'backbone_3d.conv6'

XY_BOUNDS_MIN = torch.tensor([-30.0, -30.0], device='cuda')
XY_BOUNDS_MAX = torch.tensor([30.0, 30.0], device='cuda')
ANGLE_MIN = 0.0
ANGLE_MAX = 360.0

OUTPUT_DIR = '/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3_angle_decompose/optimization/gmm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_RUNS = 50
TOP_PERCENT = 0.20
NUM_ITERS = 500


# ---------------------------------------------------------
# 1. SURROGATE MODEL (sin/cos, input_dim=4)
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
    xy_norm = (theta[:, :2] - XY_BOUNDS_MIN) / (XY_BOUNDS_MAX - XY_BOUNDS_MIN)
    xy_norm = xy_norm * 2.0 - 1.0
    angle_rad = theta[:, 2:3] * (3.14159265358979 / 180.0)
    sin_a = torch.sin(angle_rad)
    cos_a = torch.cos(angle_rad)
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
# 4. SINGLE OPTIMIZATION RUN
# ---------------------------------------------------------
def optimize_single_run(model, surrogate, data_dict, layer_to_attack, init_val,
                        head_idx=0, num_iters=500, lr=0.05, alpha=10.0, beta=5.0):
    
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    for name, module in model.named_modules():
        if 'backbone_3d' in name: module.train()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): module.eval()

    with torch.no_grad():
        _ = model(data_dict)
        clean_hm_logits = model.dense_head.forward_ret_dict['pred_dicts'][head_idx]['hm']
        H_0 = torch.sigmoid(clean_hm_logits).detach()

    mask_obj = (H_0 > 0.1)
    mask_bg  = (H_0 < 0.05)

    theta = nn.Parameter(torch.tensor([init_val], dtype=torch.float32, device='cuda'), requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)

    tracked_hm, handles = {}, []

    def hm_hook(module, inp, output):
        tracked_hm['adv_hm'] = output.features if hasattr(output, 'features') else output

    def injection_hook(module, inp, output):
        theta_encoded = encode_theta_for_surrogate(theta)
        delta = surrogate(theta_encoded)
        delta = delta.expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + delta)

    modules_dict = dict(model.named_modules())
    handles.append(model.dense_head.heads_list[head_idx].hm.register_forward_hook(hm_hook))
    handles.append(modules_dict[layer_to_attack].register_forward_hook(injection_hook))

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

        with torch.no_grad():
            theta[0, 0].clamp_(XY_BOUNDS_MIN[0].item(), XY_BOUNDS_MAX[0].item())
            theta[0, 1].clamp_(XY_BOUNDS_MIN[1].item(), XY_BOUNDS_MAX[1].item())
            theta[0, 2].clamp_(ANGLE_MIN, ANGLE_MAX)

    for h in handles: h.remove()
    
    final_theta = theta.detach().cpu().numpy()[0]
    return {
        'x': float(final_theta[0]),
        'y': float(final_theta[1]),
        'angle': float(final_theta[2]),
        'loss': float(loss.item())
    }


# ---------------------------------------------------------
# 5. MAIN: PHASE 1 + PHASE 2
# ---------------------------------------------------------
def main():
    DATA_PATH = Path('/isaacsim_projects/AdversarialGhosting/QualityAssessment/step3/isaacsim/data/ply/sample_bg.npy')
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

    # ==========================================
    # PHASE 1: Run N optimizations
    # ==========================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: RUNNING {NUM_RUNS} OPTIMIZATIONS ({NUM_ITERS} steps each)")
    print(f"{'='*60}")

    results = []
    for i in range(NUM_RUNS):
        start_x = np.random.uniform(XY_BOUNDS_MIN[0].item(), XY_BOUNDS_MAX[0].item())
        start_y = np.random.uniform(XY_BOUNDS_MIN[1].item(), XY_BOUNDS_MAX[1].item())
        start_angle = np.random.uniform(ANGLE_MIN, ANGLE_MAX)
        
        print(f"  Run {i+1:02d}/{NUM_RUNS} | Init: [X:{start_x:+.1f}, Y:{start_y:+.1f}, A:{start_angle:.0f}]", end="")
        
        result = optimize_single_run(
            model=model,
            surrogate=surrogate,
            data_dict=data_dict,
            layer_to_attack=LAYER_TO_ATTACK,
            init_val=[start_x, start_y, start_angle],
            num_iters=NUM_ITERS
        )
        
        results.append(result)
        print(f" -> [X:{result['x']:+.2f}, Y:{result['y']:+.2f}, A:{result['angle']:.0f}] loss={result['loss']:.4f}")

    # Save all results
    all_results_path = os.path.join(OUTPUT_DIR, 'all_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to: {all_results_path}")

    # ==========================================
    # PHASE 2: Filter top 20%
    # ==========================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: FILTERING TOP {int(TOP_PERCENT*100)}%")
    print(f"{'='*60}")

    results_sorted = sorted(results, key=lambda r: r['loss'], reverse=True)
    n_top = max(1, int(len(results) * TOP_PERCENT))
    top_results = results_sorted[:n_top]
    
    cutoff_loss = top_results[-1]['loss']
    print(f"Total runs: {len(results)}")
    print(f"Top {n_top} selected (loss >= {cutoff_loss:.4f})")
    
    print(f"\nTop {n_top} placements:")
    for rank, r in enumerate(top_results):
        print(f"  #{rank+1:02d} | X:{r['x']:+.2f}m | Y:{r['y']:+.2f}m | A:{r['angle']:.1f} | Loss:{r['loss']:.4f}")

    # Save top results
    top_results_path = os.path.join(OUTPUT_DIR, 'top_results.json')
    with open(top_results_path, 'w') as f:
        json.dump(top_results, f, indent=2)
    print(f"\nTop results saved to: {top_results_path}")

    print(f"\nPhase 1+2 complete. Run the GMM script next to fit and visualize.")


if __name__ == '__main__':
    main()