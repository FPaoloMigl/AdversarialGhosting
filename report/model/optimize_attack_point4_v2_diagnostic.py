import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import types
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

ALPHA = 1.0
BETA  = 1.0


# ---------------------------------------------------------
# 1. SURROGATE MODEL
# ---------------------------------------------------------
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=128):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256),       nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.net(x)


# ---------------------------------------------------------
# 2. ADVERSARIAL PIPELINE
# ---------------------------------------------------------
class AdversarialPipeline(nn.Module):
    def __init__(self, detector_model, surrogate_model, layer_name):
        super().__init__()
        self.detector  = detector_model
        self.surrogate = surrogate_model
        self.layer_name = layer_name

        # for param in self.detector.parameters():  param.requires_grad = False
        for name, param in self.detector.named_parameters():
            if ('backbone_3d.conv_out' in name or 
                'backbone_3d.shared_conv' in name or
                'dense_head' in name):
                pass  # leave requires_grad=True, set above
            else:
                param.requires_grad = False
        for param in self.surrogate.parameters(): param.requires_grad = False

        self.detector.eval()
        self.surrogate.eval()

        # Theta is the ONLY leaf variable with grad
        self.theta = nn.Parameter(
            torch.tensor([[0.0, 0.0]], dtype=torch.float32), requires_grad=True
        )

        self.capture_handles  = []
        self.injection_handle = None
        self.captured_heatmaps = []

    def _injection_hook(self, module, input, output):
        # 1. Calculate noise — gradient flows from theta through surrogate
        delta_phi = self.surrogate(self.theta)          # shape [1, 128]

        if hasattr(output, 'features'):
            # Broadcast delta_phi to match all N voxels: [1, 128] -> [N, 128]
            delta_broadcast = delta_phi.expand(output.features.shape[0], -1)
            perturbed_feats = output.features + delta_broadcast
            output = output.replace_feature(perturbed_feats)

            # if output.features.grad_fn is not None:
            #     print(f"  [Injection Hook] SUCCESS — grad_fn: {output.features.grad_fn}")
            # else:
            #     print("  [Injection Hook] WARNING: gradient LOST immediately after injection.")
        return output

    def _capture_hook(self, module, input, output):
        if hasattr(output, 'features'):
            self.captured_heatmaps.append(output.features)
        elif isinstance(output, torch.Tensor):
            self.captured_heatmaps.append(output)

    def forward(self, batch_dict):
        self.captured_heatmaps = []

        # 1. Register injection hook once
        if self.injection_handle is None:
            modules = dict(self.detector.named_modules())
            if self.layer_name in modules:
                target_module = modules[self.layer_name]
                self.injection_handle = target_module.register_forward_hook(
                    self._injection_hook
                )
            else:
                print(f"CRITICAL ERROR: Layer '{self.layer_name}' not found!")
                return None

        # 2. Register capture hooks on all detection heads once
        if not self.capture_handles:
            try:
                heads = self.detector.dense_head.heads_list
                for head in heads:
                    self.capture_handles.append(
                        head.hm.register_forward_hook(self._capture_hook)
                    )
            except AttributeError:
                print("Error: Could not find heads_list.")
                return None

        # 3. Run the full detector
        _ = self.detector(batch_dict)

        if not self.captured_heatmaps:
            return None

        return self.captured_heatmaps

    def cleanup(self):
        if self.injection_handle:
            self.injection_handle.remove()
        for h in self.capture_handles:
            h.remove()


# ---------------------------------------------------------
# 3. LOSS
# ---------------------------------------------------------
def compute_loss(curr_maps_list, clean_maps_list):
    total_J = 0.0

    for curr_map, clean_map in zip(curr_maps_list, clean_maps_list):
        curr_flat  = torch.sigmoid(curr_map).view(-1)
        clean_flat = torch.sigmoid(clean_map).view(-1)

        delta_N = torch.abs(torch.sum(curr_flat) - torch.sum(clean_flat))
        missing = torch.sum(torch.relu(clean_flat - curr_flat))
        new     = torch.sum(torch.relu(curr_flat  - clean_flat))

        total_J += delta_N + (ALPHA * missing) + (BETA * new)

    return -total_J, total_J.item()


# ---------------------------------------------------------
# 4. DATASET
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

    def __len__(self):
        return len(self.sample_file_list)

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
            points = points[:, :required_dim]

        input_dict = {'points': points, 'frame_id': index}
        data_dict  = self.prepare_data(data_dict=input_dict)
        return data_dict


# ---------------------------------------------------------
# 5. GRADIENT FLOW DIAGNOSTIC
# ---------------------------------------------------------
def diagnose_gradient_flow(pipeline, data_dict):
    """
    Traces gradient flow through every module after the injection point.
    The first line showing grad_fn=None is the exact break point.
    """
    handles  = []
    backbone = pipeline.detector.backbone_3d

    def make_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'features'):
                feat = output.features
                print(f"  [{name:<20}] sparse  | shape={str(feat.shape):<22} | "
                      f"requires_grad={feat.requires_grad} | grad_fn={feat.grad_fn}")
            elif isinstance(output, torch.Tensor):
                print(f"  [{name:<20}] tensor  | shape={str(output.shape):<22} | "
                      f"requires_grad={output.requires_grad} | grad_fn={output.grad_fn}")
            else:
                print(f"  [{name:<20}] unknown | type={type(output)}")
        return hook

    layers_to_watch = [
        ('conv5',       backbone.conv5),
        ('conv6',       backbone.conv6),
        ('conv_out',    backbone.conv_out),
        ('shared_conv', backbone.shared_conv),
    ]
    for name, module in layers_to_watch:
        handles.append(module.register_forward_hook(make_hook(name)))

    # Wrap bev_out to see input AND output grad_fn
    original_bev_out = backbone.bev_out
    def traced_bev_out(x_conv):
        print(f"  [{'bev_out INPUT':<20}] sparse  | shape={str(x_conv.features.shape):<22} | "
              f"requires_grad={x_conv.features.requires_grad} | grad_fn={x_conv.features.grad_fn}")
        result = original_bev_out(x_conv)
        print(f"  [{'bev_out OUTPUT':<20}] sparse  | shape={str(result.features.shape):<22} | "
              f"requires_grad={result.features.requires_grad} | grad_fn={result.features.grad_fn}")
        return result
    backbone.bev_out = traced_bev_out

    print("\n" + "="*70)
    print("  GRADIENT FLOW DIAGNOSTIC (first None = break point)")
    print("="*70)
    _ = pipeline(data_dict)
    print("="*70 + "\n")

    # Cleanup
    for h in handles:
        h.remove()
    backbone.bev_out = original_bev_out


# ---------------------------------------------------------
# 6. MAIN
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
    model.eval()

    # To check!!
    for param in model.backbone_3d.conv_out.parameters():
        param.requires_grad = True
    for param in model.backbone_3d.shared_conv.parameters():
        param.requires_grad = True
    for param in model.dense_head.parameters():   # ← ADD THIS
        param.requires_grad = True

    with torch.no_grad():
        data_dict = demo_dataset[0]
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

    print(f"--- Preparing Attack on '{LAYER_TO_ATTACK}' ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()

    pipeline = AdversarialPipeline(model, surrogate, LAYER_TO_ATTACK)
    pipeline.cuda()

    # ------------------------------------------------------------------
    # Clean baseline (no grad needed)
    # ------------------------------------------------------------------
    print("--- Computing Clean Baseline ---")
    with torch.no_grad():
        clean_maps_list = pipeline(data_dict)
        if clean_maps_list is None:
            print("Failed to capture baseline. Exiting.")
            return
        clean_maps_const = [t.detach() for t in clean_maps_list]
        act = sum([torch.sigmoid(t).sum().item() for t in clean_maps_const])
        print(f"Clean Baseline Activity: {act:.4f}")

    # ------------------------------------------------------------------
    # GRADIENT FLOW DIAGNOSTIC — run once before optimization
    # ------------------------------------------------------------------
    diagnose_gradient_flow(pipeline, data_dict)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    optimizer = optim.Adam([pipeline.theta], lr=0.1)
    print("--- STARTING OPTIMIZATION ---")

    history   = []
    best_J    = -1
    best_theta = None

    try:
        for i in range(50):
            optimizer.zero_grad()

            curr_maps_list = pipeline(data_dict)

            # ADD THIS
            for i, hm in enumerate(curr_maps_list):
                print(f"  heatmap[{i}] grad_fn={hm.grad_fn}, requires_grad={hm.requires_grad}")

            if curr_maps_list is None:
                print("No heatmaps captured — aborting.")
                break

            loss, J_val = compute_loss(curr_maps_list, clean_maps_const)

            loss.backward()

            print(f"  theta.grad = {pipeline.theta.grad}")

            optimizer.step()

            with torch.no_grad():
                pipeline.theta.clamp_(-3.5, 3.5)

                # To check
                model.backbone_3d.conv_out.zero_grad()
                model.backbone_3d.shared_conv.zero_grad()

            if J_val > best_J:
                best_J    = J_val
                best_theta = pipeline.theta.detach().cpu().numpy()[0].copy()

            history.append(J_val)

            if i % 10 == 0:
                theta_curr = pipeline.theta.detach().cpu().numpy()[0]
                print(f"  Step {i:>3}: theta=({theta_curr[0]:+.3f}, {theta_curr[1]:+.3f}) | J={J_val:.4f}")

    except Exception as e:
        import traceback
        print(f"\nError during optimization:\n{traceback.format_exc()}")
    finally:
        pipeline.cleanup()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if best_theta is not None:
        print(f"\n✅  OPTIMAL ATTACK FOUND")
        print(f"    Reflector position : x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
        print(f"    Max objective J    : {best_J:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(history)
        plt.xlabel("Iteration")
        plt.ylabel("J(θ)")
        plt.title("Attack Optimization — Objective over Iterations")
        plt.tight_layout()
        out_path = OUTPUT_DIR / "attack_optimization_J_theta.png"
        plt.savefig(out_path)
        print(f"    Plot saved to      : {out_path}")
    else:
        print("\n❌  Optimization failed — check diagnostic output above.")


if __name__ == '__main__':
    main()