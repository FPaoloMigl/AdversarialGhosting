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
# 4. GRADIENT DIAGNOSTIC
# ---------------------------------------------------------
def run_gradient_diagnostic(model, surrogate, data_dict):
    """
    Trace gradient flow through every layer by:
    1. Registering hooks that capture output tensors at each layer
    2. Running forward with injection at conv4
    3. Calling backward from a simple loss on the heatmaps
    4. Checking .grad on every captured tensor using retain_grad()
    """

    theta = nn.Parameter(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device='cuda'),
        requires_grad=True
    )

    # Enable grad on backbone + dense_head so grad_input can flow
    for name, param in model.named_parameters():
        if 'backbone_3d' in name or 'dense_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # ----------------------------------------------------------------
    # Track tensors at every stage
    # ----------------------------------------------------------------
    tracked = {}   # name -> tensor (with retain_grad)
    handles = []

    def make_forward_hook(name):
        def hook(module, inp, output):
            if hasattr(output, 'features'):
                t = output.features
            elif isinstance(output, torch.Tensor):
                t = output
            else:
                return
            t.retain_grad()
            tracked[name] = t
        return hook

    # ----------------------------------------------------------------
    # A) Track backbone layers
    # ----------------------------------------------------------------
    backbone_layers = [
        'backbone_3d.conv_input',
        'backbone_3d.conv1',
        'backbone_3d.conv2',
        'backbone_3d.conv3',
        'backbone_3d.conv4',
        'backbone_3d.conv5',
        'backbone_3d.conv6',
    ]

    modules_dict = dict(model.named_modules())

    for layer_name in backbone_layers:
        if layer_name in modules_dict:
            handles.append(
                modules_dict[layer_name].register_forward_hook(make_forward_hook(layer_name))
            )

    # ----------------------------------------------------------------
    # B) Track dense_head shared_conv and conv_out if they exist
    # ----------------------------------------------------------------
    for extra in ['dense_head.shared_conv', 'dense_head.conv_out']:
        if extra in modules_dict:
            handles.append(
                modules_dict[extra].register_forward_hook(make_forward_hook(extra))
            )

    # ----------------------------------------------------------------
    # C) Track each head's hm output
    # ----------------------------------------------------------------
    for i, head in enumerate(model.dense_head.heads_list):
        name = f'dense_head.heads_list[{i}].hm'
        handles.append(
            head.hm.register_forward_hook(make_forward_hook(name))
        )

    # ----------------------------------------------------------------
    # D) Injection hook at conv4
    # ----------------------------------------------------------------
    def injection_hook(module, inp, output):
        delta_phi = surrogate(theta)                                    # [1, 128]
        delta = delta_phi.expand(output.features.shape[0], -1)         # [N, 128]
        new_output = output.replace_feature(output.features + delta)
        return new_output

    injection_handle = modules_dict[LAYER_TO_ATTACK].register_forward_hook(injection_hook)

    # ----------------------------------------------------------------
    # E) Also track the surrogate output and delta directly
    # ----------------------------------------------------------------
    surrogate_out = surrogate(theta)
    surrogate_out.retain_grad()
    tracked['surrogate_output'] = surrogate_out

    # ----------------------------------------------------------------
    # F) Forward pass
    # ----------------------------------------------------------------
    print("=" * 70)
    print("RUNNING FORWARD PASS WITH INJECTION")
    print("=" * 70)

    model.zero_grad()
    surrogate.zero_grad()
    if theta.grad is not None:
        theta.grad.zero_()

    _ = model(data_dict)

    # ----------------------------------------------------------------
    # G) Build loss from all captured heatmaps
    # ----------------------------------------------------------------
    heatmap_keys = [k for k in tracked if 'heads_list' in k and 'hm' in k]
    if not heatmap_keys:
        print("❌ No heatmaps captured!")
        for h in handles:
            h.remove()
        injection_handle.remove()
        return

    loss = sum(tracked[k].sum() for k in heatmap_keys)
    print(f"\nLoss = {loss.item():.6f} (sum of all heatmap features)")

    # ----------------------------------------------------------------
    # H) Backward
    # ----------------------------------------------------------------
    print("\nRUNNING BACKWARD...")
    loss.backward()

    # ----------------------------------------------------------------
    # I) Report gradient status at every tracked point
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GRADIENT FLOW REPORT")
    print("=" * 70)

    # Order: backbone → dense_head → heatmaps → surrogate → theta
    report_order = (
        backbone_layers +
        ['dense_head.shared_conv', 'dense_head.conv_out'] +
        sorted(heatmap_keys) +
        ['surrogate_output']
    )

    for name in report_order:
        if name not in tracked:
            print(f"  {'⚪':2s} {name:50s} — NOT CAPTURED")
            continue

        t = tracked[name]
        if t.grad is not None:
            grad_norm = t.grad.abs().sum().item()
            grad_max  = t.grad.abs().max().item()
            status = "✅" if grad_norm > 0 else "⚠️  (all zeros)"
            print(f"  {status:2s} {name:50s} | grad_norm={grad_norm:.6e} | grad_max={grad_max:.6e} | shape={list(t.shape)}")
        else:
            print(f"  {'❌':2s} {name:50s} | grad=None | shape={list(t.shape)}")

    # Theta itself
    print(f"\n  --- THETA ---")
    if theta.grad is not None:
        print(f"  ✅ theta.grad = {theta.grad} | norm={theta.grad.abs().sum().item():.6e}")
    else:
        print(f"  ❌ theta.grad = None")

    # ----------------------------------------------------------------
    # J) Extra: check grad_fn chain from conv4 output
    # ----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("AUTOGRAD GRAPH TRACE FROM conv4 OUTPUT")
    print("=" * 70)

    if 'backbone_3d.conv4' in tracked:
        t = tracked['backbone_3d.conv4']
        print(f"  conv4 output grad_fn: {t.grad_fn}")

        # Walk 5 levels of grad_fn
        fn = t.grad_fn
        for depth in range(5):
            if fn is None:
                print(f"    depth {depth}: None — end of graph")
                break
            print(f"    depth {depth}: {fn.__class__.__name__}")
            next_fns = fn.next_functions
            if next_fns:
                print(f"      next_functions: {[(f[0].__class__.__name__ if f[0] else 'None') for f in next_fns]}")
                # Follow the first branch
                fn = next_fns[0][0] if next_fns[0][0] else None
            else:
                break

    # ----------------------------------------------------------------
    # K) Extra: check grad_fn chain from conv5 input perspective
    # ----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("AUTOGRAD GRAPH TRACE FROM conv5 OUTPUT")
    print("=" * 70)

    if 'backbone_3d.conv5' in tracked:
        t = tracked['backbone_3d.conv5']
        print(f"  conv5 output grad_fn: {t.grad_fn}")

        fn = t.grad_fn
        for depth in range(8):
            if fn is None:
                print(f"    depth {depth}: None — end of graph")
                break
            print(f"    depth {depth}: {fn.__class__.__name__}")
            next_fns = fn.next_functions
            if next_fns:
                names = [(f[0].__class__.__name__ if f[0] else 'None') for f in next_fns]
                print(f"      next_functions: {names}")
                # Follow first non-None branch
                fn = None
                for f in next_fns:
                    if f[0] is not None:
                        fn = f[0]
                        break
            else:
                break

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    for h in handles:
        h.remove()
    injection_handle.remove()

    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


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

    # Load model
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

    # Run diagnostic
    run_gradient_diagnostic(model, surrogate, data_dict)


if __name__ == '__main__':
    main()