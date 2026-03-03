import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy.optimize import differential_evolution

# --- IMPORT PCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# === CONFIGURATION ===
SURROGATE_PATH = Path('/isaacsim_projects/AdversarialGhosting/report/model/surrogate_model_F4.pth')
CFG_FILE  = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
CKPT_FILE = '/OpenPCDet/checkpoints/voxelnext.pth'
OUTPUT_DIR = SURROGATE_PATH.parent

LAYER_TO_ATTACK = 'backbone_3d.conv4'

ALPHA = 1.0
BETA  = 1.0

# Bounds for theta (x, y) reflector placement
THETA_BOUNDS = [(-3.5, 3.5), (-3.5, 3.5)]
# THETA_BOUNDS = [(-10.0, 10.0), (-10.0, 10.0)]

# Optimizer choice: 'de' for Differential Evolution, 'cma' for CMA-ES
OPTIMIZER = 'cma'  # change to 'de' for differential evolution


# ---------------------------------------------------------
# 1. SURROGATE MODEL (same as before)
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
# 2. DATASET (same as before)
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
# 3. GRADIENT-FREE ATTACK EVALUATOR
# ---------------------------------------------------------
class GradFreeAttack:
    """
    Evaluates J(theta) using only forward passes.
    No backward, no autograd — completely sidesteps spconv gradient issues.
    """

    def __init__(self, detector, surrogate, layer_name, data_dict, clean_maps):
        self.detector   = detector
        self.surrogate  = surrogate
        self.data_dict  = data_dict
        self.clean_maps = clean_maps  # list of detached heatmap tensors

        # Current theta value (updated each evaluation)
        self.theta_val = torch.tensor([[0.0, 0.0]], device='cuda', dtype=torch.float32)

        # Register injection hook on the target layer
        modules = dict(detector.named_modules())
        if layer_name not in modules:
            raise RuntimeError(f"Layer '{layer_name}' not found in detector.")
        self.injection_hook = modules[layer_name].register_forward_hook(self._inject)

        # Track evaluations for logging
        self.eval_count = 0

    @torch.no_grad()
    def _inject(self, module, inp, output):
        """Forward hook: inject surrogate perturbation at the target layer."""
        delta = self.surrogate(self.theta_val).expand(output.features.shape[0], -1)
        return output.replace_feature(output.features + delta)

    @torch.no_grad()
    def _capture_heatmaps(self):
        """Run detector forward and capture heatmap outputs from all heads."""
        heatmaps = []
        handles  = []

        for head in self.detector.dense_head.heads_list:
            def make_hook(lst):
                def hook(m, i, o):
                    lst.append(o.features if hasattr(o, 'features') else o)
                return hook
            handles.append(head.hm.register_forward_hook(make_hook(heatmaps)))

        self.detector(self.data_dict)

        for h in handles:
            h.remove()

        return heatmaps

    @torch.no_grad()
    def evaluate(self, theta_np):
        """
        Evaluate the attack objective J(theta).

        Args:
            theta_np: numpy array [2] — reflector (x, y) position

        Returns:
            J value (float, higher = more disruption)
        """
        self.theta_val = torch.tensor(
            [theta_np], dtype=torch.float32, device='cuda'
        )

        curr_maps = self._capture_heatmaps()

        J = 0.0
        for curr_map, clean_map in zip(curr_maps, self.clean_maps):
            curr_s  = torch.sigmoid(curr_map).view(-1)
            clean_s = torch.sigmoid(clean_map).view(-1)

            delta_N = torch.abs(curr_s.sum() - clean_s.sum())
            missing = torch.relu(clean_s - curr_s).sum()
            new     = torch.relu(curr_s  - clean_s).sum()

            J += (delta_N + ALPHA * missing + BETA * new).item()

        self.eval_count += 1
        return J

    def objective_for_scipy(self, theta_np):
        """Scipy minimizes, so we negate J to maximize disruption."""
        return -self.evaluate(theta_np)

    def cleanup(self):
        self.injection_hook.remove()


# ---------------------------------------------------------
# 4. CLEAN BASELINE (no injection)
# ---------------------------------------------------------
@torch.no_grad()
def compute_clean_baseline(detector, data_dict):
    """Run detector without any injection and capture clean heatmaps."""
    heatmaps = []
    handles  = []

    for head in detector.dense_head.heads_list:
        def make_hook(lst):
            def hook(m, i, o):
                lst.append(o.features.detach().clone() if hasattr(o, 'features') else o.detach().clone())
            return hook
        handles.append(head.hm.register_forward_hook(make_hook(heatmaps)))

    detector(data_dict)

    for h in handles:
        h.remove()

    activity = sum(torch.sigmoid(t).sum().item() for t in heatmaps)
    return heatmaps, activity


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

    # ------------------------------------------------------------------
    # Load detector
    # ------------------------------------------------------------------
    print("=" * 60)
    print("GRADIENT-FREE ADVERSARIAL ATTACK OPTIMIZATION")
    print("=" * 60)

    print("\n--- Loading VoxelNeXt ---")
    cfg_from_yaml_file(CFG_FILE, cfg)
    demo_dataset = DemoDataset(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES,
        training=False, root_path=DATA_PATH, ext='.npy', logger=logger
    )
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), demo_dataset)
    model.load_params_from_file(CKPT_FILE, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Freeze everything — no gradients needed at all
    for param in model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    # ------------------------------------------------------------------
    # Load surrogate
    # ------------------------------------------------------------------
    print("--- Loading Surrogate Model ---")
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH))
    surrogate.cuda()
    surrogate.eval()
    for param in surrogate.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Compute clean baseline (no reflector)
    # ------------------------------------------------------------------
    print("--- Computing Clean Baseline ---")
    clean_maps, clean_activity = compute_clean_baseline(model, data_dict)
    print(f"  Clean Baseline Activity: {clean_activity:.4f}")
    print(f"  Number of heatmap heads: {len(clean_maps)}\n")

    # ------------------------------------------------------------------
    # Build evaluator
    # ------------------------------------------------------------------
    attack = GradFreeAttack(
        detector=model,
        surrogate=surrogate,
        layer_name=LAYER_TO_ATTACK,
        data_dict=data_dict,
        clean_maps=clean_maps
    )

    # ------------------------------------------------------------------
    # Quick sanity check: evaluate at theta = (0, 0)
    # ------------------------------------------------------------------
    J_zero = attack.evaluate(np.array([0.0, 0.0]))
    print(f"  J(0, 0) = {J_zero:.6f}  (should be ~0 since surrogate(0,0) ≈ 0)\n")

    # ------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------
    history = []

    if OPTIMIZER == 'de':
        print("--- STARTING DIFFERENTIAL EVOLUTION ---")
        print(f"  Bounds: {THETA_BOUNDS}")
        print(f"  This is a global optimizer — no gradients needed.\n")

        def callback_de(xk, convergence):
            """Called after each generation — log progress."""
            J_val = -attack.objective_for_scipy(xk)
            history.append(J_val)
            gen = len(history)
            if gen % 5 == 0 or gen == 1:
                print(f"  Gen {gen:>3}: theta=({xk[0]:+.4f}, {xk[1]:+.4f}) | "
                      f"J={J_val:.6f} | evals={attack.eval_count}")

        result = differential_evolution(
            attack.objective_for_scipy,
            bounds=THETA_BOUNDS,
            maxiter=100,          # max generations
            popsize=15,           # population per generation
            tol=1e-6,
            seed=42,
            callback=callback_de,
            disp=False,
        )

        best_theta  = result.x
        best_J      = -result.fun
        total_evals = attack.eval_count

        print(f"\n  Optimization finished in {result.nit} generations, "
              f"{total_evals} total evaluations.")

    elif OPTIMIZER == 'cma':
        import cma

        print("--- STARTING CMA-ES ---")
        print(f"  Bounds: {THETA_BOUNDS}")
        print(f"  CMA-ES is often better for noisy or multimodal landscapes.\n")

        lower_bounds = [b[0] for b in THETA_BOUNDS]
        upper_bounds = [b[1] for b in THETA_BOUNDS]

        es = cma.CMAEvolutionStrategy(
            [0.0, 0.0],         # initial guess
            1.5,                 # initial sigma (step size)
            {
                'bounds': [lower_bounds, upper_bounds],
                'maxiter': 100,
                'seed': 42,
                'verbose': -1,   # suppress cma's own output
            }
        )

        gen = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [attack.objective_for_scipy(s) for s in solutions]
            es.tell(solutions, fitnesses)

            # Log best of this generation
            best_idx = np.argmin(fitnesses)
            J_val = -fitnesses[best_idx]
            history.append(J_val)
            gen += 1

            if gen % 5 == 0 or gen == 1:
                xk = solutions[best_idx]
                print(f"  Gen {gen:>3}: theta=({xk[0]:+.4f}, {xk[1]:+.4f}) | "
                      f"J={J_val:.6f} | evals={attack.eval_count}")

        best_theta  = np.array(es.result.xbest)
        best_J      = -es.result.fbest
        total_evals = attack.eval_count

        print(f"\n  Optimization finished in {gen} generations, "
              f"{total_evals} total evaluations.")

    else:
        raise ValueError(f"Unknown OPTIMIZER: {OPTIMIZER}. Use 'de' or 'cma'.")

    # ------------------------------------------------------------------
    # Results (cleanup AFTER landscape scan so hook is still active)
    # ------------------------------------------------------------------
    if best_J > 0:
        print(f"\n{'=' * 60}")
        print(f"✅  OPTIMAL ATTACK FOUND")
        print(f"    Reflector position : x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
        print(f"    Max objective J    : {best_J:.6f}")
        print(f"    Total evaluations  : {total_evals}")
        print(f"{'=' * 60}")

        # --- Plot optimization history ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: J over generations
        axes[0].plot(range(1, len(history) + 1), history, 'b-o', markersize=3)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("J(θ)")
        axes[0].set_title("Attack Objective over Generations")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: 2D landscape scan around the optimum
        print("\n--- Generating 2D Landscape Scan ---")
        grid_res = 25
        x_range = np.linspace(THETA_BOUNDS[0][0], THETA_BOUNDS[0][1], grid_res)
        y_range = np.linspace(THETA_BOUNDS[1][0], THETA_BOUNDS[1][1], grid_res)
        J_grid  = np.zeros((grid_res, grid_res))

        for ix, x_val in enumerate(x_range):
            for iy, y_val in enumerate(y_range):
                J_grid[iy, ix] = attack.evaluate(np.array([x_val, y_val]))
            if (ix + 1) % 5 == 0:
                print(f"  Landscape scan: {ix + 1}/{grid_res} columns done")

        X, Y = np.meshgrid(x_range, y_range)
        im = axes[1].contourf(X, Y, J_grid, levels=30, cmap='hot')
        axes[1].plot(best_theta[0], best_theta[1], 'c*', markersize=15,
                     markeredgecolor='white', markeredgewidth=1.5, label='Optimum')
        axes[1].set_xlabel("θ_x")
        axes[1].set_ylabel("θ_y")
        axes[1].set_title("J(θ) Landscape")
        axes[1].legend()
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        out_path = OUTPUT_DIR / "attack_optimization_gradfree.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n  Plot saved to: {out_path}")

        # Save results
        results_path = OUTPUT_DIR / "attack_result_gradfree.npz"
        np.savez(
            results_path,
            best_theta=best_theta,
            best_J=best_J,
            history=np.array(history),
            J_grid=J_grid,
            x_range=x_range,
            y_range=y_range,
        )
        print(f"  Results saved to: {results_path}")

    else:
        print("\n❌  Optimization failed — J never exceeded 0.")

    # Cleanup hooks
    attack.cleanup()

    # Final summary at the very end
    if best_J > 0:
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULT")
        print(f"  Reflector position : x={best_theta[0]:.4f}, y={best_theta[1]:.4f}")
        print(f"  Max objective J    : {best_J:.6f}")
        print(f"  Total evaluations  : {total_evals}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()