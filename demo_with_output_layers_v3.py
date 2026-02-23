import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# --- NEW: LAYER EXTRACTION UTILS ---
import os
frame_buffer = {}

def get_layer_hook(name):
    def hook(module, input, output):
        # global frame_buffer
        # Move to CPU immediately to prevent VRAM overflow
        frame_buffer[name] = {
            'features': output.features.detach().cpu().numpy(),
            'indices': output.indices.detach().cpu().numpy()
        }
        # Optional: Print to terminal as it happens
        # print(f"Captured {name}: {len(frame_buffer[name]['indices'])} voxels")
    return hook
# ----------------------------------


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

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    # --- NEW: SETUP SAVE DIRECTORY ---
    save_dir = Path('/isaacsim_projects/paper/working_code_2026/recorded_features')
    save_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------------------

    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    # NOTE: Added weights_only=False here to fix the torch 2.6 loading error you saw earlier
    try:
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    except:
        # Fallback if standard load fails (though we patched the template, this is double safety)
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'])

    model.cuda()
    model.eval()

    # --- REGISTRATION FIX: MATCHING BACKBONE FILE NAMES ---
    logger.info("Registering backbone hooks...")
    # These match the names (conv1, conv2...) defined in spconv_backbone_voxelnext.py
    target_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv_out', 'shared_conv']
    
    logger.info("Registering full backbone hooks (F1-F8)...")

    for name, module in model.backbone_3d.named_modules():
        if name in target_layers:
            # Determine the F-index based on the target list
            idx = target_layers.index(name) + 1
            f_name = f'F{idx}'

            module.register_forward_hook(get_layer_hook(f_name))

            logger.info(f"Successfully hooked: {name} as {f_name}")

    # --- SAFETY CHECK: Print the number of hooks ---
    active_hooks = [name for name, _ in model.backbone_3d.named_modules() if name in target_layers]
    logger.info(f"Total layers ready for recording: {len(active_hooks)}/8")
    # ---------------------------------------

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # --- NEW: SAVE DATA TO DISK ---
            if len(frame_buffer) == 0:
                logger.error(f"!!! Frame {idx}: Buffer is empty. Extraction failed !!!")
            else:
                logger.info(f"Frame {idx}: Captured {list(frame_buffer.keys())}")
                
            frame_id = Path(demo_dataset.sample_file_list[idx]).stem
            save_path = save_dir / f"recorded_{frame_id}.pth"

            # Combine the raw points, the backbone layers, and the detections
            torch.save({
                'raw_points': data_dict['points'][:, 1:].cpu().numpy(),
                'layers': frame_buffer.copy(),
                'detections': {
                    'boxes': pred_dicts[0]['pred_boxes'].cpu().numpy(),
                    'scores': pred_dicts[0]['pred_scores'].cpu().numpy(),
                    'labels': pred_dicts[0]['pred_labels'].cpu().numpy()
                }
            }, save_path)

            logger.info(f"--- Extracted features saved to: {save_path} ---")
            frame_buffer.clear() # Clear for next frame
            # -------------------------------

           # --- DEBUG ALL DETECTIONS (BEFORE FILTERING) ---
            raw_labels = pred_dicts[0]['pred_labels']
            raw_scores = pred_dicts[0]['pred_scores']
            
            logger.info(f"== RAW MODEL OUTPUT (Total: {len(raw_labels)} detections) ==")
            for i in range(len(raw_labels)):
                name = cfg.CLASS_NAMES[raw_labels[i].item() - 1]
                conf = raw_scores[i].item()
                # This prints everything. If it's messy, you can add: if conf > 0.05:
                logger.info(f"Raw Object: {name:15} | Confidence: {conf:.4f}")

           # V.draw_scenes(
           #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
           #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
           # )

            # --- 1. FILTER LOW SCORES (Add this block) ---
            # NuScenes models can be noisy. Filter out anything below 30% confidence.
            score_thresh = 0.28
            mask = pred_dicts[0]['pred_scores'] > score_thresh
            
            pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'][mask]
            pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'][mask]
            pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'][mask]
            
           # --- DEBUG PRINT START ---
 	   # Get the class names from the config
            class_names = cfg.CLASS_NAMES

	   # Extract data for the current frame
            boxes = pred_dicts[0]['pred_boxes']
            scores = pred_dicts[0]['pred_scores']
            labels = pred_dicts[0]['pred_labels']

            logger.info(f"--- Detections for Frame {idx} ---")
            if len(labels) == 0:
                logger.info("No objects detected.")
            else:
                for i in range(len(labels)):
        	  # Convert torch tensor to scalar
                   label_idx = labels[i].item()
                   score = scores[i].item()
        
        # OpenPCDet labels are 1-indexed, so we subtract 1 for the list index
                   class_name = class_names[label_idx - 1]
        
        # Print to terminal
                   logger.info(f"Object {i}: {class_name:15} | Confidence: {score:.2%}")
# --- DEBUG PRINT END ---
                   logger.info(f"Detections after filtering: {len(pred_dicts[0]['pred_labels'])}")
            # ---------------------------------------------

           # --- TRANSLATION FIX START ---
            # 1. Get the labels
            lbl = pred_dicts[0]['pred_labels'].clone()

            # 2. Map nuScenes classes (1-10) to KITTI Visualizer classes (1-3)
            #
            # Visualizer Colors (Standard OpenPCDet):

            # 1 = Green (Car)
            # 2 = Blue (Pedestrian)
            # 3 = Yellow (Cyclist)

            # Map big vehicles (Truck, Construction, Bus, Trailer) to 1 (Green/Car)
            # Assuming standard nuScenes order: 2:truck, 3:construction, 4:bus, 5:trailer
            lbl[lbl == 2] = 1 
            lbl[lbl == 3] = 1
            lbl[lbl == 4] = 1
            lbl[lbl == 5] = 1

            # Map Pedestrians (9) to 2 (Blue)
            lbl[lbl == 9] = 2

            # Map 2-wheelers & Objects (Barrier, Motorcycle, Bicycle, Cone) to 3 (Yellow)
            lbl[lbl == 6] = 3  # Barrier
            lbl[lbl == 7] = 3  # Motorcycle
            lbl[lbl == 8] = 3  # Bicycle
            lbl[lbl == 10] = 3 # Traffic Cone

            # 3. SAFETY CLAMP: Ensure absolutely no label exceeds 3
            # This catches any edge case or different class config
            lbl = torch.clamp(lbl, min=1, max=3)

            # 4. Draw with the "Safe" labels
            #V.draw_scenes(
             #   points=data_dict['points'][:, 1:], 
              #  ref_boxes=pred_dicts[0]['pred_boxes'],
               # ref_scores=pred_dicts[0]['pred_scores'], 
                #ref_labels=lbl  # <--- Pass the translated labels here!
            #)
            # --- TRANSLATION FIX END ---

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
