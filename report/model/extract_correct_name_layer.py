import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pathlib import Path
from pcdet.utils import common_utils

# CONFIG
CFG_FILE = '/root/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
DATA_PATH = Path('/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/sample_bg.npy')

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        self.sample_file_list = [root_path]
    def __len__(self): return 1
    def __getitem__(self, index): return {} 

def main():
    logger = common_utils.create_logger()
    cfg_from_yaml_file(CFG_FILE, cfg)
    demo_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False, root_path=DATA_PATH, logger=logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), demo_dataset)
    
    print("\n" + "="*60)
    print(" SEARCHING FOR 'conv4' IN MODEL MODULES ")
    print("="*60)
    
    found = False
    for name, module in model.named_modules():
        # Check if the name ends with conv4 or contains it
        if "conv4" in name:
            print(f"FOUND: '{name}'  -> {type(module).__name__}")
            found = True
            
    print("="*60 + "\n")

if __name__ == '__main__':
    main()