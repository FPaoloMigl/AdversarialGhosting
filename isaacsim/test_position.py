from isaacsim import SimulationApp
import os
import math
import pickle
import random
import numpy as np

# --- CONFIGURATION ---
# Randomization
# X_RANGE = (-30.0, 30) # We start for rivermark at -30, 30, for x is -60, 0
THETA_STAR = (-31.59, 6.87, 303.2)

# OUTPUT_DIR = "/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/dataset_out_xy/v2/demo"
OUTPUT_DIR = "AdversarialGhosting/QualityAssessment/rivermark/optimization"
# OUTPUT_DIR_PLY = "/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/dataset_out_xy/v2/demo/ply/"
OUTPUT_DIR_PLY = "AdversarialGhosting/QualityAssessment/rivermark/optimization"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ----- Launch Simulator -----
simulation_app = SimulationApp (
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "enable_motion_bvh": True,
        "width": 1280,
        "height": 720,
        "window_width": 1920,
        "window_height": 1080,
        "display_options": 3286,
        "hide_ui": False,

        "extra_args": [
        "--/app/sensors/nv/lidar/enableRtxMaterial=true", 
        "--/app/sensors/nv/lidar/enableRtxReflectanceInformation=true",
        "--/app/hydraEngine/waitIdle=1",
        "--/app/updateOrder/checkForHydraRenderComplete=1000",
        "--/exts/isaacsimsim.ros2.bridge/publish_multithreading_disabled=1",
    ]
    }
)

# Enable mouse drawing
simulation_app.set_setting("/app/window/drawMouse", True)

# ----- Enable Required Extensions -----
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.services.livestream.nvcf")

# ----- Now import necessary modules -----
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.prims import XFormPrim
import isaacsim.core.experimental.utils.transform as transform_utils
from pxr import UsdGeom, Sdf, UsdShade
import omni.kit.commands
import omni.usd
import open3d as o3d
import omni.replicator.core as rep

# ----- World Setup -----
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane() 

# ----- Add the warehouse -----
# warehouse_usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd"
warehouse_usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Outdoor/Rivermark/rivermark.usd"
add_reference_to_stage(warehouse_usd_path, "/World/Warehouse")

# Wait for the scene to fully load
import omni.usd
while omni.usd.get_context().get_stage_loading_status()[2] > 0:
    simulation_app.update()
print("Environment fully loaded.")

# ----- Add person -----
person_usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/DH_Characters_Extended/02c80685-06e3-11ef-ae8a-f4b30194174e_6/02c80685-06e3-11ef-ae8a-f4b30194174e_6.usd"
person_prim_path = "/World/Person"
add_reference_to_stage(usd_path=person_usd_path, prim_path=person_prim_path)

person = XformPrim(
    person_prim_path,
    positions=[[-8.0, 5.0, 0.0]],
    orientations=transform_utils.euler_angles_to_quaternion(np.array([0, 0, 0])),
    reset_xform_op_properties=True
)

#------------------- LIGHTS ------------------------
from isaacsim.core.experimental.objects import CylinderLight

cylinder_light = CylinderLight(
    paths="/World/CylinderLight",
    translations=[14.0, 0.0, 30.0],
    orientations=[0, 0.7071068, 0.7071068, 0],
    scales=[1.0, 1.0, 1.0],
    lengths=100.0,
    radii=5.0,
    reset_xform_op_properties=True)
cylinder_light.set_intensities(intensities=[30000])
cylinder_light.set_color_temperatures([6500])

# ---------------------- MOCKING CAR 1 ----------------------
CAR_L, CAR_W, CAR_H = 4.63, 1.95, 1.73
WHEEL_RADIUS = 0.3
CHASSIS_LIFT = 0.3

CAR_PRIM_PATH = "/World/GhostCar"
stage = omni.usd.get_context().get_stage()
container = UsdGeom.Xform.Define(stage, CAR_PRIM_PATH)
ghost_car_group = XFormPrim(CAR_PRIM_PATH)

ghost_body = VisualCuboid(
    prim_path=f"{CAR_PRIM_PATH}/Body",
    name="body",
    position=np.array([0.0, 0.0, (CAR_H / 2.0) + CHASSIS_LIFT]), 
    scale=np.array([CAR_L, CAR_W, CAR_H]),         
    color=np.array([0.2, 0.2, 0.8])             
)

def add_wheel(name, x, y, parent_path):
    return VisualCylinder(
        prim_path=f"{parent_path}/{name}",
        name=name,
        position=np.array([x, y, WHEEL_RADIUS]),
        radius=WHEEL_RADIUS, height=0.2,
        orientation=np.array([0.7071, 0.7071, 0.0, 0.0]) 
    )

wheels = [
    add_wheel("Wheel_FL",  1.45,  0.85, CAR_PRIM_PATH),
    add_wheel("Wheel_FR",  1.45, -0.85, CAR_PRIM_PATH),
    add_wheel("Wheel_RL", -1.45,  0.85, CAR_PRIM_PATH),
    add_wheel("Wheel_RR", -1.45, -0.85, CAR_PRIM_PATH)
]

def apply_omnipbr(prim_path, reflectivity, roughness, color):
    results = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl", mtl_name="OmniPBR", mtl_created_list=results,
        prim_name=prim_path, select_new_prim=False
    )
    if results:
        mat_prim = stage.GetPrimAtPath(results[0])
        rgb = (color[0], color[1], color[2])
        omni.usd.create_material_input(mat_prim, "metallic_constant", reflectivity, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mat_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mat_prim, "diffuse_color_constant", rgb, Sdf.ValueTypeNames.Color3f)

apply_omnipbr(f"{CAR_PRIM_PATH}/Body", 0.8, 0.2, (0.1, 0.1, 0.8)) 
for w in wheels: apply_omnipbr(w.prim_path, 0.1, 0.8, (0.05, 0.05, 0.05)) 

print("✅ Car Created Successfully!")

# ghost_car_group.set_local_poses(translations=np.array([[-10.0, -12.0, 0.2]]))
ghost_car_group.set_local_poses(translations=np.array([[-57.0, -1.0, 6.2]]))

#----------------------- MOCKING CAR 2 ----------------------
CAR_PRIM_PATH2 = "/World/GhostCar1"
container2 = UsdGeom.Xform.Define(stage, CAR_PRIM_PATH2)
ghost_car_group1 = XFormPrim(CAR_PRIM_PATH2)

ghost_body2 = VisualCuboid(
    prim_path=f"{CAR_PRIM_PATH2}/Body",
    name="body",
    position=np.array([0.0, 0.0, (CAR_H / 2.0) + CHASSIS_LIFT]), 
    scale=np.array([CAR_L, CAR_W, CAR_H]),         
    color=np.array([0.2, 0.2, 0.8])             
)

wheels2 = [
    add_wheel("Wheel_FL",  1.45,  0.85, CAR_PRIM_PATH2),
    add_wheel("Wheel_FR",  1.45, -0.85, CAR_PRIM_PATH2),
    add_wheel("Wheel_RL", -1.45,  0.85, CAR_PRIM_PATH2),
    add_wheel("Wheel_RR", -1.45, -0.85, CAR_PRIM_PATH2)
]

apply_omnipbr(f"{CAR_PRIM_PATH2}/Body", 0.8, 0.2, (0.1, 0.1, 0.8)) 
for w in wheels2: apply_omnipbr(w.prim_path, 0.1, 0.8, (0.05, 0.05, 0.05)) 

print("✅ Car Created Successfully!")

# ghost_car_group1.set_local_poses(translations=np.array([[-10.0, 10.0, 0.2]]))
ghost_car_group1.set_local_poses(translations=np.array([[-27.0, 15.5, 6.2]]))

#------------------------------ DEFINE WALL ----------------------
def create_mirror_using_kit_commands(stage, prim_path_to_bind):
    print(f"Creating OmniPBR Mirror for: {prim_path_to_bind}")
    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl", mtl_name="MirrorMaterial",
        mtl_created_list=mtl_created_list, select_new_prim=False
    )
    mtl_prim_path = mtl_created_list[0]
    mtl_prim = stage.GetPrimAtPath(mtl_prim_path)
    omni.usd.create_material_input(mtl_prim, "metallic_constant", 1.0, Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(mtl_prim, "reflection_roughness_constant", 0.1, Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(mtl_prim, "diffuse_color_constant", (1.0, 1.0, 1.0), Sdf.ValueTypeNames.Color3f)
    
    cube_prim = stage.GetPrimAtPath(prim_path_to_bind)
    cube_mat_shade = UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(cube_prim).Bind(cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)

WALL_SCALE = np.array([0.1, 5.0, 6.0])
WALL_HEIGHT = WALL_SCALE[2]

mirror = VisualCuboid(
    prim_path="/World/MirrorWall",
    name="mirror_wall",
    # Set Z to exactly half of the height to keep it completely above ground
    position=np.array([5.0, 7.0, WALL_HEIGHT / 2.0]), 
    scale=WALL_SCALE
)
create_mirror_using_kit_commands(world.stage, mirror.prim_path)

#-------------------------- ADDING LIDAR --------------------------
config = {"omni:sensor:Core:maxReturns": 2}
ouster = LidarRtx(
    prim_path="/World/OusterLidar",
    name="OusterLidar",
    config_file_name="Ouster_VLS_128",
    position=(2, 5, 1.73),
    **config
)
ouster.prim.GetAttribute("omni:sensor:Core:maxReturns").Set(3)
ouster.prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Set(10)
ouster.prim.GetAttribute("omni:sensor:tickRate").Set(60)
# ouster.set_local_pose(translation=np.array([1.0, 7.0, 1.73]))
ouster.set_local_pose(translation=np.array([-44.0, 9.5, 7.73]))

ouster.initialize()
render_product_path = ouster.get_render_product_path()
annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
annotator.initialize(
    outputDistance=False, outputAzimuth=False, outputElevation=False,
    outputIntensity=True, outputTimestamp=False, outputEmitterId=False,
    outputMaterialId=False, outputObjectId=False, enablePerFrameOutput=False
)
annotator.attach([render_product_path])

# --- HELPER FUNCTIONS ---

def get_look_at_quaternion(object_pos, target_pos):
    """Calculates rotation so the mirror face always points to the Lidar."""
    direction = target_pos - object_pos
    angle_z = np.arctan2(direction[1], direction[0])
    cy = np.cos(angle_z * 0.5)
    sy = np.sin(angle_z * 0.5)
    return np.array([cy, 0, 0, sy])


def save_to_ply(points, intensity, filename="lidar_output.ply"):
    folder_path = os.path.dirname(filename)
    if folder_path and not os.path.exists(folder_path):
        print(f"Creating directory: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
    if np.max(intensity) == 0:
        # print(f"⚠️ Warning: Zero Intensity detected in {filename}. Patching with random noise...")
        intensity = np.full(intensity.shape, 0.5, dtype=np.float32)

    n = len(points)
    # print(f"Saving {n} valid points to {filename}...")
    header = f"ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n"
    with open(filename, "w") as f:
        f.write(header)
        for i in range(n):
            p = points[i]
            inten = float(intensity[i])
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {inten:.4f}\n")

    print("Save complete")

def flush_buffer():
    for _ in range(8):
        world.step(render=True)

def scan_until_valid(timeout_frames=15):
    for i in range(timeout_frames+2):
        world.step(render=True)
        data = annotator.get_data()
        if data is not None and "data" in data:
            raw = data["data"]
            if raw.size > 1000:
                if raw.dtype.names:
                    points = np.stack([raw['x'], raw['y'], raw['z']], axis=1)
                    intensity = raw['intensity'].reshape(-1, 1)
                    return points, intensity
                else:
                    return raw, np.zeros((len(raw), 1))
    
    print("    ⚠️ Warning: Lidar Scan Timeout")
    return np.zeros((0, 3)), np.zeros((0, 1))

# --- MAIN LOOP ---
world.reset()

pos = np.array([THETA_STAR[0],THETA_STAR[1], (WALL_HEIGHT / 2.0) + 5.0])
angle = math.radians(THETA_STAR[2])

 # Apply purely Z-axis rotation
rot_quat = transform_utils.euler_angles_to_quaternion(np.array([angle, 0.0, 0.0]))

if hasattr(rot_quat, "numpy"):
        rot_quat = rot_quat.numpy()
else:
    rot_quat = np.array(rot_quat)

mirror.set_world_pose(position=pos, orientation=rot_quat)

# 3. Flush & Scan
world.step(render=True)
flush_buffer()

points, _ = scan_until_valid(timeout_frames=100)
ply_points = os.path.join(OUTPUT_DIR_PLY, "sample_test.ply")
save_to_ply(points, _, ply_points)

print(f"Test Captured.")

simulation_app.close()
