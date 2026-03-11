from isaacsim import SimulationApp
import os

# --- CONFIGURATION ---
NUM_SAMPLES = 100
# Randomization
X_RANGE = (-3.5, 3.5) 
Y_RANGE = (-3.5, 3.5)
Z_HEIGHT = 1.0

OUTPUT_DIR = "/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/dataset_out_xy/"
OUTPUT_DIR_PLY = "/home/francesco/Francesco/isaacsim_projects/paper/working_code_2026/dataset_out_xy/ply/"
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

# ----- OPTIONAL: To see the UI ------
# Enable mouse drawing
simulation_app.set_setting("/app/window/drawMouse", True)

# ----- Enable Required Extensions -----
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.services.livestream.nvcf")

# ----- Now import necessary modules -----
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.core.utils.stage import add_reference_to_stage

# ----- World Setup -----
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane() 

# ----- Add the warehouse to the world using add_reference_to_stage -----
warehouse_usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd"
add_reference_to_stage(warehouse_usd_path, "/World/Warehouse")

# ----- Add person 
person_usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/DH_Characters_Extended/02c80685-06e3-11ef-ae8a-f4b30194174e_6/02c80685-06e3-11ef-ae8a-f4b30194174e_6.usd"
person_prim_path = "/World/Person"
add_reference_to_stage(
    usd_path=person_usd_path,
    prim_path=person_prim_path
)

from isaacsim.core.experimental.prims import XformPrim
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np

person = XformPrim(
    person_prim_path,
    positions=[[-8.0, 5.0, 0.0]],
    orientations=transform_utils.euler_angles_to_quaternion(np.array([0, 0, 0])),
    reset_xform_op_properties=True # <--- THIS IS THE KEY FIX
)
# person.set_local_poses([2.0, 5.0, 0.0])
#------------------- PERSON ADDED ------------------

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
#----------------------- END LIGHTS

# ---------------------- MOCKING CAR 1
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom, Sdf, Gf
import omni.kit.commands
import omni.usd
import numpy as np

# --- SETUP VARIABLES ---
CAR_L, CAR_W, CAR_H = 4.63, 1.95, 1.73
WHEEL_RADIUS = 0.3
CHASSIS_LIFT = 0.3

# Parent Path (The Container)
CAR_PRIM_PATH = "/World/GhostCar"

# 1. CREATE THE PARENT CONTAINER (The "Prim" you asked for)
# -----------------------------------------------------------
# We explicitly define an 'Xform' prim. This acts as the "Folder" you can move.
stage = omni.usd.get_context().get_stage()
container = UsdGeom.Xform.Define(stage, CAR_PRIM_PATH)

# Wrap it in XFormPrim so we can easily move it later
ghost_car_group = XFormPrim(CAR_PRIM_PATH)

# 2. CREATE GEOMETRY (Relative to the Parent)
# -----------------------------------------------------------
# Notice: positions are now LOCAL (0.0), not Global (10.0)

# Body
ghost_body = VisualCuboid(
    prim_path=f"{CAR_PRIM_PATH}/Body",
    name="body",
    # Local Position: Centered in X/Y, Lifted in Z
    position=np.array([0.0, 0.0, (CAR_H / 2.0) + CHASSIS_LIFT]), 
    scale=np.array([CAR_L, CAR_W, CAR_H]),         
    color=np.array([0.2, 0.2, 0.8])             
)

# Wheels Helper
def add_wheel(name, x, y):
    return VisualCylinder(
        prim_path=f"{CAR_PRIM_PATH}/{name}",
        name=name,
        # Local Position: Relative to the car center
        position=np.array([x, y, WHEEL_RADIUS]),
        radius=WHEEL_RADIUS, height=0.2,
        orientation=np.array([0.7071, 0.7071, 0.0, 0.0]) 
    )

wheels = [
    add_wheel("Wheel_FL",  1.45,  0.85),
    add_wheel("Wheel_FR",  1.45, -0.85),
    add_wheel("Wheel_RL", -1.45,  0.85),
    add_wheel("Wheel_RR", -1.45, -0.85)
]

# 3. APPLY MATERIALS (The "Ready-to-use" Fix)
# -----------------------------------------------------------
def apply_omnipbr(prim_path, reflectivity, roughness, color):
    results = []
    # Create standard material
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl", 
        mtl_name="OmniPBR",      
        mtl_created_list=results,
        prim_name=prim_path,
        select_new_prim=False
    )
    # Modify it
    if results:
        mat_prim = stage.GetPrimAtPath(results[0])
        rgb = (color[0], color[1], color[2])
        omni.usd.create_material_input(mat_prim, "metallic_constant", reflectivity, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mat_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mat_prim, "diffuse_color_constant", rgb, Sdf.ValueTypeNames.Color3f)

# Apply Materials
apply_omnipbr(f"{CAR_PRIM_PATH}/Body", 0.8, 0.2, (0.1, 0.1, 0.8)) # Shiny Blue
for w in wheels:
    apply_omnipbr(w.prim_path, 0.1, 0.8, (0.05, 0.05, 0.05)) # Dull Black

print("✅ Car Created Successfully!")

ghost_car_group.set_local_poses(translations=np.array([[-10.0, -12.0, 0.2]]))

# 4. HOW TO MOVE IT
# -----------------------------------------------------------
# Now you can move the ENTIRE car by just moving 'ghost_car_group'

# Move to X=10, Y=0
# ghost_car_group.set_world_pose(position=np.array([10.0, 0.0, 0.0]))
# print("Moved Car to (10, 0, 0)")

# Want to animate it? Just update this loop!
# ghost_car_group.set_world_pose(position=np.array([15.0, 5.0, 0.0]))
# ------------------------------------------------------

#----------------------- SECOND CAR
# --- SETUP VARIABLES ---
CAR_L, CAR_W, CAR_H = 4.63, 1.95, 1.73
WHEEL_RADIUS = 0.3
CHASSIS_LIFT = 0.3

# Parent Path (The Container)
CAR_PRIM_PATH = "/World/GhostCar1"

# 1. CREATE THE PARENT CONTAINER (The "Prim" you asked for)
# -----------------------------------------------------------
# We explicitly define an 'Xform' prim. This acts as the "Folder" you can move.
stage = omni.usd.get_context().get_stage()
container = UsdGeom.Xform.Define(stage, CAR_PRIM_PATH)

# Wrap it in XFormPrim so we can easily move it later
ghost_car_group1 = XFormPrim(CAR_PRIM_PATH)

# 2. CREATE GEOMETRY (Relative to the Parent)
# -----------------------------------------------------------
# Notice: positions are now LOCAL (0.0), not Global (10.0)

# Body
ghost_body = VisualCuboid(
    prim_path=f"{CAR_PRIM_PATH}/Body",
    name="body",
    # Local Position: Centered in X/Y, Lifted in Z
    position=np.array([0.0, 0.0, (CAR_H / 2.0) + CHASSIS_LIFT]), 
    scale=np.array([CAR_L, CAR_W, CAR_H]),         
    color=np.array([0.2, 0.2, 0.8])             
)

# Wheels Helper
def add_wheel(name, x, y):
    return VisualCylinder(
        prim_path=f"{CAR_PRIM_PATH}/{name}",
        name=name,
        # Local Position: Relative to the car center
        position=np.array([x, y, WHEEL_RADIUS]),
        radius=WHEEL_RADIUS, height=0.2,
        orientation=np.array([0.7071, 0.7071, 0.0, 0.0]) 
    )

wheels = [
    add_wheel("Wheel_FL",  1.45,  0.85),
    add_wheel("Wheel_FR",  1.45, -0.85),
    add_wheel("Wheel_RL", -1.45,  0.85),
    add_wheel("Wheel_RR", -1.45, -0.85)
]

# 3. APPLY MATERIALS (The "Ready-to-use" Fix)
# -----------------------------------------------------------
def apply_omnipbr(prim_path, reflectivity, roughness, color):
    results = []
    # Create standard material
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl", 
        mtl_name="OmniPBR",      
        mtl_created_list=results,
        prim_name=prim_path,
        select_new_prim=False
    )
    # Modify it
    if results:
        mat_prim = stage.GetPrimAtPath(results[0])
        rgb = (color[0], color[1], color[2])
        omni.usd.create_material_input(mat_prim, "metallic_constant", reflectivity, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mat_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mat_prim, "diffuse_color_constant", rgb, Sdf.ValueTypeNames.Color3f)

# Apply Materials
apply_omnipbr(f"{CAR_PRIM_PATH}/Body", 0.8, 0.2, (0.1, 0.1, 0.8)) # Shiny Blue
for w in wheels:
    apply_omnipbr(w.prim_path, 0.1, 0.8, (0.05, 0.05, 0.05)) # Dull Black

print("✅ Car Created Successfully!")
ghost_car_group1.set_local_poses(translations=np.array([[-10.0, 10.0, 0.2]]))
#----------------------- END CAR

#------------------------------ DEFINE WALL
from pxr import Sdf, UsdShade

def create_mirror_using_kit_commands(stage, prim_path_to_bind):
    print(f"Creating OmniPBR Mirror for: {prim_path_to_bind}")
    
    mtl_created_list = []

    # 1. Create Material from Internal Library (Prevents C120/URL errors)
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="MirrorMaterial",
        mtl_created_list=mtl_created_list,
        select_new_prim=False
    )
    
    # 2. Configure Material Inputs for Perfect Reflection
    mtl_prim_path = mtl_created_list[0]
    mtl_prim = stage.GetPrimAtPath(mtl_prim_path)

    # Metallic = 1.0 (Metal)
    omni.usd.create_material_input(
        mtl_prim, "metallic_constant", 1.0, Sdf.ValueTypeNames.Float
    )
    
    # Roughness = 0.0 (Perfectly smooth)
    omni.usd.create_material_input(
        mtl_prim, "reflection_roughness_constant", 0.1, Sdf.ValueTypeNames.Float
    )
    
    # Color = White (High Albedo for reflection)
    omni.usd.create_material_input(
        mtl_prim, "diffuse_color_constant", (1.0, 1.0, 1.0), Sdf.ValueTypeNames.Color3f
    )

    # 3. Bind to the Wall
    cube_prim = stage.GetPrimAtPath(prim_path_to_bind)
    cube_mat_shade = UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(cube_prim).Bind(
        cube_mat_shade, 
        UsdShade.Tokens.strongerThanDescendants
    )

# B. MIRROR WALL (Grey Surface)
mirror = VisualCuboid(
    prim_path="/World/MirrorWall",
    name="mirror_wall",
    position=np.array([5.0, 7.0, 1.0]), 
    # scale=np.array([0.1, 4.0, 5.0]), 
    scale=np.array([0.1, 1.0, 3.0])
)

# Apply the Mirror Material
create_mirror_using_kit_commands(world.stage, mirror.prim_path)

#--------------------------------------------------------------------
#|                                                                  |
#|                          ADDING LIDAR                            |
#|                                                                  |
#--------------------------------------------------------------------
config = {"omni:sensor:Core:maxReturns": 2}
ouster = LidarRtx(
    prim_path="/World/OusterLidar",
    name="OusterLidar",
    # config_file_name="Ouster_VLS_128",  # Using VLS-128 configuration
    config_file_name="Ouster_VLS_128",
    # translation=(1, 5, 0.5)
    position=(2, 5, 1.73),
    **config
)
ouster.prim.GetAttribute("omni:sensor:Core:maxReturns").Set(3)
ouster.prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Set(10)
ouster.prim.GetAttribute("omni:sensor:tickRate").Set(60)
ouster.set_local_pose(translation=np.array([1.0, 7.0, 1.73]))

# ----- Initialize Lidar and Attach Annotator -----
ouster.initialize()
# Get the render product path created by LidarRtx
render_product_path = ouster.get_render_product_path()
import omni.replicator.core as rep
annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
# We want XYZ + intensity (you can enable more flags later if needed)
annotator.initialize(
    outputDistance=False,
    outputAzimuth=False,
    outputElevation=False,
    outputIntensity=True,
    outputTimestamp=False,
    outputEmitterId=False,
    outputMaterialId=False,   # set True if you want to debug materials
    outputObjectId=False,
    enablePerFrameOutput=False
    # outputCompleteFullScanOnly=True
    # keepOnlyLastFullScan=True
)

# Attach annotator to lidar render product
annotator.attach([render_product_path])

import numpy as np
import os

def save_to_ply(points, intensity, filename="lidar_output.ply"):
    """
    Saves a numpy array of (N, 3) points to a standard PLY file.
    """
    
    # --- NEW: Create the directory if it doesn't exist ---
    # 1. Get the folder path from the filename
    folder_path = os.path.dirname(filename)
    
    # 2. If a folder path was provided and it doesn't exist, create it
    if folder_path and not os.path.exists(folder_path):
        print(f"Creating directory: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
    # -----------------------------------------------------

    # Filter out invalid points (points at 0,0,0 usually indicate no return)
    # norms = np.linalg.norm(points, axis=1)
    # valid_mask = norms > 0.001

    # valid_points = points[valid_mask]
    # valid_intensity = intensity[valid_mask]

    # --- FIX 1: Intensity Patch ---
    # If Isaac Sim returns all 0s (common), we inject noise to prevent OpenPCDet Crash
    if np.max(intensity) == 0:
        print(f"⚠️ Warning: Zero Intensity detected in {filename}. Patching with random noise...")
        # Random float between 0.1 and 0.9
        # intensity = np.random.uniform(0.1, 0.9, size=intensity.shape).astype(np.float32)
        intensity = np.full(intensity.shape, 0.5, dtype=np.float32)

    n = len(points)
    print(f"Saving {n} valid points to {filename}...")

    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
property float intensity
end_header
"""
    
    with open(filename, "w") as f:
        f.write(header)
        for i in range(n):
            p = points[i]
            # inten = intensity[i]

            # --- ROBUST FIX ---
            # Cast to python float. This handles:
            # - Shape (N, 1) -> intensity[i] is [val] -> float([val]) = val
            # - Shape (N,)   -> intensity[i] is val   -> float(val)   = val
            inten = float(intensity[i])
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {inten:.4f}\n")
    
    print("Save complete.")

# --- HELPER FUNCTIONS ---
import open3d as o3d

# ----- HELPER FUNCTIONS -----

def get_look_at_quaternion(object_pos, target_pos):
    """Calculates rotation so the mirror face always points to the Lidar."""
    direction = target_pos - object_pos
    angle_z = np.arctan2(direction[1], direction[0])
    cy = np.cos(angle_z * 0.5)
    sy = np.sin(angle_z * 0.5)
    return np.array([cy, 0, 0, sy])


# def scan_one_rotation(timeout_frames=60):
#     """
#     Manually enforces the 'Full Scan' by waiting exactly 6 frames.
#     This replaces the need for the crashing parameter.
#     """
#     for _ in range(timeout_frames):
#         world.step(render=True)
        
#     data = annotator.get_data()
    
#     if data is not None and "data" in data:
#         raw = data["data"]
#         if raw.dtype.names:
#             points = np.stack([raw['x'], raw['y'], raw['z']], axis=1)
#         else:
#             points = raw
#         return points
#     return np.zeros((0, 3))

def compute_difference(bg_points, current_points, current_intensity, threshold=0.05):
    if len(bg_points) == 0: return current_points, current_intensity
    if len(current_points) == 0: return np.zeros((0, 3)), np.zeros((0,1))

    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(bg_points.astype(np.float64))
    
    pcd_curr = o3d.geometry.PointCloud()
    pcd_curr.points = o3d.utility.Vector3dVector(current_points.astype(np.float64))

    dists = pcd_curr.compute_point_cloud_distance(pcd_bg)
    dists = np.asarray(dists)

    indices = np.where(dists > threshold)[0]
    return current_points[indices], current_intensity[indices]

def flush_buffer():
    """Spins the Lidar for one rotation to clear old data."""
    for _ in range(8): # Wait 10 frames (approx 1.5 rotations) to be safe
        world.step(render=True)
    # We do NOT read data here; we just let it overwrite the internal buffer.


def scan_until_valid(timeout_frames=15):
    """
    Returns:
        points: (N, 3) numpy array
        intensity: (N, 1) numpy array
    """
    for i in range(timeout_frames+2):
        world.step(render=True)
        data = annotator.get_data()
        
        if data is not None and "data" in data:
            raw = data["data"]
            if raw.size > 1000:
                if raw.dtype.names:
                    # 1. Extract XYZ
                    points = np.stack([raw['x'], raw['y'], raw['z']], axis=1)
                    # 2. Extract Intensity
                    intensity = raw['intensity'].reshape(-1, 1)
                    return points, intensity
                else:
                    return raw, np.zeros((len(raw), 1))
    
    print("    ⚠️ Warning: Lidar Scan Timeout")
    return np.zeros((0, 3)), np.zeros((0, 1))


# --- MAIN LOOP ---
import pickle
import random

world.reset()

print("--- Step 1: Capturing Clean Background ---")
mirror.set_world_pose(position=np.array([0, 0, -100.0]))
# flush_buffer() # Warmup

# Warmup spin
# scan_one_rotation() 

# Loop until we get a valid background (Critical)
bg_points = np.zeros((0, 3))
while len(bg_points) == 0:
    print("    Waiting for valid background scan...")
    bg_points, _ = scan_until_valid(timeout_frames=100)
ply_path_background = os.path.join(OUTPUT_DIR_PLY, f"sample_bg.ply")
save_to_ply(bg_points, _, ply_path_background)

# bg_points = scan_one_rotation()
print(f"Background captured with {len(bg_points)} points.")
print(f"--- Step 2: Starting Collection of {NUM_SAMPLES} Samples ---")
lidar_pos = np.array([2.0, 5.0, 1.73])

for i in range(NUM_SAMPLES):
    
    rand_x = random.uniform(X_RANGE[0], X_RANGE[1])
    rand_y = random.uniform(Y_RANGE[0], Y_RANGE[1])
    pos = np.array([rand_x, rand_y, Z_HEIGHT])
    
    #CHECK IF ONE WALL IN FRONT OF THE LIDAR RETURNS FULL SCAN
    # pos =  np.array([5.0, 7.0, 1.0])

    print(f"Position: {pos}")
    
    
    # 1. Rotate & Move
    rot_quat = get_look_at_quaternion(pos, lidar_pos)
    mirror.set_world_pose(position=pos, orientation=rot_quat)
    
    
    # 2. Flush / Warmup
    # Step a few times to let physics update and clear old ghost data
    # for _ in range(10):
        # world.reset()
    world.step(render=True)
    flush_buffer()


    # Allow 1 frame for physics update
    # world.step(render=True)
    
    # Scan for exactly 0.1 seconds (6 frames)
    # This guarantees a 360 scan without the crash-prone flag
    # current_points = scan_one_rotation()
    current_points, current_intensity = scan_until_valid()

    # If capture failed despite retries, skip
    if len(current_points) == 0:
        continue
    
    diff_points, diff_intensity = compute_difference(bg_points, current_points, current_intensity)
    
    # We save the tuple as requested
    if len(diff_points) > 50:
        sample_data = {
            "id": i,
            "object_position": pos,
            "points": diff_points,      # The filtered object cloud
            "full_scan": current_points, # (Optional) The raw scene if you need it later
            "intensity": current_intensity
        }
        
        filename = os.path.join(OUTPUT_DIR, f"sample_{i:04d}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(sample_data, f)

            # --- Save as PLY (For Visualization) ---
        ply_path = os.path.join(OUTPUT_DIR_PLY, f"sample_{i:04d}.ply")
        save_to_ply(current_points, current_intensity, ply_path)

        if i % 6 == 0:
            print(f"Sample {i}/{NUM_SAMPLES} | Pos: ({rand_x:.2f}, {rand_y:.2f}) | Points: {len(diff_points)} | Bakcground: {len(current_points)}")
    else:
        if i % 10 == 0:
             print(f"Sample {i}/{NUM_SAMPLES} | ⚠️ Mirror Missed (0 diff points)")

print("Dataset collection complete.")

simulation_app.close()