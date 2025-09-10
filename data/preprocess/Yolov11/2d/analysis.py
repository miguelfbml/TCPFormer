import numpy as np
import matplotlib.pyplot as plt

# Load your YOLO dataset
data = np.load('data_test_3dhp.npz', allow_pickle=True)['data'].item()

print("=== ORIGINAL DATASET ANALYSIS ===")
for seq_name, seq_data in data.items():
    poses_2d = seq_data['data_2d']
    valid_flags = seq_data['valid']
    
    # Count zero poses
    zero_poses = np.sum(np.all(poses_2d == 0, axis=(1, 2)))
    valid_count = np.sum(valid_flags)
    total_frames = len(poses_2d)
    
    print(f"\n{seq_name}:")
    print(f"  Total frames: {total_frames}")
    print(f"  Valid flags: {valid_count}")
    print(f"  Zero poses: {zero_poses}")
    print(f"  Success rate: {(total_frames-zero_poses)/total_frames*100:.1f}%")
    
    # Check coordinate ranges
    non_zero_mask = ~np.all(poses_2d == 0, axis=(1, 2))
    if np.any(non_zero_mask):
        valid_poses = poses_2d[non_zero_mask]
        print(f"  Coord range X: [{np.min(valid_poses[:,:,0]):.1f}, {np.max(valid_poses[:,:,0]):.1f}]")
        print(f"  Coord range Y: [{np.min(valid_poses[:,:,1]):.1f}, {np.max(valid_poses[:,:,1]):.1f}]")
    
    # Check available keys in sequence data
    print(f"  Available keys: {list(seq_data.keys())}")
    
    # Check camera parameters
    if 'camera' in seq_data:
        camera_params = seq_data['camera']
        print(f"  Camera params available: YES")
        print(f"  Camera params type: {type(camera_params)}")
        
        if camera_params is not None:
            if isinstance(camera_params, np.ndarray):
                print(f"  Camera params shape: {camera_params.shape}")
                print(f"  Camera params sample: {camera_params.flatten()[:10] if camera_params.size > 0 else 'Empty'}")
            elif isinstance(camera_params, dict):
                print(f"  Camera params keys: {list(camera_params.keys())}")
                for key, value in camera_params.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape {value.shape}, sample: {value.flatten()[:5]}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  Camera params content: {camera_params}")
        else:
            print(f"  Camera params: None")
    else:
        print(f"  Camera params available: NO")
    
    # Check data_3d structure
    if 'data_3d' in seq_data:
        data_3d = seq_data['data_3d']
        print(f"  3D data shape: {data_3d.shape}")
        print(f"  3D data range: [{np.min(data_3d):.1f}, {np.max(data_3d):.1f}]")
    
    print("-" * 50)

print("\n\n=== CAMERA CALIBRATION FILE ANALYSIS ===")

# Check available calibration files
calib_paths = [
    '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/test_util/camera_calibration/ts1-4cameras.calib',
    '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/test_util/camera_calibration/ts5-6cameras.calib'
]

for calib_path in calib_paths:
    try:
        print(f"\nReading calibration file: {calib_path}")
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        print("First 20 lines:")
        for i, line in enumerate(lines[:20]):
            print(f"  {i+1:2d}: {line.strip()}")
        
        if len(lines) > 20:
            print("  ... (showing first 20 lines only)")
            
    except FileNotFoundError:
        print(f"File not found: {calib_path}")
    except Exception as e:
        print(f"Error reading {calib_path}: {e}")

# Also check for other potential calibration files
import os
calib_dir = '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/test_util/camera_calibration/'
if os.path.exists(calib_dir):
    print(f"\nAll files in calibration directory:")
    for file in os.listdir(calib_dir):
        print(f"  - {file}")