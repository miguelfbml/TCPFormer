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