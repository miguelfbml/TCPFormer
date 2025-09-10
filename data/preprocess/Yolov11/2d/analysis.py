import numpy as np
import matplotlib.pyplot as plt

# Load your YOLO dataset
data = np.load('data_test_3dhp.npz', allow_pickle=True)['data'].item()

for seq_name, seq_data in data.items():
    poses_2d = seq_data['data_2d']
    valid_flags = seq_data['valid']
    
    # Count zero poses
    zero_poses = np.sum(np.all(poses_2d == 0, axis=(1, 2)))
    valid_count = np.sum(valid_flags)
    total_frames = len(poses_2d)
    
    print(f"{seq_name}:")
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
    print()