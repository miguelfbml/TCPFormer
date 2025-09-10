import numpy as np
import os

print("🔍 ANALYZING REAL GROUND TRUTH FROM data_test_3dhp.npz")
print("=" * 60)

# Check if the file exists in current directory
gt_file = 'data_test_3dhp.npz'
if not os.path.exists(gt_file):
    print(f"❌ {gt_file} not found in current directory")
    print("Files in current directory:")
    for f in os.listdir('.'):
        if f.endswith('.npz'):
            print(f"  - {f}")
    exit()

print(f"📁 Loading: {gt_file}")
try:
    gt_data = np.load(gt_file, allow_pickle=True)['data'].item()
    print(f"✓ Loaded {len(gt_data)} sequences: {list(gt_data.keys())}")
    
    print("\n📊 REAL GROUND TRUTH COORDINATE ANALYSIS:")
    
    for seq_name, seq_data in gt_data.items():
        poses_2d = seq_data['data_2d']
        valid_flags = seq_data['valid']
        
        print(f"\n{seq_name}:")
        print(f"  Shape: {poses_2d.shape}")
        print(f"  Valid flags type: {valid_flags.dtype}")
        print(f"  Valid frames: {np.sum(valid_flags.astype(bool))}/{len(valid_flags)}")
        
        # Get valid poses only
        valid_bool = valid_flags.astype(bool)
        if np.any(valid_bool):
            valid_poses = poses_2d[valid_bool]
            
            # Get all coordinates
            all_x = valid_poses[:, :, 0].flatten()
            all_y = valid_poses[:, :, 1].flatten()
            
            # Remove zeros (failed detections)
            non_zero_mask = (all_x != 0) | (all_y != 0)
            if np.any(non_zero_mask):
                x_coords = all_x[non_zero_mask]
                y_coords = all_y[non_zero_mask]
                
                print(f"  Coordinate Statistics:")
                print(f"    X range: [{np.min(x_coords):.1f}, {np.max(x_coords):.1f}]")
                print(f"    Y range: [{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
                print(f"    X mean±std: {np.mean(x_coords):.1f}±{np.std(x_coords):.1f}")
                print(f"    Y mean±std: {np.mean(y_coords):.1f}±{np.std(y_coords):.1f}")
                
                # Check for negative coordinates
                neg_x = np.sum(x_coords < 0)
                neg_y = np.sum(y_coords < 0)
                print(f"    Negative coords: X={neg_x}/{len(x_coords)} ({neg_x/len(x_coords)*100:.1f}%)")
                print(f"                     Y={neg_y}/{len(y_coords)} ({neg_y/len(y_coords)*100:.1f}%)")
                
                # Check coordinates beyond typical image boundaries
                if seq_name in ['TS1', 'TS2', 'TS3', 'TS4']:
                    img_w, img_h = 2048, 2048
                else:
                    img_w, img_h = 1920, 1080
                
                beyond_x = np.sum(x_coords > img_w)
                beyond_y = np.sum(y_coords > img_h)
                print(f"    Beyond image bounds: X>{img_w}: {beyond_x}/{len(x_coords)} ({beyond_x/len(x_coords)*100:.1f}%)")
                print(f"                         Y>{img_h}: {beyond_y}/{len(y_coords)} ({beyond_y/len(y_coords)*100:.1f}%)")
                
                # Sample some actual coordinate values
                print(f"    Sample coordinates (first 5 valid poses, first 3 joints):")
                for i in range(min(5, len(valid_poses))):
                    for j in range(3):
                        x, y = valid_poses[i, j, 0], valid_poses[i, j, 1]
                        print(f"      Frame {i}, Joint {j}: ({x:.1f}, {y:.1f})")
            else:
                print(f"  ❌ No valid coordinates found")
        else:
            print(f"  ❌ No valid frames")
    
    print("\n🤔 COORDINATE SYSTEM ANALYSIS:")
    print("Based on the coordinate ranges:")
    
    # Determine coordinate system
    all_sequences_pixel_like = True
    all_sequences_camera_like = True
    
    for seq_name, seq_data in gt_data.items():
        poses_2d = seq_data['data_2d']
        valid_flags = seq_data['valid'].astype(bool)
        
        if np.any(valid_flags):
            valid_poses = poses_2d[valid_flags]
            all_coords = valid_poses.reshape(-1, 2)
            non_zero_coords = all_coords[(all_coords != 0).any(axis=1)]
            
            if len(non_zero_coords) > 0:
                x_min, x_max = np.min(non_zero_coords[:, 0]), np.max(non_zero_coords[:, 0])
                y_min, y_max = np.min(non_zero_coords[:, 1]), np.max(non_zero_coords[:, 1])
                
                # Check if it looks like pixel coordinates
                if seq_name in ['TS1', 'TS2', 'TS3', 'TS4']:
                    expected_max = 2048
                else:
                    expected_max = max(1920, 1080)
                
                pixel_like = (x_min >= 0 and y_min >= 0 and 
                             x_max <= expected_max * 1.1 and y_max <= expected_max * 1.1)
                
                # Check if it looks like camera coordinates (can have negatives, larger ranges)
                camera_like = (x_min < -100 or y_min < -100 or 
                              x_max > expected_max * 1.2 or y_max > expected_max * 1.2)
                
                print(f"  {seq_name}: Range ({x_min:.1f}-{x_max:.1f}, {y_min:.1f}-{y_max:.1f})")
                print(f"    Pixel-like: {pixel_like}, Camera-like: {camera_like}")
                
                if not pixel_like:
                    all_sequences_pixel_like = False
                if not camera_like:
                    all_sequences_camera_like = False
    
    print(f"\n🎯 CONCLUSION:")
    if all_sequences_pixel_like:
        print("✅ Ground truth appears to be in PIXEL COORDINATES")
        print("   → Your YOLO should output pixel coordinates (no camera conversion needed)")
    elif all_sequences_camera_like:
        print("✅ Ground truth appears to be in CAMERA COORDINATES") 
        print("   → Your YOLO needs camera coordinate conversion")
    else:
        print("⚠️  Mixed coordinate system or unclear - need manual inspection")
        print("   → Check the sample coordinates above to determine the system")

except Exception as e:
    print(f"❌ Error loading ground truth: {e}")
    import traceback
    traceback.print_exc()