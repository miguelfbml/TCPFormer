import os
import sys
import numpy as np

# Add TCPFormer root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tcpformer_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, tcpformer_root)

print("🧪 TESTING DATA COMPATIBILITY WITH TCPFORMER")
print("=" * 60)

# Load original ground truth data
print("1. Loading original ground truth data...")
try:
    original_data = np.load('data_test_3dhp.npz', allow_pickle=True)['data'].item()
    print(f"   ✓ Loaded {len(original_data)} sequences")
    
    # Analyze ground truth coordinate ranges
    print("\n📊 Ground Truth Coordinate Analysis:")
    for seq_name in list(original_data.keys())[:2]:  # Check first 2 sequences
        seq_data = original_data[seq_name]
        poses_2d = seq_data['data_2d']
        valid_flags = seq_data['valid']
        
        print(f"\n  {seq_name}:")
        print(f"    Shape: {poses_2d.shape}")
        print(f"    Valid frames: {np.sum(valid_flags)}/{len(valid_flags)}")
        
        # Get coordinate statistics
        valid_poses = poses_2d[valid_flags]
        if len(valid_poses) > 0:
            x_coords = valid_poses[:, :, 0].flatten()
            y_coords = valid_poses[:, :, 1].flatten()
            
            print(f"    X range: [{np.min(x_coords):.1f}, {np.max(x_coords):.1f}]")
            print(f"    Y range: [{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
            print(f"    X mean±std: {np.mean(x_coords):.1f}±{np.std(x_coords):.1f}")
            print(f"    Y mean±std: {np.mean(y_coords):.1f}±{np.std(y_coords):.1f}")
            
            # Check for negative values
            neg_x = np.sum(x_coords < 0)
            neg_y = np.sum(y_coords < 0)
            print(f"    Negative coords: X={neg_x}/{len(x_coords)}, Y={neg_y}/{len(y_coords)}")

except Exception as e:
    print(f"   ❌ Error loading ground truth: {e}")

# Check if your YOLO dataset exists and compare
print("\n2. Checking YOLO dataset compatibility...")
yolo_files = [
    'custom_yolo_dataset.npz',
    'custom_yolo_dataset_camera_coords.npz',
    'data_test_3dhp_yolo.npz'
]

for yolo_file in yolo_files:
    if os.path.exists(yolo_file):
        print(f"\n📁 Found YOLO dataset: {yolo_file}")
        try:
            yolo_data = np.load(yolo_file, allow_pickle=True)['data'].item()
            
            # Compare structure with ground truth
            print(f"    Sequences: {list(yolo_data.keys())}")
            
            for seq_name in list(yolo_data.keys())[:1]:  # Check first sequence
                yolo_seq = yolo_data[seq_name]
                orig_seq = original_data[seq_name]
                
                print(f"\n    {seq_name} comparison:")
                print(f"      Original 2D shape: {orig_seq['data_2d'].shape}")
                print(f"      YOLO 2D shape: {yolo_seq['data_2d'].shape}")
                print(f"      Shapes match: {orig_seq['data_2d'].shape == yolo_seq['data_2d'].shape}")
                
                # Compare coordinate ranges
                yolo_poses = yolo_seq['data_2d']
                yolo_valid = yolo_seq['valid']
                
                valid_yolo_poses = yolo_poses[yolo_valid]
                if len(valid_yolo_poses) > 0:
                    x_coords = valid_yolo_poses[:, :, 0].flatten()
                    y_coords = valid_yolo_poses[:, :, 1].flatten()
                    
                    print(f"      YOLO X range: [{np.min(x_coords):.1f}, {np.max(x_coords):.1f}]")
                    print(f"      YOLO Y range: [{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
                    
                    # Compare with ground truth
                    orig_poses = orig_seq['data_2d'][orig_seq['valid']]
                    if len(orig_poses) > 0:
                        orig_x = orig_poses[:, :, 0].flatten()
                        orig_y = orig_poses[:, :, 1].flatten()
                        
                        x_ratio = (np.max(x_coords) - np.min(x_coords)) / (np.max(orig_x) - np.min(orig_x))
                        y_ratio = (np.max(y_coords) - np.min(y_coords)) / (np.max(orig_y) - np.min(orig_y))
                        
                        print(f"      Range ratio (YOLO/GT): X={x_ratio:.3f}, Y={y_ratio:.3f}")
                        print(f"      Coordinate system match: {abs(x_ratio - 1.0) < 0.1 and abs(y_ratio - 1.0) < 0.1}")
                
        except Exception as e:
            print(f"    ❌ Error loading YOLO dataset: {e}")

# Try to test with TCPFormer data loader if possible
print("\n3. Testing with TCPFormer data loader...")
try:
    # Try to find and import the data loader
    possible_paths = [
        os.path.join(tcpformer_root, 'data'),
        os.path.join(tcpformer_root, 'lib', 'data'),
        os.path.join(tcpformer_root, 'common')
    ]
    
    data_loader_found = False
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                # Try common data loader names
                import data_loader_3dhp
                print(f"   ✓ Found data loader in {path}")
                
                # Try to inspect what it does with 2D coordinates
                if hasattr(data_loader_3dhp, 'MPI3DHP'):
                    print("   📋 Inspecting MPI3DHP class...")
                    # This would require actually running the data loader
                    # which might need more setup
                    
                data_loader_found = True
                break
            except ImportError:
                continue
    
    if not data_loader_found:
        print("   ❌ Could not import TCPFormer data loader")
        print("   💡 This suggests we should check the coordinate compatibility manually")

except Exception as e:
    print(f"   ❌ Error testing data loader: {e}")

print("\n🎯 COMPATIBILITY ASSESSMENT:")
print("1. Check if coordinate ranges match between YOLO and ground truth")
print("2. Verify that negative coordinates are preserved in YOLO dataset")
print("3. Ensure coordinate scales are similar (not normalized to [0,1] or [-1,1])")
print("4. Test both pixel coordinates and camera coordinates to see which works better")

print("\n💡 NEXT STEPS:")
print("1. Run this analysis to see coordinate compatibility")
print("2. If ranges don't match, the camera coordinate conversion might be the issue")
print("3. Consider testing with pixel coordinates (without camera conversion)")
print("4. Compare MPJPE results between different coordinate systems")