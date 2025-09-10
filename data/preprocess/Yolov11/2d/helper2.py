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
        print(f"    Valid flags type: {type(valid_flags[0])}")
        print(f"    Valid frames: {np.sum(valid_flags)}/{len(valid_flags)}")
        
        # Fix: Convert valid_flags to boolean if needed
        if valid_flags.dtype != bool:
            valid_bool = valid_flags.astype(bool)
        else:
            valid_bool = valid_flags
        
        # Get coordinate statistics
        valid_poses = poses_2d[valid_bool]
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

# Check YOLO dataset
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
            
            for seq_name in list(yolo_data.keys())[:1]:  # Check first sequence
                yolo_seq = yolo_data[seq_name]
                orig_seq = original_data[seq_name]
                
                print(f"\n    {seq_name} comparison:")
                print(f"      Original 2D shape: {orig_seq['data_2d'].shape}")
                print(f"      YOLO 2D shape: {yolo_seq['data_2d'].shape}")
                
                # Fix valid flags for both datasets
                orig_valid = orig_seq['valid'].astype(bool)
                yolo_valid = yolo_seq['valid'].astype(bool)
                
                valid_yolo_poses = yolo_seq['data_2d'][yolo_valid]
                if len(valid_yolo_poses) > 0:
                    x_coords = valid_yolo_poses[:, :, 0].flatten()
                    y_coords = valid_yolo_poses[:, :, 1].flatten()
                    
                    print(f"      YOLO X range: [{np.min(x_coords):.1f}, {np.max(x_coords):.1f}]")
                    print(f"      YOLO Y range: [{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
                    
                    # Compare with ground truth
                    orig_poses = orig_seq['data_2d'][orig_valid]
                    if len(orig_poses) > 0:
                        orig_x = orig_poses[:, :, 0].flatten()
                        orig_y = orig_poses[:, :, 1].flatten()
                        
                        print(f"      GT X range: [{np.min(orig_x):.1f}, {np.max(orig_x):.1f}]")
                        print(f"      GT Y range: [{np.min(orig_y):.1f}, {np.max(orig_y):.1f}]")
                        
                        # Check if coordinates are in similar ranges
                        x_scale_diff = abs(np.max(x_coords) - np.max(orig_x)) / max(abs(np.max(orig_x)), 1)
                        y_scale_diff = abs(np.max(y_coords) - np.max(orig_y)) / max(abs(np.max(orig_y)), 1)
                        
                        print(f"      Coordinate scale match: X_diff={x_scale_diff:.3f}, Y_diff={y_scale_diff:.3f}")
                        print(f"      Good match: {x_scale_diff < 0.1 and y_scale_diff < 0.1}")
                
        except Exception as e:
            print(f"    ❌ Error loading YOLO dataset: {e}")

print("\n3. Examining TCPFormer's data processing...")

# Check the key files we found
key_files = [
    'utils/data.py',
    'data/reader/generator_3dhp.py'
]

for file_path in key_files:
    full_path = os.path.join(tcpformer_root, file_path)
    if os.path.exists(full_path):
        print(f"\n📄 Examining: {file_path}")
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Look for normalization/coordinate processing
            lines = content.split('\n')
            relevant_lines = []
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['normalize', '2d', 'coord', 'data_2d', 'transform']):
                    # Get some context around the line
                    start = max(0, i-1)
                    end = min(len(lines), i+2)
                    context = lines[start:end]
                    relevant_lines.extend(context)
            
            if relevant_lines:
                print("   🔍 Relevant code sections:")
                for line in relevant_lines[:20]:  # Show first 20 relevant lines
                    if line.strip():
                        print(f"     {line}")
                if len(relevant_lines) > 20:
                    print(f"     ... ({len(relevant_lines)-20} more lines)")
            
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")

print("\n🎯 FINAL ASSESSMENT:")
print("Based on the analysis:")
print("1. TCPFormer uses a 'denormalize' function - this suggests it normalizes coordinates")
print("2. Ground truth coordinates have negative values and extend beyond image boundaries")
print("3. Your YOLO coordinates should match the ground truth coordinate system")
print("4. If YOLO coordinates are very different from GT, the camera conversion might be wrong")