import numpy as np

print("🔍 DIAGNOSING MPJPE ISSUE")
print("=" * 60)

# Load your YOLO dataset
yolo_file = 'custom_yolo_dataset.npz'
if os.path.exists(yolo_file):
    print(f"📁 Loading YOLO dataset: {yolo_file}")
    yolo_data = np.load(yolo_file, allow_pickle=True)['data'].item()
    
    # Load ground truth for comparison
    gt_data = np.load('data_test_3dhp.npz', allow_pickle=True)['data'].item()
    
    total_detection_failures = 0
    total_frames = 0
    
    for seq_name in yolo_data.keys():
        print(f"\n{seq_name} Analysis:")
        
        yolo_seq = yolo_data[seq_name]
        gt_seq = gt_data[seq_name]
        
        yolo_poses = yolo_seq['data_2d']
        yolo_valid = yolo_seq['valid'].astype(bool)
        gt_valid = gt_seq['valid'].astype(bool)
        
        print(f"  Total frames: {len(yolo_poses)}")
        print(f"  GT valid frames: {np.sum(gt_valid)}")
        print(f"  YOLO valid frames: {np.sum(yolo_valid)}")
        
        # Check how many frames YOLO completely failed on
        zero_poses = np.sum(np.all(yolo_poses == 0, axis=(1, 2)))
        print(f"  YOLO zero poses: {zero_poses}/{len(yolo_poses)} ({zero_poses/len(yolo_poses)*100:.1f}%)")
        
        # Check frames where GT is valid but YOLO failed
        gt_valid_yolo_failed = np.sum(gt_valid & ~yolo_valid)
        print(f"  GT valid but YOLO failed: {gt_valid_yolo_failed}")
        
        # Check coordinate ranges for successful detections
        valid_yolo_poses = yolo_poses[yolo_valid]
        if len(valid_yolo_poses) > 0:
            valid_coords = valid_yolo_poses[~np.all(valid_yolo_poses == 0, axis=(1,2))]
            if len(valid_coords) > 0:
                x_coords = valid_coords[:, :, 0].flatten()
                y_coords = valid_coords[:, :, 1].flatten()
                print(f"  YOLO coord range: X[{np.min(x_coords):.1f}-{np.max(x_coords):.1f}], Y[{np.min(y_coords):.1f}-{np.max(y_coords):.1f}]")
        
        total_detection_failures += zero_poses
        total_frames += len(yolo_poses)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total detection failure rate: {total_detection_failures}/{total_frames} ({total_detection_failures/total_frames*100:.1f}%)")
    
    if total_detection_failures/total_frames > 0.3:
        print("🚨 HIGH DETECTION FAILURE RATE - This is likely the main MPJPE issue!")
        print("Solutions:")
        print("1. Lower YOLO confidence threshold (try 0.1 instead of 0.25)")
        print("2. Use multiple confidence thresholds and pick best")
        print("3. Implement pose tracking/interpolation for failed frames")

else:
    print(f"❌ YOLO dataset not found: {yolo_file}")

print("\n🔍 LIKELY CAUSES OF HIGH MPJPE:")
print("1. **Detection Failures**: YOLO not detecting people in many frames")
print("2. **Confidence Too High**: Missing valid poses due to strict thresholds") 
print("3. **Keypoint Accuracy**: YOLO keypoints not precise enough")
print("4. **Scale Issues**: Image preprocessing affecting detection quality")

print("\n💡 SOLUTIONS TO TRY:")
print("1. **Lower confidence threshold**: Try 0.1 or even 0.05")
print("2. **Multiple model sizes**: Try YOLOv11l or YOLOv11x for better accuracy")
print("3. **TTA (Test Time Augmentation)**: Multiple scales/flips during inference")
print("4. **Post-processing**: Smooth poses over time, interpolate missing frames")