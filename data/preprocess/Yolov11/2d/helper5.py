import numpy as np
import os

print("🎯 ANALYZING KEYPOINT PRECISION")
print("=" * 60)

# Load datasets
yolo_data = np.load('custom_yolo_dataset.npz', allow_pickle=True)['data'].item()
gt_data = np.load('data_test_3dhp.npz', allow_pickle=True)['data'].item()

total_pixel_errors = []
per_joint_errors = [[] for _ in range(17)]

print("📊 COMPARING YOLO vs GROUND TRUTH 2D KEYPOINTS:")

for seq_name in yolo_data.keys():
    print(f"\n{seq_name}:")
    
    yolo_poses = yolo_data[seq_name]['data_2d']
    gt_poses = gt_data[seq_name]['data_2d']
    yolo_valid = yolo_data[seq_name]['valid'].astype(bool)
    gt_valid = gt_data[seq_name]['valid'].astype(bool)
    
    # Get frames that are valid in both datasets
    both_valid = yolo_valid & gt_valid
    valid_yolo = yolo_poses[both_valid]
    valid_gt = gt_poses[both_valid]
    
    print(f"  Frames valid in both: {np.sum(both_valid)}")
    
    if np.sum(both_valid) == 0:
        continue
    
    # Remove frames where YOLO has zero poses
    non_zero_mask = ~np.all(valid_yolo == 0, axis=(1, 2))
    if np.sum(non_zero_mask) == 0:
        print("  No non-zero YOLO poses found")
        continue
        
    final_yolo = valid_yolo[non_zero_mask]
    final_gt = valid_gt[non_zero_mask]
    
    print(f"  Non-zero YOLO poses: {len(final_yolo)}")
    
    # Calculate pixel-level errors
    pixel_errors = np.sqrt(np.sum((final_yolo - final_gt) ** 2, axis=2))  # Euclidean distance
    
    # Per-frame average error
    frame_errors = np.mean(pixel_errors, axis=1)
    print(f"  Average pixel error per frame: {np.mean(frame_errors):.1f}±{np.std(frame_errors):.1f} pixels")
    print(f"  Min/Max frame error: {np.min(frame_errors):.1f}/{np.max(frame_errors):.1f} pixels")
    
    # Per-joint errors
    joint_errors = np.mean(pixel_errors, axis=0)
    print(f"  Per-joint errors (pixels):")
    joint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    for j, (joint_name, error) in enumerate(zip(joint_names, joint_errors)):
        print(f"    {j:2d} {joint_name:12s}: {error:.1f} pixels")
        per_joint_errors[j].extend(pixel_errors[:, j])
    
    # Check for systematic biases
    mean_diff_x = np.mean(final_yolo[:, :, 0] - final_gt[:, :, 0])
    mean_diff_y = np.mean(final_yolo[:, :, 1] - final_gt[:, :, 1])
    print(f"  Systematic bias: X={mean_diff_x:.1f}, Y={mean_diff_y:.1f} pixels")
    
    total_pixel_errors.extend(frame_errors)

print(f"\n🎯 OVERALL PRECISION ANALYSIS:")
if total_pixel_errors:
    overall_error = np.mean(total_pixel_errors)
    print(f"Overall average pixel error: {overall_error:.1f}±{np.std(total_pixel_errors):.1f} pixels")
    
    print(f"\nWorst performing joints:")
    joint_avg_errors = [np.mean(errors) if errors else 0 for errors in per_joint_errors]
    worst_joints = np.argsort(joint_avg_errors)[-5:]  # Top 5 worst
    
    for idx in worst_joints:
        if per_joint_errors[idx]:
            print(f"  {idx:2d} {joint_names[idx]:12s}: {np.mean(per_joint_errors[idx]):.1f} pixels")
    
    print(f"\n💡 ANALYSIS:")
    if overall_error > 20:
        print("🚨 HIGH PIXEL ERROR - YOLO keypoints are significantly inaccurate")
        print("This explains the high MPJPE. Solutions:")
        print("1. Use a better pre-trained YOLO pose model")
        print("2. Fine-tune YOLO on similar data")
        print("3. Use pose refinement/post-processing")
        print("4. Try different YOLO model sizes (YOLOv11x vs YOLOv11n)")
    elif overall_error > 10:
        print("⚠️  MODERATE PIXEL ERROR - Room for improvement")
        print("Solutions:")
        print("1. Try larger YOLO model (YOLOv11l or YOLOv11x)")
        print("2. Apply temporal smoothing")
        print("3. Use test-time augmentation")
    else:
        print("✅ GOOD PIXEL ACCURACY - The problem might be elsewhere")
        print("Check:")
        print("1. 3D lifting algorithm compatibility")
        print("2. Coordinate system assumptions in TCPFormer")
        print("3. Data preprocessing in the training pipeline")

else:
    print("❌ No valid data found for comparison")

print(f"\n📈 EXPECTED MPJPE vs PIXEL ERROR:")
print("- 5 pixels → ~30-50mm MPJPE")
print("- 10 pixels → ~50-80mm MPJPE") 
print("- 20 pixels → ~80-120mm MPJPE")
print("- 30+ pixels → >120mm MPJPE")