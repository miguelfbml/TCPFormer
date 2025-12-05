"""
Real-time 2D Pose Detection using YOLO model with webcam
Displays live pose estimation with skeleton overlay on video feed

Usage:
python YOLOestimation.py
python YOLOestimation.py --camera 0 --conf 0.5
python YOLOestimation.py --show-fps
"""

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
from thop import profile, clever_format

# MPI-INF-3DHP joint names and connections for visualization
JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand', 
    'RShoulder', 'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle',
    'RHip', 'RKnee', 'RAnkle', 'Sacrum', 'Spine', 'Neck'
]

CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

# Color scheme for skeleton
SKELETON_COLOR = (0, 255, 0)  # Green
JOINT_COLOR = (255, 0, 0)  # Blue
TEXT_COLOR = (255, 255, 255)  # White
FPS_COLOR = (0, 255, 255)  # Yellow
ANALYTICS_COLOR = (255, 128, 0)  # Orange

# Default model path
DEFAULT_MODEL_PATH = 'weights/yolo/best.pt'

def check_gpu():
    """Check if GPU is available"""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda:0'
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ GPU not available, using CPU")
    return device

def get_model_complexity(model, img_size=640):
    """
    Calculate model complexity metrics: parameters, GFLOPs
    """
    try:
        # Get the underlying PyTorch model
        if hasattr(model, 'model'):
            pytorch_model = model.model
        else:
            pytorch_model = model
        
        # Create dummy input
        device = next(pytorch_model.parameters()).device
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        
        # Calculate FLOPs and parameters
        flops, params = profile(pytorch_model, inputs=(dummy_input,), verbose=False)
        
        # Convert to billions (GFLOPs) and millions (M params)
        gflops = flops / 1e9
        params_m = params / 1e6
        
        return {
            'params': params,
            'params_m': params_m,
            'flops': flops,
            'gflops': gflops
        }
    except Exception as e:
        print(f"⚠ Could not calculate model complexity: {e}")
        return {
            'params': 0,
            'params_m': 0,
            'flops': 0,
            'gflops': 0
        }

def draw_skeleton(frame, keypoints, confidences, conf_threshold=0.3):
    """Draw skeleton on frame with keypoints and connections"""
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    # Draw connections
    for connection in CONNECTIONS_2D:
        joint1_idx, joint2_idx = connection
        if joint1_idx < len(keypoints) and joint2_idx < len(keypoints):
            # Check confidence
            if confidences[joint1_idx] > conf_threshold and confidences[joint2_idx] > conf_threshold:
                pt1 = tuple(map(int, keypoints[joint1_idx]))
                pt2 = tuple(map(int, keypoints[joint2_idx]))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)
    
    # Draw joints
    for idx, (x, y) in enumerate(keypoints):
        if confidences[idx] > conf_threshold:
            center = (int(x), int(y))
            cv2.circle(frame, center, 5, JOINT_COLOR, -1)
            cv2.circle(frame, center, 6, (255, 255, 255), 1)
            
            # Draw joint index (optional, can comment out for cleaner view)
            # cv2.putText(frame, str(idx), (int(x)+10, int(y)+10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    return frame

def print_keypoint_data(frame_num, keypoints, confidences):
    """Print keypoint coordinates and confidences in the order they would be saved"""
    print(f"\n{'='*80}")
    print(f"Frame {frame_num}")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Joint':<20} {'X':<12} {'Y':<12} {'Confidence':<12}")
    print(f"{'-'*80}")
    
    # Print each joint's data (17 joints)
    for idx in range(17):
        if idx < len(keypoints) and idx < len(confidences):
            x, y = keypoints[idx]
            conf = confidences[idx]
            joint_name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Joint_{idx}"
            print(f"{joint_name:<20} {x:<12.2f} {y:<12.2f} {conf:<12.4f}")
        else:
            joint_name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Joint_{idx}"
            print(f"{joint_name:<20} {0.0:<12.2f} {0.0:<12.2f} {0.0:<12.4f}")
    
    print(f"{'='*80}\n")

def draw_performance_analytics(frame, fps, gflops, params_m, inference_time_ms):
    """Draw performance analytics overlay on frame"""
    # Create semi-transparent overlay box
    overlay = frame.copy()
    box_height = 150
    box_width = 320
    cv2.rectangle(overlay, (10, frame.shape[0] - box_height - 10), 
                  (10 + box_width, frame.shape[0] - 10), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw analytics text
    y_offset = frame.shape[0] - box_height + 5
    line_height = 30
    
    cv2.putText(frame, '=== Performance Analytics ===', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANALYTICS_COLOR, 1)
    y_offset += line_height
    
    cv2.putText(frame, f'FPS: {fps:.1f}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FPS_COLOR, 2)
    y_offset += line_height
    
    cv2.putText(frame, f'Inference: {inference_time_ms:.1f} ms', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FPS_COLOR, 2)
    y_offset += line_height
    
    cv2.putText(frame, f'GFLOPs: {gflops:.2f}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANALYTICS_COLOR, 2)
    y_offset += line_height
    
    cv2.putText(frame, f'Parameters: {params_m:.2f}M', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANALYTICS_COLOR, 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Real-time 2D Pose Detection with YOLO')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for YOLO inference (default: 640)')
    parser.add_argument('--show-fps', action='store_true',
                       help='Show FPS counter on display')
    parser.add_argument('--show-analytics', action='store_true',
                       help='Show performance analytics (GFLOPs, params, FPS)')
    parser.add_argument('--print-keypoints', action='store_true',
                       help='Print keypoint coordinates and confidences to console')
    args = parser.parse_args()
    
    # Use default model path
    model_path = DEFAULT_MODEL_PATH
    
    print("="*60)
    print("Real-time 2D Pose Detection with YOLO")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Camera: {args.camera}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Image size: {args.img_size}")
    print(f"Print keypoints: {args.print_keypoints}")
    print(f"Show analytics: {args.show_analytics}")
    print("="*60)
    print("Press 'q' to quit")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print(f"   Please ensure the model exists at: {os.path.abspath(model_path)}")
        return
    
    # Check GPU availability
    device = check_gpu()
    
    # Load YOLO model
    print(f"Loading YOLO model...")
    try:
        model = YOLO(model_path)
        model.to(device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Calculate model complexity
    print("\nCalculating model complexity...")
    complexity = get_model_complexity(model, img_size=args.img_size)
    
    print(f"\n{'='*60}")
    print("Model Complexity Metrics:")
    print(f"{'='*60}")
    print(f"Parameters: {complexity['params_m']:.2f}M ({complexity['params']:,})")
    print(f"GFLOPs: {complexity['gflops']:.2f} ({complexity['flops']:,} FLOPs)")
    print(f"{'='*60}\n")
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened successfully")
    print("Starting real-time detection...")
    
    # FPS calculation variables for inference only
    inference_times = []
    window_size = 30  # Average over last 30 frames
    current_inference_fps = 0
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Measure inference time only
            inference_start = time.time()
            
            # Run YOLO inference
            results = model.predict(
                frame,
                verbose=False,
                imgsz=args.img_size,
                conf=args.conf,
                device=device
            )
            
            inference_end = time.time()
            inference_time = inference_end - inference_start
            
            # Store inference time and calculate FPS
            inference_times.append(inference_time)
            if len(inference_times) > window_size:
                inference_times.pop(0)
            
            # Calculate average inference FPS
            avg_inference_time = np.mean(inference_times)
            current_inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            
            # Process results
            if (results and len(results) > 0 and 
                hasattr(results[0], 'keypoints') and 
                results[0].keypoints is not None and 
                len(results[0].keypoints.xy) > 0):
                
                # Get keypoints and confidences for first detection
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                confidences = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
                
                # Print keypoint data to console if requested
                if args.print_keypoints:
                    print_keypoint_data(frame_count, keypoints, confidences)
                
                # Draw skeleton on frame
                frame = draw_skeleton(frame, keypoints, confidences, conf_threshold=args.conf)
                
                # Display detection info
                num_detections = len(results[0].keypoints.xy)
                cv2.putText(frame, f'Detections: {num_detections}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            else:
                # No detection
                if args.print_keypoints:
                    print(f"\nFrame {frame_count}: No person detected")
                cv2.putText(frame, 'No person detected', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display performance analytics
            if args.show_analytics:
                frame = draw_performance_analytics(
                    frame, 
                    current_inference_fps, 
                    complexity['gflops'], 
                    complexity['params_m'],
                    inference_time * 1000
                )
            elif args.show_fps:
                # Just show FPS if analytics not requested
                cv2.putText(frame, f'Inference FPS: {current_inference_fps:.1f}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, FPS_COLOR, 2)
                cv2.putText(frame, f'Inference Time: {inference_time*1000:.1f}ms', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, FPS_COLOR, 2)
            
            # Display instructions
            cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            
            # Show frame
            cv2.imshow('Real-time Pose Detection', frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released and windows closed")
        
        # Print final statistics
        print(f"\n{'='*60}")
        print("Final Performance Summary:")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_count}")
        if len(inference_times) > 0:
            final_avg_inference = np.mean(inference_times)
            final_inference_fps = 1.0 / final_avg_inference
            print(f"Average inference FPS: {final_inference_fps:.1f}")
            print(f"Average inference time: {final_avg_inference*1000:.1f}ms")
        print(f"\nModel Complexity:")
        print(f"Parameters: {complexity['params_m']:.2f}M")
        print(f"GFLOPs: {complexity['gflops']:.2f}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()