'''
Generate MPI-INF-3DHP training videos with keypoint annotations overlaid

# Generate all sequences with annotations (camera 0 only by default)
python generateTrainVideos.py

# Generate specific subject and sequence (camera 0)
python generateTrainVideos.py --subject S1 --sequence Seq1

# Generate all cameras for a specific sequence
python generateTrainVideos.py --subject S1 --sequence Seq1 --all-cameras

# Custom output and fps
python generateTrainVideos.py --fps 25 --output-dir ./train_annotated_videos
'''

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# MPI-INF-3DHP skeleton connections for 2D visualization
connections_2d = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

# Joint names for reference
JOINT_NAMES = [
    'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

# Colors for different body parts (BGR format for OpenCV)
JOINT_COLORS = {
    'head': (0, 255, 255),      # Yellow - Head, Nose
    'torso': (255, 0, 0),       # Blue - Spine, Thorax, Root
    'left_arm': (0, 255, 0),    # Green - Left arm joints
    'right_arm': (0, 0, 255),   # Red - Right arm joints
    'left_leg': (255, 255, 0),  # Cyan - Left leg joints
    'right_leg': (255, 0, 255), # Magenta - Right leg joints
}

def get_joint_color(joint_idx):
    """Get color for specific joint based on body part"""
    if joint_idx in [9, 10]:  # Nose, Head
        return JOINT_COLORS['head']
    elif joint_idx in [0, 7, 8, 15, 16]:  # Root, Spine, Thorax
        return JOINT_COLORS['torso']
    elif joint_idx in [11, 12, 13]:  # Left arm
        return JOINT_COLORS['left_arm']
    elif joint_idx in [14, 15, 16]:  # Right arm
        return JOINT_COLORS['right_arm']
    elif joint_idx in [4, 5, 6]:  # Left leg
        return JOINT_COLORS['left_leg']
    elif joint_idx in [1, 2, 3]:  # Right leg
        return JOINT_COLORS['right_leg']
    else:
        return (128, 128, 128)  # Gray for unknown

def load_train_annotations(annotations_path):
    """Load training annotations from .npz file"""
    print(f"Loading training annotations from: {annotations_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Training annotations file not found: {annotations_path}")
    
    data = np.load(annotations_path, allow_pickle=True)['data'].item()
    print(f"✓ Loaded training annotations for sequences: {list(data.keys())}")
    
    return data

def denormalize_2d_coordinates(pose_2d, width, height):
    """Convert normalized coordinates [0,1] to pixel coordinates"""
    pose_2d_pixel = pose_2d.copy()
    pose_2d_pixel[:, 0] = pose_2d[:, 0] * width
    pose_2d_pixel[:, 1] = pose_2d[:, 1] * height
    return pose_2d_pixel

def draw_keypoints_on_frame(frame, pose_2d, confidence_threshold=0.1, show_labels=True):
    """Draw 2D keypoints and skeleton on frame"""
    h, w = frame.shape[:2]
    
    # Check if coordinates are normalized or already in pixels
    if np.max(pose_2d[:, :2]) <= 1.0:
        # Convert normalized coordinates to pixel coordinates
        pose_2d_pixel = denormalize_2d_coordinates(pose_2d, w, h)
    else:
        pose_2d_pixel = pose_2d.copy()
    
    # Draw skeleton connections first (behind joints)
    for connection in connections_2d:
        joint1, joint2 = connection
        if joint1 < len(pose_2d_pixel) and joint2 < len(pose_2d_pixel):
            # Check confidence if available
            conf1 = pose_2d_pixel[joint1, 2] if pose_2d_pixel.shape[1] > 2 else 1.0
            conf2 = pose_2d_pixel[joint2, 2] if pose_2d_pixel.shape[1] > 2 else 1.0
            
            if conf1 > confidence_threshold and conf2 > confidence_threshold:
                pt1 = (int(pose_2d_pixel[joint1, 0]), int(pose_2d_pixel[joint1, 1]))
                pt2 = (int(pose_2d_pixel[joint2, 0]), int(pose_2d_pixel[joint2, 1]))
                
                # Use color based on connection type
                color = get_joint_color(joint1)
                cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
    
    # Draw joints on top of skeleton
    for joint_idx, joint_pos in enumerate(pose_2d_pixel):
        # Check confidence if available
        confidence = joint_pos[2] if pose_2d_pixel.shape[1] > 2 else 1.0
        
        if confidence > confidence_threshold:
            x, y = int(joint_pos[0]), int(joint_pos[1])
            
            # Skip joints outside frame boundaries
            if 0 <= x < w and 0 <= y < h:
                color = get_joint_color(joint_idx)
                
                # Draw joint circle
                cv2.circle(frame, (x, y), 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 8, (0, 0, 0), 2, cv2.LINE_AA)  # Black border
                
                # Draw joint labels if requested
                if show_labels:
                    label = f"{joint_idx}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Position label above joint
                    label_x = x - label_size[0] // 2
                    label_y = y - 15
                    
                    # Draw background rectangle for label
                    cv2.rectangle(frame, 
                                (label_x - 2, label_y - label_size[1] - 2),
                                (label_x + label_size[0] + 2, label_y + 2),
                                (0, 0, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (label_x, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame

def create_train_annotated_video(image_folder, camera_data, subject, sequence, camera_id, output_path, fps=25, resize=None, show_labels=True):
    """Create video with keypoint annotations overlaid on frames for training data"""
    
    # Get image files
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
    
    if not image_files:
        print(f"No JPG files found in {image_folder}")
        return False
    
    # Sort files numerically
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]) 
                     if '_' in os.path.basename(x) else int(os.path.splitext(os.path.basename(x))[0]))
    
    print(f"Found {len(image_files)} images for {subject} {sequence} camera {camera_id}")
    
    # Get poses data for this camera
    poses_2d = camera_data['data_2d']  # Shape: (frames, 17, 2)
    poses_3d = camera_data.get('data_3d', None)  # Shape: (frames, 17, 3)
    
    print(f"Loaded {len(poses_2d)} pose annotations for {subject} {sequence} camera {camera_id}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Could not read first image: {image_files[0]}")
        return False
    
    height, width, channels = first_frame.shape
    original_size = (width, height)
    
    # Resize if specified
    if resize:
        width, height = resize
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    print(f"Original image size: {original_size}, Resizing: {'Yes' if resize else 'No'}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False
    
    # Process each frame
    frames_with_annotations = 0
    frames_processed = 0
    
    max_frames = min(len(image_files), len(poses_2d))
    
    for i, image_file in enumerate(tqdm(image_files[:max_frames], desc=f"Processing {subject}_{sequence}_cam{camera_id}")):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        
        # Resize frame if specified
        if resize:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Add annotations if available for this frame
        if i < len(poses_2d):
            pose_2d = poses_2d[i]  # Shape: (17, 2)
            
            # Add dummy confidence scores (training data typically doesn't have confidence)
            if pose_2d.shape[1] == 2:
                confidence_scores = np.ones((pose_2d.shape[0], 1))
                pose_2d = np.concatenate([pose_2d, confidence_scores], axis=1)
            
            # Draw keypoints on frame
            frame = draw_keypoints_on_frame(frame, pose_2d, 
                                          confidence_threshold=0.1, 
                                          show_labels=show_labels)
            frames_with_annotations += 1
        
        # Add frame info
        info_text = f"Frame: {i+1}/{max_frames} | {subject} {sequence} Cam{camera_id}"
        
        # Add background for info text
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (5, 5), (text_size[0] + 15, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write frame to video
        video_writer.write(frame)
        frames_processed += 1
    
    # Release video writer
    video_writer.release()
    
    print(f"✓ Created {subject} {sequence} camera {camera_id} annotated video:")
    print(f"  - Total frames: {frames_processed}")
    print(f"  - Frames with annotations: {frames_with_annotations}")
    print(f"  - Output: {output_path}")
    
    return True

def process_all_train_sequences(base_path, annotations, output_dir, fps=25, resize=None, show_labels=True, all_cameras=False):
    """Process all MPI-INF-3DHP training sequences with annotations (camera 0 by default)"""
    
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    sequences = ['Seq1', 'Seq2']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    for subject in subjects:
        for sequence in sequences:
            print(f"\n{'='*70}")
            print(f"Processing {subject} {sequence}")
            print(f"{'='*70}")
            
            # Create sequence key for annotations
            seq_key = f"{subject} {sequence}"
            
            if seq_key not in annotations:
                print(f"Warning: No annotations found for {seq_key}")
                continue
            
            seq_data = annotations[seq_key]
            
            # Check if this is the nested structure [camera_data, fps]
            if isinstance(seq_data, list) and len(seq_data) >= 1:
                camera_dict = seq_data[0]  # First element contains camera data
                seq_fps = seq_data[1] if len(seq_data) > 1 else fps  # Second element is fps
            else:
                camera_dict = seq_data
                seq_fps = fps
            
            print(f"Available cameras for {seq_key}: {list(camera_dict.keys())}")
            
            # By default only process camera 0, unless all_cameras is True
            if all_cameras:
                cameras_to_process = list(camera_dict.keys())
            else:
                cameras_to_process = ['0'] if '0' in camera_dict else []
                if not cameras_to_process:
                    print(f"Warning: Camera 0 not found for {seq_key}")
                    continue
            
            for camera_id in cameras_to_process:
                total_count += 1
                
                # Input folder path
                image_folder = os.path.join(base_path, subject, sequence, 'imageFrames', f'video_{camera_id}')
                
                # Output video path
                output_path = os.path.join(output_dir, f"{subject}_{sequence}_cam{camera_id}_annotated.mp4")
                
                # Check if input folder exists
                if not os.path.exists(image_folder):
                    print(f"Warning: Image folder not found: {image_folder}")
                    continue
                
                # Check if camera data exists
                camera_data = camera_dict[camera_id]
                if not camera_data or 'data_2d' not in camera_data:
                    print(f"Warning: No 2D data found for {seq_key} camera {camera_id}")
                    continue
                
                # Skip if output already exists (optional)
                if os.path.exists(output_path):
                    response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
                    if response.lower() != 'y':
                        print(f"Skipping {subject} {sequence} camera {camera_id}")
                        continue
                
                # Create annotated video
                if create_train_annotated_video(image_folder, camera_data, subject, sequence, 
                                              camera_id, output_path, seq_fps, resize, show_labels):
                    success_count += 1
                    
                    # Print video info
                    cap = cv2.VideoCapture(output_path)
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / seq_fps
                        print(f"✓ Video info: {frame_count} frames, {duration:.2f}s duration")
                        cap.release()
                else:
                    print(f"✗ Failed to create annotated video for {subject} {sequence} camera {camera_id}")
    
    print(f"\n{'='*70}")
    print(f"TRAINING DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully created annotated videos: {success_count}/{total_count}")
    print(f"Camera mode: {'All cameras' if all_cameras else 'Camera 0 only (default)'}")
    print(f"Output directory: {output_dir}")

def process_specific_sequence(base_path, annotations, subject, sequence, output_dir, fps=25, resize=None, show_labels=True, all_cameras=False):
    """Process a specific training sequence (camera 0 by default)"""
    
    # Create sequence key for annotations
    seq_key = f"{subject} {sequence}"
    
    if seq_key not in annotations:
        print(f"Error: No annotations found for {seq_key}")
        print(f"Available sequences: {list(annotations.keys())}")
        return False
    
    seq_data = annotations[seq_key]
    
    # Check if this is the nested structure [camera_data, fps]
    if isinstance(seq_data, list) and len(seq_data) >= 1:
        camera_dict = seq_data[0]  # First element contains camera data
        seq_fps = seq_data[1] if len(seq_data) > 1 else fps  # Second element is fps
    else:
        camera_dict = seq_data
        seq_fps = fps
    
    print(f"Available cameras for {seq_key}: {list(camera_dict.keys())}")
    
    # By default only process camera 0, unless all_cameras is True
    if all_cameras:
        cameras_to_process = list(camera_dict.keys())
    else:
        cameras_to_process = ['0'] if '0' in camera_dict else []
        if not cameras_to_process:
            print(f"Error: Camera 0 not found for {seq_key}")
            return False
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    
    for cam_id in cameras_to_process:
        # Input folder path
        image_folder = os.path.join(base_path, subject, sequence, 'imageFrames', f'video_{cam_id}')
        
        # Output video path
        output_path = os.path.join(output_dir, f"{subject}_{sequence}_cam{cam_id}_annotated.mp4")
        
        # Check if input folder exists
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue
        
        # Get camera data
        camera_data = camera_dict[cam_id]
        if not camera_data or 'data_2d' not in camera_data:
            print(f"Warning: No 2D data found for {seq_key} camera {cam_id}")
            continue
        
        # Create annotated video
        if create_train_annotated_video(image_folder, camera_data, subject, sequence, 
                                      cam_id, output_path, seq_fps, resize, show_labels):
            success_count += 1
            print(f"✓ Successfully created {subject}_{sequence}_cam{cam_id}_annotated.mp4")
        else:
            print(f"✗ Failed to create annotated video for {subject} {sequence} camera {cam_id}")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Generate MPI-INF-3DHP training videos with keypoint annotations (camera 0 by default)")
    parser.add_argument('--base-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp',
                       help='Base path to mpi_inf_3dhp training directory')
    parser.add_argument('--annotations-path', type=str,
                       default='/nas-ctm01/homes/mfbrandao/TCPFormerForked/data/motion3d/data_train_3dhp.npz',
                       help='Path to training annotations .npz file')
    parser.add_argument('--output-dir', type=str, default='../train_annotated_videos',
                       help='Output directory for annotated MP4 files')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second for output videos')
    parser.add_argument('--resize', type=str, default=None,
                       help='Resize frames to WxH (e.g., "1280x720"), leave empty for original size')
    parser.add_argument('--subject', type=str, default=None,
                       help='Process specific subject only (S1, S2, etc.)')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Process specific sequence only (Seq1, Seq2)')
    parser.add_argument('--all-cameras', action='store_true',
                       help='Process all available cameras (default: camera 0 only)')
    parser.add_argument('--no-labels', action='store_true',
                       help='Hide joint index labels on keypoints')
    parser.add_argument('--preview', action='store_true',
                       help='Preview annotations info before processing')
    
    args = parser.parse_args()
    
    # Parse resize parameter
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize = (width, height)
            print(f"Will resize frames to {width}x{height}")
        except ValueError:
            print("Invalid resize format. Use WIDTHxHEIGHT (e.g., 1280x720)")
            return
    
    # Check if base path exists
    if not os.path.exists(args.base_path):
        print(f"Error: Base path does not exist: {args.base_path}")
        return
    
    try:
        # Load annotations
        annotations = load_train_annotations(args.annotations_path)
        
        # Preview mode
        if args.preview:
            print("\nTraining Annotations Preview:")
            print("="*60)
            for seq_name, seq_data in annotations.items():
                print(f"{seq_name}:")
                
                if isinstance(seq_data, list) and len(seq_data) >= 1:
                    camera_dict = seq_data[0]
                    seq_fps = seq_data[1] if len(seq_data) > 1 else "Unknown"
                    print(f"  - FPS: {seq_fps}")
                    print(f"  - Cameras: {list(camera_dict.keys())}")
                    print(f"  - Camera 0 available: {'Yes' if '0' in camera_dict else 'No'}")
                    
                    if '0' in camera_dict:
                        cam_data = camera_dict['0']
                        if 'data_2d' in cam_data:
                            data_2d = cam_data['data_2d']
                            print(f"    Camera 0: 2D poses shape {data_2d.shape}")
                        if 'data_3d' in cam_data:
                            data_3d = cam_data['data_3d']
                            print(f"    Camera 0: 3D poses shape {data_3d.shape}")
                else:
                    print(f"  - Unexpected data structure")
            return
        
        print(f"\nConfiguration:")
        print(f"  - Base path: {args.base_path}")
        print(f"  - Annotations: {args.annotations_path}")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - FPS: {args.fps}")
        print(f"  - Show labels: {not args.no_labels}")
        print(f"  - Camera mode: {'All cameras' if args.all_cameras else 'Camera 0 only (default)'}")
        if resize:
            print(f"  - Resize to: {resize[0]}x{resize[1]}")
        
        # Process sequences
        if args.subject and args.sequence:
            # Process specific subject and sequence
            if args.subject not in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
                print(f"Invalid subject: {args.subject}. Must be one of: S1-S8")
                return
            
            if args.sequence not in ['Seq1', 'Seq2']:
                print(f"Invalid sequence: {args.sequence}. Must be one of: Seq1, Seq2")
                return
            
            print(f"\nProcessing specific sequence: {args.subject} {args.sequence}")
            
            if process_specific_sequence(args.base_path, annotations, args.subject, args.sequence, 
                                       args.output_dir, args.fps, resize, not args.no_labels, args.all_cameras):
                print(f"✓ Successfully processed {args.subject} {args.sequence}")
            else:
                print(f"✗ Failed to process {args.subject} {args.sequence}")
        else:
            # Process all sequences
            print(f"\nProcessing all training sequences...")
            process_all_train_sequences(args.base_path, annotations, args.output_dir, 
                                      args.fps, resize, not args.no_labels, args.all_cameras)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()