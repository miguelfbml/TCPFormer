'''
Generate MPI-INF-3DHP test videos with keypoint annotations overlaid

# Generate all sequences with annotations
python generateTestVideosWithAnnotations.py

# Generate specific sequence
python generateTestVideosWithAnnotations.py --sequence TS1

# Custom output and fps
python generateTestVideosWithAnnotations.py --fps 25 --output-dir ./annotated_videos
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

def load_annotations(annotations_path):
    """Load ground truth annotations from .npz file"""
    print(f"Loading annotations from: {annotations_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    data = np.load(annotations_path, allow_pickle=True)['data'].item()
    print(f"✓ Loaded annotations for sequences: {list(data.keys())}")
    
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

def create_annotated_video(image_folder, annotations, seq_name, output_path, fps=30, resize=None, show_labels=True):
    """Create video with keypoint annotations overlaid on frames"""
    
    # Get image files
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
    
    if not image_files:
        print(f"No JPG files found in {image_folder}")
        return False
    
    # Sort files numerically
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]) 
                     if '_' in os.path.basename(x) else int(os.path.splitext(os.path.basename(x))[0]))
    
    print(f"Found {len(image_files)} images for {seq_name}")
    
    # Load annotations for this sequence
    if seq_name not in annotations:
        print(f"No annotations found for sequence {seq_name}")
        return False
    
    seq_data = annotations[seq_name]
    poses_2d = seq_data['data_2d']  # Shape: (frames, 17, 3) or (frames, 17, 2)
    valid_frames = seq_data.get('valid', np.ones(len(poses_2d), dtype=bool))
    
    print(f"Loaded {len(poses_2d)} pose annotations for {seq_name}")
    
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
    
    # Get sequence-specific image dimensions for coordinate normalization
    if seq_name in ["TS5", "TS6"]:
        original_width, original_height = 1920, 1080
    else:
        original_width, original_height = 2048, 2048
    
    # Process each frame
    frames_with_annotations = 0
    frames_processed = 0
    
    for i, image_file in enumerate(tqdm(image_files, desc=f"Processing {seq_name}")):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        
        # Resize frame if specified
        if resize:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Add annotations if available for this frame
        if i < len(poses_2d) and (i >= len(valid_frames) or valid_frames[i]):
            pose_2d = poses_2d[i]  # Shape: (17, 3) or (17, 2)
            
            # Ensure pose has confidence scores
            if pose_2d.shape[1] == 2:
                # Add dummy confidence scores if not present
                confidence_scores = np.ones((pose_2d.shape[0], 1))
                pose_2d = np.concatenate([pose_2d, confidence_scores], axis=1)
            
            # Draw keypoints on frame
            frame = draw_keypoints_on_frame(frame, pose_2d, 
                                          confidence_threshold=0.1, 
                                          show_labels=show_labels)
            frames_with_annotations += 1
            
            # Add frame info
            info_text = f"Frame: {i+1}/{len(image_files)} | Seq: {seq_name}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
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
    
    print(f"✓ Created {seq_name} annotated video:")
    print(f"  - Total frames: {frames_processed}")
    print(f"  - Frames with annotations: {frames_with_annotations}")
    print(f"  - Output: {output_path}")
    
    return True

def process_all_sequences(base_path, annotations, output_dir, fps=30, resize=None, show_labels=True):
    """Process all MPI-INF-3DHP test sequences with annotations"""
    
    sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq}")
        print(f"{'='*60}")
        
        # Input folder path
        image_folder = os.path.join(base_path, seq, 'imageSequence')
        
        # Output video path
        output_path = os.path.join(output_dir, f"{seq}_annotated.mp4")
        
        # Check if input folder exists
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue
        
        # Check if sequence has annotations
        if seq not in annotations:
            print(f"Warning: No annotations found for {seq}")
            continue
        
        # Skip if output already exists (optional)
        if os.path.exists(output_path):
            response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print(f"Skipping {seq}")
                continue
        
        # Create annotated video
        if create_annotated_video(image_folder, annotations, seq, output_path, fps, resize, show_labels):
            success_count += 1
            
            # Print video info
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                print(f"✓ Video info: {frame_count} frames, {duration:.2f}s duration")
                cap.release()
        else:
            print(f"✗ Failed to create annotated video for {seq}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully created annotated videos: {success_count}/{len(sequences)}")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate MPI-INF-3DHP test videos with keypoint annotations")
    parser.add_argument('--base-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
                       help='Base path to mpi_inf_3dhp_test_set directory')
    parser.add_argument('--annotations-path', type=str,
                       default='/nas-ctm01/homes/mfbrandao/TCPFormerForked/data/motion3d/data_test_3dhp.npz',
                       help='Path to annotations .npz file')
    parser.add_argument('--output-dir', type=str, default='../annotated_videos',
                       help='Output directory for annotated MP4 files')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second for output videos')
    parser.add_argument('--resize', type=str, default=None,
                       help='Resize frames to WxH (e.g., "1280x720"), leave empty for original size')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Process specific sequence only (TS1, TS2, etc.)')
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
        annotations = load_annotations(args.annotations_path)
        
        # Preview mode
        if args.preview:
            print("\nAnnotations Preview:")
            print("="*50)
            for seq_name, seq_data in annotations.items():
                data_2d = seq_data['data_2d']
                valid_frames = seq_data.get('valid', np.ones(len(data_2d), dtype=bool))
                valid_count = np.sum(valid_frames) if len(valid_frames) > 0 else len(data_2d)
                
                print(f"{seq_name}:")
                print(f"  - 2D poses shape: {data_2d.shape}")
                print(f"  - Valid frames: {valid_count}/{len(data_2d)}")
                print(f"  - Coordinate range: X[{np.min(data_2d[:,:,0]):.3f}, {np.max(data_2d[:,:,0]):.3f}], "
                      f"Y[{np.min(data_2d[:,:,1]):.3f}, {np.max(data_2d[:,:,1]):.3f}]")
            return
        
        print(f"\nConfiguration:")
        print(f"  - Base path: {args.base_path}")
        print(f"  - Annotations: {args.annotations_path}")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - FPS: {args.fps}")
        print(f"  - Show labels: {not args.no_labels}")
        if resize:
            print(f"  - Resize to: {resize[0]}x{resize[1]}")
        
        # Process sequences
        if args.sequence:
            if args.sequence not in ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']:
                print(f"Invalid sequence: {args.sequence}. Must be one of: TS1, TS2, TS3, TS4, TS5, TS6")
                return
            
            print(f"\nProcessing single sequence: {args.sequence}")
            image_folder = os.path.join(args.base_path, args.sequence, 'imageSequence')
            output_path = os.path.join(args.output_dir, f"{args.sequence}_annotated.mp4")
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            if create_annotated_video(image_folder, annotations, args.sequence, output_path, 
                                    args.fps, resize, not args.no_labels):
                print(f"✓ Successfully created {args.sequence}_annotated.mp4")
            else:
                print(f"✗ Failed to create annotated video for {args.sequence}")
        else:
            # Process all sequences
            process_all_sequences(args.base_path, annotations, args.output_dir, 
                                args.fps, resize, not args.no_labels)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()