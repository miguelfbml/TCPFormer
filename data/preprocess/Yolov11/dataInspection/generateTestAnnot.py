import os
import cv2
import argparse
import numpy as np
import glob
from pathlib import Path

def load_test_3d_data_from_dataset(sequence_name):
    """Load test data for a specific sequence (same as compare_gt_yolo_2d.py)"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
        '../../../motion3d/data_test_3dhp.npz',
        '../../../../motion3d/data_test_3dhp.npz'
    ]
    
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)['data'].item()
            
            if sequence_name in data:
                seq_data = data[sequence_name]
                poses_2d = seq_data['data_2d']  # Shape: (frames, 17, 2)
                poses_3d = seq_data['data_3d']  # Shape: (frames, 17, 3)
                
                return poses_2d, poses_3d, sequence_name
    
    return None, None, None

def get_available_sequences():
    """Get list of available test sequences from data file (same as compare_gt_yolo_2d.py)"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
        '../../../motion3d/data_test_3dhp.npz',
        '../../../../motion3d/data_test_3dhp.npz'
    ]
    
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)['data'].item()
            return list(data.keys())
    
    # Fallback to default sequences
    return ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

def get_image_files_list(sequence_name):
    """Get the list of image files for a sequence without loading them"""
    test_image_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
        '../motion3d/mpi_inf_3dhp_test_set',
        '../../motion3d/mpi_inf_3dhp_test_set',
        '../../../motion3d/mpi_inf_3dhp_test_set'
    ]
    
    for base_path in test_image_paths:
        image_folder = os.path.join(base_path, sequence_name, 'imageSequence')
        if os.path.exists(image_folder):
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
            image_files.sort()
            
            if image_files:
                print(f"✓ Found image folder: {image_folder}")
                print(f"✓ Found {len(image_files)} images")
                return image_files
    
    print(f"✗ No image folder found for {sequence_name}")
    return []

def load_single_frame(image_path):
    """Load a single frame from file path"""
    frame = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    return frame, filename

def parse_yolo_annotation_file(sequence_name, frame_filename):
    """Parse YOLO annotation for a specific frame in the Yolo dataset"""
    # Try to find the YOLO annotation files
    yolo_label_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo/labels/val',
        '/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo/labels/train',
        './mpi_inf_3dhp_Yolo/labels/val',
        './mpi_inf_3dhp_Yolo/labels/train'
    ]
    
    # Convert image filename to annotation filename
    base_filename = os.path.splitext(frame_filename)[0]  # Remove extension: frame000450
    annotation_filename = f"{sequence_name}_{base_filename}.txt"  # TS6_frame000450.txt
    
    # Debug: print what we're looking for
    print(f"DEBUG: Looking for annotation file: {annotation_filename}")
    print(f"DEBUG: Image filename: {frame_filename} -> Base: {base_filename}")
    
    for base_path in yolo_label_paths:
        # Try different directory structures
        possible_paths = [
            os.path.join(base_path, sequence_name, annotation_filename),  # /path/TS6/TS6_frame000450.txt
            os.path.join(base_path, annotation_filename),  # /path/TS6_frame000450.txt
        ]
        
        for annotation_path in possible_paths:
            print(f"DEBUG: Checking path: {annotation_path}")
            if os.path.exists(annotation_path):
                print(f"✓ Found annotation file: {annotation_path}")
                return parse_yolo_annotation(annotation_path)
            else:
                print(f"✗ Not found: {annotation_path}")
    
    print(f"✗ No annotation file found for {frame_filename}")
    return []

def parse_yolo_annotation(annotation_path):
    """Parse YOLO format annotation file with pose keypoints"""
    annotations = []
    print(f"DEBUG: Parsing annotation file: {annotation_path}")
    
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            print(f"DEBUG: File has {len(lines)} lines")
            
            for line_num, line in enumerate(lines):
                parts = line.strip().split()
                print(f"DEBUG: Line {line_num}: {len(parts)} parts")
                
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    print(f"DEBUG: Parsed bbox - class:{class_id}, center:({x_center:.3f},{y_center:.3f}), size:({width:.3f},{height:.3f})")
                    
                    # Parse keypoints (after bbox, all remaining values are keypoints in x,y,v format)
                    keypoints = []
                    keypoint_start = 5
                    num_remaining = len(parts) - keypoint_start
                    
                    print(f"DEBUG: {num_remaining} values remaining for keypoints")
                    
                    # Process keypoints in groups of 3 (x, y, visibility)
                    if num_remaining >= 3 and num_remaining % 3 == 0:
                        for i in range(keypoint_start, len(parts), 3):
                            if i + 2 < len(parts):
                                kpt_x = float(parts[i])
                                kpt_y = float(parts[i + 1])
                                kpt_v = float(parts[i + 2])  # visibility
                                keypoints.append((kpt_x, kpt_y, kpt_v))
                    
                    print(f"DEBUG: Parsed {len(keypoints)} keypoints")
                    if keypoints:
                        visible_count = sum(1 for _, _, v in keypoints if v > 0)
                        print(f"DEBUG: {visible_count} visible keypoints")
                        print(f"DEBUG: First few keypoints: {keypoints[:3]}")
                    
                    annotations.append((class_id, x_center, y_center, width, height, keypoints))
                else:
                    print(f"DEBUG: Line {line_num} has insufficient parts: {len(parts)}")
    
    print(f"DEBUG: Total annotations parsed: {len(annotations)}")
    return annotations

# MPI-INF-3DHP joint connections for skeleton drawing
CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

# Joint names for reference
JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand', 'RShoulder', 
    'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle', 
    'Sacrum', 'Spine', 'Neck'
]

def draw_annotations(image, annotations, img_width, img_height):
    """Draw bounding boxes and keypoints on image"""
    for annotation in annotations:
        if len(annotation) == 6:
            class_id, x_center, y_center, width, height, keypoints = annotation
        else:
            # Fallback for old format without keypoints
            class_id, x_center, y_center, width, height = annotation[:5]
            keypoints = []
        
        print(f"DEBUG: Drawing annotation with {len(keypoints)} keypoints")
        
        # Draw bounding box
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw class label
        label = f"Person"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw keypoints if available
        if keypoints and len(keypoints) > 0:
            # Convert normalized keypoint coordinates to pixel coordinates
            pixel_keypoints = []
            visible_keypoints = 0
            
            for i, (kpt_x, kpt_y, kpt_v) in enumerate(keypoints):
                if kpt_v > 0:  # Only process visible keypoints
                    px = int(kpt_x * img_width)
                    py = int(kpt_y * img_height)
                    pixel_keypoints.append((px, py, kpt_v))
                    visible_keypoints += 1
                    print(f"DEBUG: Keypoint {i}: ({kpt_x:.3f},{kpt_y:.3f}) -> ({px},{py}) visible")
                else:
                    pixel_keypoints.append((0, 0, 0))
                    print(f"DEBUG: Keypoint {i}: not visible")
            
            print(f"DEBUG: {visible_keypoints} visible keypoints to draw")
            
            # Draw keypoint connections (skeleton) only if we have enough keypoints
            if len(pixel_keypoints) >= 17:
                for connection in CONNECTIONS_2D:
                    joint1_idx, joint2_idx = connection
                    if (joint1_idx < len(pixel_keypoints) and joint2_idx < len(pixel_keypoints) and
                        pixel_keypoints[joint1_idx][2] > 0 and pixel_keypoints[joint2_idx][2] > 0):
                        
                        pt1 = (pixel_keypoints[joint1_idx][0], pixel_keypoints[joint1_idx][1])
                        pt2 = (pixel_keypoints[joint2_idx][0], pixel_keypoints[joint2_idx][1])
                        cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # Blue lines for skeleton
            
            # Draw keypoints as circles
            for i, (px, py, kpt_v) in enumerate(pixel_keypoints):
                if kpt_v > 0:
                    # Different colors for different body parts
                    if i == 0:  # Head
                        color = (0, 0, 255)  # Red
                    elif i in [2, 3, 4, 5, 6, 7]:  # Arms
                        color = (255, 255, 0)  # Cyan
                    elif i in [8, 9, 10, 11, 12, 13]:  # Legs
                        color = (0, 255, 255)  # Yellow
                    else:  # Torso
                        color = (255, 0, 255)  # Magenta
                    
                    cv2.circle(image, (px, py), 4, color, -1)
                    # Add joint number
                    cv2.putText(image, str(i), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    print(f"DEBUG: Drew keypoint {i} at ({px},{py})")
    
    return image

def create_annotated_video(sequence, output_path, max_frames=None):
    """Create video with annotations for a specific sequence - memory efficient version"""
    print(f"Loading image file list for sequence: {sequence}")
    
    # Get list of image files without loading them
    image_files = get_image_files_list(sequence)
    
    if not image_files:
        print(f"No image files found for sequence: {sequence}")
        return False
    
    # Limit number of frames if specified
    if max_frames is not None:
        image_files = image_files[:max_frames]
    
    print(f"Found {len(image_files)} frames to process")
    
    # Load first frame to get dimensions
    first_frame, _ = load_single_frame(image_files[0])
    if first_frame is None:
        print(f"Could not load first frame: {image_files[0]}")
        return False
    
    height, width, _ = first_frame.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    processed_count = 0
    frames_with_annotations = 0
    frames_with_keypoints = 0
    total_keypoints_found = 0
    
    # Process frames one by one
    for i, image_path in enumerate(image_files):
        print(f"\n--- Processing frame {i+1}/{len(image_files)} ---")
        
        # Load single frame
        frame, filename = load_single_frame(image_path)
        
        if frame is None:
            print(f"Could not load frame: {image_path}")
            continue
        
        # Get YOLO annotations for this frame
        annotations = parse_yolo_annotation_file(sequence, filename)
        
        # Count annotations with keypoints
        has_keypoints = False
        frame_keypoints = 0
        for annotation in annotations:
            if len(annotation) == 6 and annotation[5]:  # Has keypoints
                keypoints = annotation[5]
                visible_kpts = sum(1 for _, _, v in keypoints if v > 0)
                if visible_kpts > 0:
                    has_keypoints = True
                    frame_keypoints += visible_kpts
        
        total_keypoints_found += frame_keypoints
        
        print(f"Frame {filename}: {len(annotations)} annotations, {frame_keypoints} keypoints")
        
        # Draw annotations on image
        annotated_image = draw_annotations(frame, annotations, width, height)
        
        # Add frame info
        keypoint_info = f" | Keypoints: {frame_keypoints}" if has_keypoints else " | Keypoints: 0"
        frame_info = f"Frame: {filename} | Annotations: {len(annotations)}{keypoint_info}"
        cv2.putText(annotated_image, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add sequence info
        seq_info = f"Sequence: {sequence} | Progress: {i+1}/{len(image_files)}"
        cv2.putText(annotated_image, seq_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        legend_y = 90
        cv2.putText(annotated_image, "Legend: Green=BBox, Blue=Skeleton, Red=Head, Cyan=Arms, Yellow=Legs, Magenta=Torso", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Write frame to video
        video_writer.write(annotated_image)
        processed_count += 1
        
        if len(annotations) > 0:
            frames_with_annotations += 1
        if has_keypoints:
            frames_with_keypoints += 1
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} frames...")
        
        # Clear frame from memory
        del frame, annotated_image
        
        # Process only first few frames for debugging
        if max_frames and i >= 5:
            print("DEBUG: Processing only first 5 frames for debugging")
            break
    
    video_writer.release()
    print(f"Created annotated video: {output_path}")
    print(f"Processed {processed_count} frames")
    print(f"Frames with annotations: {frames_with_annotations}/{processed_count}")
    print(f"Frames with keypoints: {frames_with_keypoints}/{processed_count}")
    print(f"Total visible keypoints found: {total_keypoints_found}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate annotated videos to check ground truth annotations with keypoints')
    parser.add_argument('--sequence', type=str, help='Sequence name to generate video for (e.g., TS1, TS2, etc.)')
    parser.add_argument('--all', action='store_true', help='Process all available sequences')
    parser.add_argument('--list', action='store_true', help='List available sequences and exit')
    parser.add_argument('--output_dir', type=str, default='./annotated_videos', help='Output directory for videos')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process per sequence')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List sequences and exit if requested
    if args.list:
        sequences = get_available_sequences()
        print("Available sequences:")
        for seq in sequences:
            print(f"  - {seq}")
        return
    
    # Determine which sequences to process
    if args.all:
        sequences = get_available_sequences()
        if not sequences:
            print("No sequences found")
            return
        print(f"Processing all {len(sequences)} sequences: {sequences}")
    elif args.sequence:
        sequences = [args.sequence]
        print(f"Processing sequence: {args.sequence}")
    else:
        print("Error: Must specify either --sequence <name> or --all or --list")
        parser.print_help()
        return
    
    # Generate videos for selected sequences
    successful = 0
    failed = 0
    
    for sequence in sequences:
        output_path = os.path.join(args.output_dir, f"{sequence}_annotated_with_keypoints.mp4")
        print(f"\n{'='*50}")
        print(f"Processing sequence: {sequence}")
        print(f"{'='*50}")
        
        success = create_annotated_video(sequence, output_path, args.max_frames)
        if success:
            successful += 1
        else:
            failed += 1
            print(f"Failed to create video for sequence: {sequence}")
    
    print(f"\n{'='*50}")
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()