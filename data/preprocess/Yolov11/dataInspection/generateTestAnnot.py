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

def load_test_frames_batch(sequence_name, start_frame=0, num_frames=None):
    """Load test frames for the sequence in batches (same as compare_gt_yolo_2d.py)"""
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
                # Get the requested batch of frames
                end_frame = start_frame + num_frames if num_frames is not None else len(image_files)
                batch_files = image_files[start_frame:end_frame]
                
                frames = []
                for img_path in batch_files:
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        frames.append((frame, os.path.basename(img_path)))
                
                return frames, len(image_files)  # Return frames with filenames and total count
    
    return None, 0

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
    annotation_filename = os.path.splitext(frame_filename)[0] + '.txt'
    
    for base_path in yolo_label_paths:
        annotation_path = os.path.join(base_path, sequence_name, annotation_filename)
        if os.path.exists(annotation_path):
            return parse_yolo_annotation(annotation_path)
    
    return []

def parse_yolo_annotation(annotation_path):
    """Parse YOLO format annotation file"""
    annotations = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def draw_annotations(image, annotations, img_width, img_height):
    """Draw bounding boxes on image"""
    for class_id, x_center, y_center, width, height in annotations:
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw class label
        label = f"Person"  # Class 0 is person in YOLO
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

def create_annotated_video(sequence, output_path, max_frames=None):
    """Create video with annotations for a specific sequence"""
    print(f"Loading frames for sequence: {sequence}")
    
    # Load frames using the same method as compare_gt_yolo_2d.py
    frames_data, total_frames = load_test_frames_batch(sequence, 0, max_frames)
    
    if frames_data is None or len(frames_data) == 0:
        print(f"No frames found for sequence: {sequence}")
        return False
    
    print(f"Found {len(frames_data)} frames to process")
    
    # Get dimensions from first frame
    first_frame, _ = frames_data[0]
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    processed_count = 0
    frames_with_annotations = 0
    
    for frame, filename in frames_data:
        # Get YOLO annotations for this frame
        annotations = parse_yolo_annotation_file(sequence, filename)
        
        # Draw annotations on image
        annotated_image = draw_annotations(frame, annotations, width, height)
        
        # Add frame info
        frame_info = f"Frame: {filename} | Annotations: {len(annotations)}"
        cv2.putText(annotated_image, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add sequence info
        seq_info = f"Sequence: {sequence}"
        cv2.putText(annotated_image, seq_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        video_writer.write(annotated_image)
        processed_count += 1
        
        if len(annotations) > 0:
            frames_with_annotations += 1
    
    video_writer.release()
    print(f"Created annotated video: {output_path}")
    print(f"Processed {processed_count} frames")
    print(f"Frames with annotations: {frames_with_annotations}/{processed_count}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate annotated videos to check ground truth annotations')
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
        output_path = os.path.join(args.output_dir, f"{sequence}_annotated.mp4")
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