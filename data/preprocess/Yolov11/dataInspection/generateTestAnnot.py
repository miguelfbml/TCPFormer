import os
import cv2
import argparse
import numpy as np
from pathlib import Path

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
        label = f"Class {class_id}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

def create_annotated_video(images_dir, labels_dir, sequence, output_path):
    """Create video with annotations for a specific sequence"""
    sequence_images_dir = os.path.join(images_dir, sequence)
    sequence_labels_dir = os.path.join(labels_dir, sequence)
    
    if not os.path.exists(sequence_images_dir):
        print(f"Images directory not found: {sequence_images_dir}")
        return False
    
    if not os.path.exists(sequence_labels_dir):
        print(f"Labels directory not found: {sequence_labels_dir}")
        return False
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(sequence_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"No image files found in {sequence_images_dir}")
        return False
    
    # Read first image to get dimensions
    first_image_path = os.path.join(sequence_images_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Could not read first image: {first_image_path}")
        return False
    
    height, width, _ = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    processed_count = 0
    for image_file in image_files:
        image_path = os.path.join(sequence_images_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        
        # Get corresponding annotation file
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(sequence_labels_dir, annotation_file)
        
        # Parse annotations
        annotations = parse_yolo_annotation(annotation_path)
        
        # Draw annotations on image
        annotated_image = draw_annotations(image, annotations, width, height)
        
        # Add frame info
        frame_info = f"Frame: {image_file} | Annotations: {len(annotations)}"
        cv2.putText(annotated_image, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        video_writer.write(annotated_image)
        processed_count += 1
    
    video_writer.release()
    print(f"Created annotated video: {output_path}")
    print(f"Processed {processed_count} frames")
    return True

def list_available_sequences(images_dir):
    """List all available sequences in the images directory"""
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return []
    
    sequences = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    return sorted(sequences)

def main():
    parser = argparse.ArgumentParser(description='Generate annotated videos to check ground truth annotations')
    parser.add_argument('--sequence', type=str, help='Sequence name to generate video for (e.g., TS1, TS2, etc.)')
    parser.add_argument('--all', action='store_true', help='Process all available sequences')
    parser.add_argument('--list', action='store_true', help='List available sequences and exit')
    parser.add_argument('--output_dir', type=str, default='./annotated_videos', help='Output directory for videos')
    
    args = parser.parse_args()
    
    # Define paths
    images_dir = '/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo/images/val'
    labels_dir = '/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo/labels/val'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List sequences and exit if requested
    if args.list:
        sequences = list_available_sequences(images_dir)
        print("Available sequences:")
        for seq in sequences:
            print(f"  - {seq}")
        return
    
    # Determine which sequences to process
    if args.all:
        sequences = list_available_sequences(images_dir)
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
        print(f"\nProcessing sequence: {sequence}")
        success = create_annotated_video(images_dir, labels_dir, sequence, output_path)
        if success:
            successful += 1
        else:
            failed += 1
            print(f"Failed to create video for sequence: {sequence}")
    
    print(f"\nSummary: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()