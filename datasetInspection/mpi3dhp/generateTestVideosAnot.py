'''
python generateTestVideos.py
python generateTestVideos.py --fps 30 --output-dir ../../test_videos
'''

import os
import cv2
import glob
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

def create_video_from_images(image_folder, output_path, fps=30, resize=None):
    """
    Convert a sequence of JPG images to MP4 video
    
    Args:
        image_folder: Path to folder containing JPG images
        output_path: Path for output MP4 file
        fps: Frames per second for output video
        resize: Tuple (width, height) to resize frames, None to keep original size
    """
    
    # Get all jpg files and sort them
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
    
    if not image_files:
        print(f"No JPG files found in {image_folder}")
        return False
    
    # Sort files numerically (important for correct frame order)
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]) 
                     if '_' in os.path.basename(x) else int(os.path.splitext(os.path.basename(x))[0]))
    
    print(f"Found {len(image_files)} images in {image_folder}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Could not read first image: {image_files[0]}")
        return False
    
    height, width, channels = first_frame.shape
    
    # Resize if specified
    if resize:
        width, height = resize
        first_frame = cv2.resize(first_frame, (width, height))
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False
    
    # Process each image
    for i, image_file in enumerate(tqdm(image_files, desc="Processing frames")):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        
        # Resize if specified
        if resize:
            frame = cv2.resize(frame, (width, height))
        
        # Write frame to video
        video_writer.write(frame)
    
    # Release video writer
    video_writer.release()
    
    print(f"Video saved to: {output_path}")
    return True

def process_mpi_test_sequences(base_path, output_dir, fps=30, resize=None):
    """
    Process all MPI-INF-3DHP test sequences (TS1-TS6)
    
    Args:
        base_path: Base path to mpi_inf_3dhp_test_set
        output_dir: Directory to save MP4 files
        fps: Frames per second for output videos
        resize: Tuple (width, height) to resize frames, None to keep original size
    """
    
    sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for seq in sequences:
        print(f"\n{'='*50}")
        print(f"Processing sequence: {seq}")
        print(f"{'='*50}")
        
        # Input folder path
        image_folder = os.path.join(base_path, seq, 'imageSequence')
        
        # Output video path
        output_path = os.path.join(output_dir, f"{seq}.mp4")
        
        # Check if input folder exists
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue
        
        # Skip if output already exists (optional)
        if os.path.exists(output_path):
            response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print(f"Skipping {seq}")
                continue
        
        # Create video
        if create_video_from_images(image_folder, output_path, fps, resize):
            success_count += 1
            
            # Print video info
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                print(f"✓ Created {seq}.mp4: {frame_count} frames, {duration:.2f}s duration")
                cap.release()
        else:
            print(f"✗ Failed to create video for {seq}")
    
    print(f"\n{'='*50}")
    print(f"Summary: Successfully created {success_count}/{len(sequences)} videos")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Convert MPI-INF-3DHP test sequence images to MP4 videos")
    parser.add_argument('--base-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
                       help='Base path to mpi_inf_3dhp_test_set directory')
    parser.add_argument('--output-dir', type=str, default='../videos_test_sequences',
                       help='Output directory for MP4 files')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for output videos (default: 25)')
    parser.add_argument('--resize', type=str, default=None,
                       help='Resize frames to WxH (e.g., "1920x1080"), leave empty for original size')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Process specific sequence only (TS1, TS2, etc.)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview first few frames before processing')
    
    args = parser.parse_args()
    
    # Parse resize parameter
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize = (width, height)
            print(f"Will resize frames to {width}x{height}")
        except ValueError:
            print("Invalid resize format. Use WIDTHxHEIGHT (e.g., 1920x1080)")
            return
    
    # Check if base path exists
    if not os.path.exists(args.base_path):
        print(f"Error: Base path does not exist: {args.base_path}")
        return
    
    # Preview mode
    if args.preview:
        print("Preview mode: Showing first frame from each sequence...")
        sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
        for seq in sequences:
            image_folder = os.path.join(args.base_path, seq, 'imageSequence')
            if os.path.exists(image_folder):
                images = glob.glob(os.path.join(image_folder, "*.jpg"))
                if images:
                    images.sort()
                    first_image = cv2.imread(images[0])
                    if first_image is not None:
                        h, w = first_image.shape[:2]
                        print(f"{seq}: {len(images)} images, first frame: {w}x{h}")
                    else:
                        print(f"{seq}: Could not read first image")
                else:
                    print(f"{seq}: No images found")
            else:
                print(f"{seq}: Directory not found")
        return
    
    # Process single sequence or all sequences
    if args.sequence:
        if args.sequence not in ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']:
            print(f"Invalid sequence: {args.sequence}. Must be one of: TS1, TS2, TS3, TS4, TS5, TS6")
            return
        
        print(f"Processing single sequence: {args.sequence}")
        image_folder = os.path.join(args.base_path, args.sequence, 'imageSequence')
        output_path = os.path.join(args.output_dir, f"{args.sequence}.mp4")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        if create_video_from_images(image_folder, output_path, args.fps, resize):
            print(f"✓ Successfully created {args.sequence}.mp4")
        else:
            print(f"✗ Failed to create video for {args.sequence}")
    else:
        # Process all sequences
        process_mpi_test_sequences(args.base_path, args.output_dir, args.fps, resize)

if __name__ == "__main__":
    main()