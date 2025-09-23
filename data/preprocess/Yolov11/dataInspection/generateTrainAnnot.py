'''
python generateTrainAnnot.py --S1 --Seq1 --cam 4


python generateTrainAnnot.py --S1 --Seq1 --Seq2 --all_cams

'''


import os
import cv2
import argparse
import numpy as np
import glob
from pathlib import Path

def load_train_3d_data_from_dataset(subject_num, sequence_num):
    """Load training data for a specific subject and sequence from .npz file"""
    train_data_paths = [
        '../../motion3d/data_train_3dhp.npz',
        '../../../motion3d/data_train_3dhp.npz',
        '../../../../motion3d/data_train_3dhp.npz',
        'data_train_3dhp.npz'
    ]
    
    sequence_key = f"S{subject_num} Seq{sequence_num}"
    
    for data_path in train_data_paths:
        if os.path.exists(data_path):
            print(f"✓ Loading training data from: {os.path.abspath(data_path)}")
            data = np.load(data_path, allow_pickle=True)['data'].item()
            
            if sequence_key in data:
                seq_data = data[sequence_key]
                print(f"✓ Found sequence: {sequence_key}")
                return seq_data
            else:
                print(f"❌ Sequence {sequence_key} not found in training data")
                print(f"Available sequences: {list(data.keys())}")
                return None
    
    print("❌ Training dataset not found. Tried:")
    for path in train_data_paths:
        print(f"  - {path}")
    return None

def get_image_files_list(subject_num, sequence_num, camera_num=None):
    """Get the list of image files for a specific subject, sequence, and camera from original dataset"""
    original_image_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp',
        '../motion3d/mpi_inf_3dhp',
        '../../motion3d/mpi_inf_3dhp',
        './mpi_inf_3dhp'
    ]
    
    all_image_files = []
    cameras_found = set()
    
    available_cameras = [0, 1, 2, 4, 5, 6, 7, 8]
    
    if camera_num is not None and camera_num not in available_cameras:
        print(f"❌ Invalid camera number: {camera_num}. Available cameras: {available_cameras}")
        return [], []
    
    for base_path in original_image_paths:
        subject_path = os.path.join(base_path, f"S{subject_num}", f"Seq{sequence_num}", "imageFrames")
        
        if os.path.exists(subject_path):
            print(f"✓ Found subject path: {subject_path}")
            
            search_cameras = [camera_num] if camera_num is not None else available_cameras
            
            for cam in search_cameras:
                camera_path = os.path.join(subject_path, f"video_{cam}")
                
                if os.path.exists(camera_path):
                    image_files = glob.glob(os.path.join(camera_path, "*.jpg"))
                    image_files.extend(glob.glob(os.path.join(camera_path, "*.JPG")))
                    image_files.extend(glob.glob(os.path.join(camera_path, "*.png")))
                    
                    if image_files:
                        image_files.sort()
                        all_image_files.extend(image_files)
                        cameras_found.add(cam)
                        print(f"✓ Found {len(image_files)} images for cam{cam} in: {camera_path}")
    
    if all_image_files:
        all_image_files.sort()
        if camera_num is not None:
            cam_info = f"cam{camera_num}"
        else:
            cam_info = f"cams[{','.join(map(str, sorted(cameras_found)))}]"
        print(f"✓ Total found {len(all_image_files)} images for S{subject_num}_Seq{sequence_num}_{cam_info}")
        return all_image_files, sorted(cameras_found)
    
    print(f"✗ No images found for S{subject_num}_Seq{sequence_num}")
    return [], []

def load_single_frame(image_path):
    """Load a single frame from file path"""
    frame = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    return frame, filename

def extract_camera_from_filename(filename):
    """Extract camera number from filename like 'frame_000001.jpg' in video_X folder"""
    return None

def get_frame_index_from_filename(filename):
    """Extract frame index from filename like 'frame_000001.jpg'"""
    try:
        base_name = os.path.splitext(filename)[0]
        if 'frame_' in base_name:
            frame_str = base_name.split('frame_')[-1]
            return int(frame_str)
        return None
    except:
        return None

CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand', 'RShoulder', 
    'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle', 
    'Sacrum', 'Spine', 'Neck'
]

def draw_annotations_from_npz(image, poses_2d, poses_2d_conf, frame_idx, img_width, img_height):
    """Draw keypoints from .npz data on image with confidence values"""
    if frame_idx >= len(poses_2d):
        return image, 0
    
    keypoints_2d = poses_2d[frame_idx]
    
    if keypoints_2d.shape[0] != 17:
        print(f"Warning: Expected 17 keypoints, got {keypoints_2d.shape[0]}")
        return image, 0
    
    if poses_2d_conf is not None and frame_idx < len(poses_2d_conf):
        confidences = poses_2d_conf[frame_idx]
    else:
        confidences = np.ones(17)
    
    pixel_keypoints = []
    for i, (x, y) in enumerate(keypoints_2d):
        conf = confidences[i] if i < len(confidences) else 1.0
        if x > 0 and y > 0:
            pixel_keypoints.append((int(x), int(y), conf))
        else:
            pixel_keypoints.append((0, 0, 0))
    
    for connection in CONNECTIONS_2D:
        joint1_idx, joint2_idx = connection
        if (joint1_idx < len(pixel_keypoints) and joint2_idx < len(pixel_keypoints) and
            pixel_keypoints[joint1_idx][2] > 0 and pixel_keypoints[joint2_idx][2] > 0):
            pt1 = (pixel_keypoints[joint1_idx][0], pixel_keypoints[joint1_idx][1])
            pt2 = (pixel_keypoints[joint2_idx][0], pixel_keypoints[joint2_idx][1])
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)
    
    visible_keypoints = 0
    for i, (px, py, confidence) in enumerate(pixel_keypoints):
        if confidence > 0:
            visible_keypoints += 1
            if i == 0:
                color = (0, 0, 255)
            elif i in [2, 3, 4, 5, 6, 7]:
                color = (255, 255, 0)
            elif i in [8, 9, 10, 11, 12, 13]:
                color = (0, 255, 255)
            else:
                color = (255, 0, 255)
            
            radius = max(2, int(4 * confidence))
            cv2.circle(image, (px, py), radius, color, -1)
            cv2.putText(image, f"{i}", (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(image, f"{confidence:.2f}", (px+5, py+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1)
    
    return image, visible_keypoints

def create_annotated_video(subject_num, sequence_num, camera_num, output_path, max_frames=None):
    """Create video with annotations from .npz file for a specific subject, sequence, and camera"""
    if camera_num is not None:
        sequence_name = f"S{subject_num}_Seq{sequence_num}_cam{camera_num}"
    else:
        sequence_name = f"S{subject_num}_Seq{sequence_num}_all_cams"
    
    print(f"Loading annotations and images for: {sequence_name}")
    
    seq_data = load_train_3d_data_from_dataset(subject_num, sequence_num)
    if seq_data is None:
        print(f"Failed to load pose data for S{subject_num} Seq{sequence_num}")
        return False
    
    if camera_num is not None:
        camera_key = str(camera_num)
        if camera_key not in seq_data[0]:
            print(f"Camera {camera_num} not found in pose data. Available cameras: {list(seq_data[0].keys())}")
            return False
        
        camera_data = seq_data[0][camera_key]
        poses_2d = camera_data['data_2d']
        
        # Convert poses_2d to NumPy array if it's a list
        if isinstance(poses_2d, list):
            poses_2d = np.array(poses_2d)
        
        poses_2d_conf = None
        if 'data_2d_conf' in camera_data:
            poses_2d_conf = camera_data['data_2d_conf']
        elif 'conf_2d' in camera_data:
            poses_2d_conf = camera_data['conf_2d']
        elif 'confidences' in camera_data:
            poses_2d_conf = camera_data['confidences']
        
        # Convert poses_2d_conf to NumPy array if it's a list
        if isinstance(poses_2d_conf, list):
            poses_2d_conf = np.array(poses_2d_conf)
        
        print(f"✓ Loaded 2D poses for cam{camera_num}: {poses_2d.shape}")
        if poses_2d_conf is not None:
            print(f"✓ Loaded confidences for cam{camera_num}: {poses_2d_conf.shape}")
            # Print keypoint confidences
            print(f"\nKeypoint Confidences for {sequence_name}:")
            for frame_idx in range(poses_2d_conf.shape[0]):
                print(f"Frame {frame_idx}:")
                confidences = poses_2d_conf[frame_idx]
                for kp_idx, conf in enumerate(confidences):
                    print(f"  Keypoint {kp_idx} ({JOINT_NAMES[kp_idx]}): Confidence = {conf:.4f}")
        else:
            print(f"No confidence data found for cam{camera_num}. Available keys: {list(camera_data.keys())}")
    else:
        camera_data = seq_data[0]['0']
        poses_2d = camera_data['data_2d']
        if isinstance(poses_2d, list):
            poses_2d = np.array(poses_2d)
        
        poses_2d_conf = camera_data.get('data_2d_conf', camera_data.get('conf_2d', camera_data.get('confidences', None)))
        if isinstance(poses_2d_conf, list):
            poses_2d_conf = np.array(poses_2d_conf)
        
        print(f"✓ Loaded 2D poses (using cam0 as reference): {poses_2d.shape}")
        if poses_2d_conf is not None:
            print(f"✓ Loaded confidences (using cam0 as reference): {poses_2d_conf.shape}")
            # Print keypoint confidences
            print(f"\nKeypoint Confidences for {sequence_name}:")
            for frame_idx in range(poses_2d_conf.shape[0]):
                print(f"Frame {frame_idx}:")
                confidences = poses_2d_conf[frame_idx]
                for kp_idx, conf in enumerate(confidences):
                    print(f"  Keypoint {kp_idx} ({JOINT_NAMES[kp_idx]}): Confidence = {conf:.4f}")
    
    result = get_image_files_list(subject_num, sequence_num, camera_num)
    if len(result) == 2:
        image_files, cameras_found = result
    else:
        image_files, cameras_found = result, []
    
    if not image_files:
        print(f"No image files found for sequence: {sequence_name}")
        return False
    
    if max_frames is not None:
        image_files = image_files[:max_frames]
    
    print(f"Found {len(image_files)} frames to process")
    
    first_frame, _ = load_single_frame(image_files[0])
    if first_frame is None:
        print(f"Could not load first frame: {image_files[0]}")
        return False
    
    height, width, _ = first_frame.shape
    print(f"Video dimensions: {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    processed_count = 0
    frames_with_annotations = 0
    total_keypoints_found = 0
    total_confidence = 0.0
    
    print(f"Processing {len(image_files)} frames...")
    
    for i, image_path in enumerate(image_files):
        frame, filename = load_single_frame(image_path)
        
        if frame is None:
            continue
        
        frame_idx = get_frame_index_from_filename(filename)
        if frame_idx is None:
            frame_idx = i
        
        if frame_idx < len(poses_2d):
            annotated_image, visible_keypoints = draw_annotations_from_npz(
                frame, poses_2d, poses_2d_conf, frame_idx, width, height)
            total_keypoints_found += visible_keypoints
            
            if poses_2d_conf is not None and frame_idx < len(poses_2d_conf):
                frame_confidences = poses_2d_conf[frame_idx]
                valid_confidences = frame_confidences[frame_confidences > 0]
                frame_avg_conf = np.mean(valid_confidences) if len(valid_confidences) > 0 else 0.0
                total_confidence += frame_avg_conf
            else:
                frame_avg_conf = 1.0
                total_confidence += frame_avg_conf
            
            if visible_keypoints > 0:
                frames_with_annotations += 1
        else:
            annotated_image = frame
            visible_keypoints = 0
            frame_avg_conf = 0.0
        
        frame_info = f"Frame: {filename} (idx:{frame_idx}) | Keypoints: {visible_keypoints} | Avg Conf: {frame_avg_conf:.3f}"
        cv2.putText(annotated_image, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        seq_info = f"Sequence: {sequence_name} | Progress: {i + 1}/{len(image_files)}"
        cv2.putText(annotated_image, seq_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if camera_num is None and cameras_found:
            cam_info = f"Cameras found: {cameras_found}"
            cv2.putText(annotated_image, cam_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y = 120
        else:
            legend_y = 90
        
        legend_text = "Legend: NPZ GT Data | Blue=Skeleton, Red=Head, Cyan=Arms, Yellow=Legs, Magenta=Torso"
        cv2.putText(annotated_image, legend_text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        conf_text = "Confidence values shown below each keypoint | Circle size = confidence"
        cv2.putText(annotated_image, conf_text, (10, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        video_writer.write(annotated_image)
        processed_count += 1
        
        del frame, annotated_image
        
        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            avg_conf = total_confidence / (i + 1) if (i + 1) > 0 else 0.0
            print(f"Processed {i + 1}/{len(image_files)} frames | "
                  f"Frames with keypoints: {frames_with_annotations} | "
                  f"Total keypoints: {total_keypoints_found} | "
                  f"Avg confidence: {avg_conf:.3f}")
            
            import gc
            gc.collect()
    
    video_writer.release()
    avg_confidence = total_confidence / processed_count if processed_count > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Created annotated video: {output_path}")
    print(f"Processed {processed_count} frames")
    print(f"Frames with annotations: {frames_with_annotations}/{processed_count}")
    print(f"Total visible keypoints found: {total_keypoints_found}")
    print(f"Average confidence across all frames: {avg_confidence:.3f}")
    print(f"Source: Ground truth from .npz training data")
    if cameras_found:
        print(f"Cameras included: {cameras_found}")
    print(f"{'='*60}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate annotated videos using ground truth annotations from .npz training data')
    
    for i in range(1, 9):
        parser.add_argument(f'--S{i}', action='store_true', help=f'Process subject S{i}')
    
    parser.add_argument('--Seq1', action='store_true', help='Process sequence 1')
    parser.add_argument('--Seq2', action='store_true', help='Process sequence 2')
    
    parser.add_argument('--cam', type=int, choices=[0, 1, 2, 4, 5, 6, 7, 8], 
                       help='Specific camera to process (0, 1, 2, 4, 5, 6, 7, 8). Note: camera 3 is not available in MPI-INF-3DHP')
    parser.add_argument('--all_cams', action='store_true', 
                       help='Process all available cameras for selected subjects/sequences')
    
    parser.add_argument('--all', action='store_true', help='Process all available subjects, sequences, and cameras (S1-S8, Seq1-Seq2, all cams)')
    parser.add_argument('--output_dir', type=str, default='./annotated_videos_train_npz', help='Output directory for videos')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process per sequence')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    subjects = []
    if args.all:
        subjects = list(range(1, 9))
    else:
        for i in range(1, 9):
            if getattr(args, f'S{i}'):
                subjects.append(i)
    
    sequences = []
    if args.all:
        sequences = [1, 2]
    else:
        if args.Seq1:
            sequences.append(1)
        if args.Seq2:
            sequences.append(2)
    
    cameras = []
    if args.all or args.all_cams:
        cameras = [0, 1, 2, 4, 5, 6, 7, 8]
    elif args.cam is not None:
        cameras = [args.cam]
    else:
        cameras = [0]
    
    if not subjects:
        print("Error: Must specify at least one subject (--S1 to --S8) or --all")
        parser.print_help()
        return
    
    if not sequences:
        print("Error: Must specify at least one sequence (--Seq1, --Seq2) or --all")
        parser.print_help()
        return
    
    print(f"Processing subjects: {['S' + str(s) for s in subjects]}")
    print(f"Processing sequences: {['Seq' + str(s) for s in sequences]}")
    print(f"Processing cameras: {cameras}")
    print(f"Source: Ground truth annotations from .npz training data")
    
    successful = 0
    failed = 0
    
    for subject_num in subjects:
        for sequence_num in sequences:
            for camera_num in cameras:
                sequence_name = f"S{subject_num}_Seq{sequence_num}_cam{camera_num}"
                output_filename = f"{sequence_name}_gt_annotated.mp4"
                output_path = os.path.join(args.output_dir, output_filename)
                
                print(f"\n{'='*50}")
                print(f"Processing: {sequence_name}")
                print(f"{'='*50}")
                
                success = create_annotated_video(subject_num, sequence_num, camera_num, output_path, args.max_frames)
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"Failed to create video for: {sequence_name}")
    
    print(f"\n{'='*50}")
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"Videos show ground truth keypoints from .npz training data")
    print(f"Available cameras in MPI-INF-3DHP: 0, 1, 2, 4, 5, 6, 7, 8 (camera 3 is missing)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()