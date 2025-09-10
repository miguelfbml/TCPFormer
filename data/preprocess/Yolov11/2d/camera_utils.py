import numpy as np
import os

def parse_camera_calibration(calib_file_path):
    """Parse MPI-INF-3DHP camera calibration file"""
    calib_data = {}
    
    try:
        with open(calib_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('tc camera') or line.startswith('colorCorrection') or line.startswith('red') or line.startswith('green') or line.startswith('blue'):
                continue
                
            if line.startswith('camera'):
                parts = line.split()
                calib_data['camera_id'] = int(parts[1])
                calib_data['camera_name'] = parts[2] if len(parts) > 2 else ''
                
            elif line.startswith('frame'):
                parts = line.split()
                calib_data['frame'] = int(parts[1])
                
            elif line.startswith('sensorSize'):
                parts = line.split()
                calib_data['sensor_width'] = float(parts[1])
                calib_data['sensor_height'] = float(parts[2])
                
            elif line.startswith('focalLength'):
                parts = line.split()
                calib_data['focal_length'] = float(parts[1])
                
            elif line.startswith('pixelAspect'):
                parts = line.split()
                calib_data['pixel_aspect'] = float(parts[1])
                
            elif line.startswith('centerOffset'):
                parts = line.split()
                calib_data['center_offset_x'] = float(parts[1])
                calib_data['center_offset_y'] = float(parts[2])
                
            elif line.startswith('distortion') and not line.startswith('distortionModel'):
                parts = line.split()
                calib_data['distortion'] = [float(x) for x in parts[1:]]
                
            elif line.startswith('origin'):
                parts = line.split()
                calib_data['origin'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                
            elif line.startswith('up'):
                parts = line.split()
                calib_data['up'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                
            elif line.startswith('right'):
                parts = line.split()
                calib_data['right'] = [float(parts[1]), float(parts[2]), float(parts[3])]
        
        return calib_data
        
    except Exception as e:
        print(f"Error parsing calibration file {calib_file_path}: {e}")
        return None

def get_camera_calibration_for_sequence(seq_name):
    """Get camera calibration for a specific sequence"""
    calib_base_path = '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/test_util/camera_calibration/'
    
    # Map sequences to calibration files
    if seq_name in ['TS1', 'TS2', 'TS3', 'TS4']:
        calib_file = os.path.join(calib_base_path, 'ts1-4cameras.calib')
    elif seq_name in ['TS5', 'TS6']:
        calib_file = os.path.join(calib_base_path, 'ts5-6cameras.calib')
    else:
        print(f"Unknown sequence: {seq_name}")
        return None
    
    return parse_camera_calibration(calib_file)

def compute_camera_intrinsics(calib_data, image_width, image_height):
    """Compute camera intrinsic matrix from calibration data"""
    # Convert focal length from mm to pixels
    sensor_width_mm = calib_data['sensor_width']
    sensor_height_mm = calib_data['sensor_height']
    focal_length_mm = calib_data['focal_length']
    
    # Focal length in pixels
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = (focal_length_mm / sensor_height_mm) * image_height * calib_data.get('pixel_aspect', 1.0)
    
    # Principal point (image center + offset)
    cx = image_width / 2 + (calib_data.get('center_offset_x', 0) / sensor_width_mm) * image_width
    cy = image_height / 2 + (calib_data.get('center_offset_y', 0) / sensor_height_mm) * image_height
    
    # Camera intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K, fx, fy, cx, cy

def pixel_to_camera_coordinates(pixel_coords, calib_data, image_width, image_height, depth=1000):
    """
    Convert pixel coordinates to camera coordinate system
    
    Args:
        pixel_coords: (N, 2) array of pixel coordinates
        calib_data: Camera calibration data
        image_width: Image width in pixels
        image_height: Image height in pixels
        depth: Assumed depth (default 1000mm, adjust as needed)
    
    Returns:
        (N, 2) array of camera coordinates
    """
    K, fx, fy, cx, cy = compute_camera_intrinsics(calib_data, image_width, image_height)
    
    # Convert to camera coordinates
    camera_coords = np.zeros_like(pixel_coords)
    camera_coords[:, 0] = (pixel_coords[:, 0] - cx) * depth / fx
    camera_coords[:, 1] = (pixel_coords[:, 1] - cy) * depth / fy
    
    return camera_coords

def test_coordinate_conversion():
    """Test the coordinate conversion with sample data"""
    print("=== TESTING COORDINATE CONVERSION ===")
    
    for seq_name in ['TS1', 'TS5']:
        print(f"\nTesting {seq_name}:")
        calib_data = get_camera_calibration_for_sequence(seq_name)
        
        if calib_data:
            print(f"  Calibration loaded successfully")
            print(f"  Focal length: {calib_data['focal_length']:.2f}mm")
            print(f"  Sensor size: {calib_data['sensor_width']:.1f}x{calib_data['sensor_height']:.1f}mm")
            
            # Get image dimensions
            if seq_name in ['TS1', 'TS2', 'TS3', 'TS4']:
                img_w, img_h = 2048, 2048
            else:
                img_w, img_h = 1920, 1080
            
            K, fx, fy, cx, cy = compute_camera_intrinsics(calib_data, img_w, img_h)
            print(f"  Computed intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            
            # Test with center pixel
            test_pixel = np.array([[img_w/2, img_h/2]])
            camera_coord = pixel_to_camera_coordinates(test_pixel, calib_data, img_w, img_h)
            print(f"  Center pixel {test_pixel[0]} -> Camera coord {camera_coord[0]}")
        else:
            print(f"  Failed to load calibration")

if __name__ == "__main__":
    test_coordinate_conversion()