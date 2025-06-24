# Create a new file: inspect_cameras.py
import os
import scipy.io as scio

data_path = '/nas-ctm01/datasets/public/mpi_inf_3dhp'

# Find first .mat file to inspect
for root, dirs, files in os.walk(data_path):
    if 'mpi_inf_3dhp_test_set' in root:
        continue
        
    for file in files:
        if file.endswith("mat"):
            path = root.split("/")
            
            if len(path[-2]) < 2 or len(path[-1]) < 4:
                continue
                
            if not (path[-2].startswith('S') and path[-1].startswith('Seq')):
                continue
            
            print(f"Inspecting: {os.path.join(root, file)}")
            
            data = scio.loadmat(os.path.join(root, file))
            
            print(f"\nKeys in .mat file: {list(data.keys())}")
            
            if 'cameras' in data:
                cameras = data['cameras'][0]
                print(f"Available cameras: {cameras}")
                print(f"Number of cameras: {len(cameras)}")
                print(f"Camera indices: {list(range(len(cameras)))}")
            
            if 'annot2' in data:
                print(f"2D annotation shape: {data['annot2'].shape}")
                print(f"2D data available for cameras: {list(range(data['annot2'].shape[0]))}")
            
            if 'univ_annot3' in data:
                print(f"3D annotation shape: {data['univ_annot3'].shape}")
                print(f"3D data available for cameras: {list(range(data['univ_annot3'].shape[0]))}")
            
            # Exit after inspecting first file
            exit()