import os
import numpy as np

import scipy.io as scio

def mpii_get_sequence_info(subject_id, sequence):

    switcher = {
        "1 1": [6416,25],
        "1 2": [12430,50],
        "2 1": [6502,25],
        "2 2": [6081,25],
        "3 1": [12488,50],
        "3 2": [12283,50],
        "4 1": [6171,25],
        "4 2": [6675,25],
        "5 1": [12820,50],
        "5 2": [12312,50],
        "6 1": [6188,25],
        "6 2": [6145,25],
        "7 1": [6239,25],
        "7 2": [6320,25],
        "8 1": [6468,25],
        "8 2": [6054,25],

    }
    return switcher.get(subject_id+" "+sequence)


data_path = '/nas-ctm01/datasets/public/mpi_inf_3dhp'
cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

dic_seq={}

for root, dirs, files in os.walk(data_path):
    # Skip the test data directory
    if 'mpi_inf_3dhp_test_set' in root:
        continue
        
    for file in files:
        if file.endswith("mat"):
            path = root.split("/")
            
            # Check if folders have expected format for training data
            if len(path[-2]) < 2 or len(path[-1]) < 4:
                continue
                
            # Only process training data (S1-S8, Seq1-Seq2)
            if not (path[-2].startswith('S') and path[-1].startswith('Seq')):
                continue
                
            subject = path[-2][1]
            seq = path[-1][3]
            print("loading %s %s..."%(path[-2],path[-1]))

            temp = mpii_get_sequence_info(subject, seq)
            if temp is None:
                print(f"Warning: No sequence info found for subject {subject} sequence {seq}")
                continue

            frames = temp[0]
            fps = temp[1]

            data = scio.loadmat(os.path.join(root, file))
            cameras = data['cameras'][0]
            for cam_idx in range(len(cameras)):
                assert cameras[cam_idx] == cam_idx

            data_2d = data['annot2'][cam_set]
            data_3d = data['univ_annot3'][cam_set]

            dic_cam = {}
            for cam_idx in range(len(data_2d)):
                data_2d_cam = data_2d[cam_idx][0]
                data_3d_cam = data_3d[cam_idx][0]

                data_2d_cam = data_2d_cam.reshape(data_2d_cam.shape[0], 28,2)
                data_3d_cam = data_3d_cam.reshape(data_3d_cam.shape[0], 28,3)

                data_2d_select = data_2d_cam[:frames, joint_set]
                data_3d_select = data_3d_cam[:frames, joint_set]

                dic_data = {"data_2d":data_2d_select,"data_3d":data_3d_select}

                dic_cam.update({str(cam_set[cam_idx]):dic_data})

            dic_seq.update({path[-2]+" "+path[-1]:[dic_cam, fps]})

np.savez_compressed('../motion3d/data_train_3dhp', data=dic_seq)