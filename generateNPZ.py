import os
import torch
import numpy as np
from tqdm import tqdm

from utils.data import denormalize
from data.reader.motion_dataset import Fusion
from utils.tools import get_config, load_model_TCPFormer

def input_augmentation(input_2D, model, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape
    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]

    output_3D_flip = model(input_2D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip
    return input_2D, output_3D

def export_predictions(model, test_loader, n_frames, output_path):
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    out_data = {}

    for data in tqdm(test_loader, desc="Exporting predictions"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = [x.cuda() for x in [input_2D, gt_3D, batch_cam, scale, bb_box]]

        N = input_2D.size(0)
        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)

        input_2D, output_3D = input_augmentation(input_2D, model, joints_left, joints_right)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
        pad = (n_frames - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)
        pred_out[..., 14, :] = 0
        pred_out = denormalize(pred_out, seq)
        pred_out = pred_out - pred_out[..., 14:15, :]  # Root-relative prediction

        # Also get the input 2D keypoints (after denormalization if needed)
        input_2D_np = input_2D.cpu().numpy()  # shape: (N, T, 17, 2) or (N, T, 17, 3)
        # For each sequence in the batch
        for seq_cnt in range(len(seq)):
            seq_name = seq[seq_cnt]
            pred_3d_np = pred_out[seq_cnt].permute(2, 1, 0).cpu().numpy()  # (17, 1, 1) -> (17, 1, 1)
            pred_3d_np = np.squeeze(pred_3d_np, axis=(1,2)) if pred_3d_np.ndim == 3 else pred_3d_np
            input_2d_np = input_2D_np[seq_cnt, pad]  # (17, 2) or (17, 3)
            valid = np.ones(pred_3d_np.shape[0], dtype=np.int32)

            if seq_name not in out_data:
                out_data[seq_name] = {
                    'data_2d': [],
                    'data_3d': [],
                    'valid': []
                }
            out_data[seq_name]['data_2d'].append(input_2d_np)
            out_data[seq_name]['data_3d'].append(pred_3d_np)
            out_data[seq_name]['valid'].append(valid)

    # Stack arrays and save in the same structure as data_test_3dhp.npz
    for seq_name in out_data:
        out_data[seq_name]['data_2d'] = np.stack(out_data[seq_name]['data_2d'], axis=0)
        out_data[seq_name]['data_3d'] = np.stack(out_data[seq_name]['data_3d'], axis=0)
        out_data[seq_name]['valid'] = np.stack(out_data[seq_name]['valid'], axis=0)

    np.savez_compressed(output_path, data=out_data)
    print(f"✓ Saved predictions to: {output_path}")

def main():
    config_path = "configs/mpi/TCPFormer_mpi_27.yaml"  # Change as needed
    checkpoint_path = "checkpoint/best_epoch.pth.tr"   # Change as needed
    output_path = "predicted_test_3dhp.npz"

    args = get_config(config_path)
    args.n_frames = 27
    test_dataset = Fusion(args, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=4, pin_memory=True)
    model = load_model_TCPFormer(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'], strict=True)

    export_predictions(model, test_loader, args.n_frames, output_path)

if __name__ == "__main__":
    main()