import os
import torch
import numpy as np
from tqdm import tqdm

from utils.data import denormalize
from data.reader.motion_dataset import Fusion
from utils.tools import get_config
from utils.learning import load_model_TCPFormer

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

def export_predictions(model, test_loader, n_frames, original_data, output_path):
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    # Prepare output structure with empty lists for each sequence
    out_data = {}
    for seq_name in original_data.keys():
        out_data[seq_name] = {
            'data_2d': original_data[seq_name]['data_2d'],
            'data_3d': [None] * original_data[seq_name]['data_2d'].shape[0],
            'valid': original_data[seq_name]['valid']
        }

    # Track the next frame to fill for each sequence
    frame_counters = {seq_name: 0 for seq_name in original_data.keys()}

    for data in tqdm(test_loader, desc="Exporting predictions"):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = [x.cuda() for x in [input_2D, gt_3D, batch_cam, scale, bb_box]]
        input_2D = input_2D.float()

        N = input_2D.size(0)
        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)

        input_2D, output_3D = input_augmentation(input_2D, model, joints_left, joints_right)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
        pad = (n_frames - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)
        pred_out[..., 14, :] = 0
        pred_out = denormalize(pred_out.detach(), seq)
        pred_out = pred_out - pred_out[..., 14:15, :]  # Root-relative prediction

        for seq_cnt in range(len(seq)):
            seq_name = seq[seq_cnt]
            pred_3d_np = pred_out[seq_cnt].permute(2, 1, 0).cpu().numpy()
            pred_3d_np = np.reshape(pred_3d_np, (17, 3))  # Ensure shape is always (17, 3)

            # Fill the next available slot for this sequence
            slot = frame_counters[seq_name]
            out_data[seq_name]['data_3d'][slot] = pred_3d_np
            frame_counters[seq_name] += 1

        # Free CUDA memory after each batch
        del input_2D, gt_3D, batch_cam, scale, bb_box, output_3D, pred_out
        torch.cuda.empty_cache()

    # Convert lists to arrays and fill missing frames with zeros if needed
    for seq_name in out_data:
        for i in range(len(out_data[seq_name]['data_3d'])):
            if out_data[seq_name]['data_3d'][i] is None:
                out_data[seq_name]['data_3d'][i] = np.zeros((17, 3), dtype=np.float32)
            else:
                out_data[seq_name]['data_3d'][i] = np.reshape(out_data[seq_name]['data_3d'][i], (17, 3))
        out_data[seq_name]['data_3d'] = np.stack(out_data[seq_name]['data_3d'], axis=0)

    np.savez_compressed(output_path, data=out_data)
    print(f"✓ Saved predictions to: {output_path}")

def main():
    config_path = "configs/mpi/TCPFormer_mpi_27.yaml"  # Change as needed
    checkpoint_path = "checkpoint_mpi/best_epoch.pth.tr"   # Change as needed
    output_path = "predicted_test_3dhp.npz"
    original_npz_path = "data/motion3d/data_test_3dhp.npz"  # Path to original test npz

    # Load original .npz file
    original = np.load(original_npz_path, allow_pickle=True)['data'].item()

    args = get_config(config_path)
    args.n_frames = 27
    args.test_batch_size = 1  # Reduce batch size to minimize memory usage

    test_dataset = Fusion(args, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=0,
        pin_memory=False
    )
    model = load_model_TCPFormer(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'], strict=True)

    export_predictions(model, test_loader, args.n_frames, original, output_path)

if __name__ == "__main__":
    main()