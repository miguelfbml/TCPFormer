import argparse
import os

import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from tqdm import tqdm

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity, miloss
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

from utils.learning import AverageMeter, decay_lr_exponentially, load_model_TCPFormer
from utils.tools import count_param_numbers
from utils.data import Augmenter2D
from utils.utils_3dhp import AccumLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/TCPFormer_h36m_243.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', default=True, action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()        
    optimizer.zero_grad()
    accumulation_steps = 1
    i = 0
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0

        pred = model(x)

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av 
                    # args.lambda_mi * loss_mi

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total = loss_total / accumulation_steps
        loss_total.backward()
        if(i+1)%accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        i += 1

def evaluate(args, model, test_loader, datareader, device):
    print("[INFO] Evaluation with comprehensive metrics (MPJPE, P-MPJPE, PCK, AUC)")
    results_all = []
    model.eval()
    
    # Initialize comprehensive metrics tracking - same as train_3dhp.py
    pck_results = {
        'PCK@10%_torso': 0.0, 'PCK@20%_torso': 0.0, 'PCK@30%_torso': 0.0, 'PCK@100%_torso': 0.0,
        'PCK@10%_150mm': 0.0, 'PCK@20%_150mm': 0.0, 'PCK@30%_150mm': 0.0, 'PCK@100%_150mm': 0.0
    }
    auc_sum = 0.0
    valid_samples = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

    # Use the exact same denormalization process as MPJPE calculation
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, args.num_joints))
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    
    # Collect data for PCK calculation using ABSOLUTE coordinates (NOT root-relative)
    pck_pred_frames = []
    pck_gt_frames = []
    
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Store ABSOLUTE (denormalized) data for PCK calculation (center frame only)
        center_frame_idx = pred.shape[0] // 2
        pck_pred_frames.append(pred[center_frame_idx])  # Shape: (17, 3) - ABSOLUTE coords
        pck_gt_frames.append(gt[center_frame_idx])      # Shape: (17, 3) - ABSOLUTE coords

        # Root-relative Errors (for MPJPE calculation only)
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    
    # Calculate PCK using ABSOLUTE coordinates - same logic as train_3dhp.py
    if pck_pred_frames:
        pck_pred_frames = np.array(pck_pred_frames)  # Shape: (N, 17, 3) - ABSOLUTE
        pck_gt_frames = np.array(pck_gt_frames)      # Shape: (N, 17, 3) - ABSOLUTE
        
        # Convert to torch tensors for compatibility with train_3dhp.py functions
        pck_pred_tensor = torch.from_numpy(pck_pred_frames).float()
        pck_gt_tensor = torch.from_numpy(pck_gt_frames).float()
        
        # Calculate torso diameters for PCK using ABSOLUTE coordinates
        torso_diameters = calculate_torso_diameter_h36m(pck_gt_tensor)
        
        # Compute PCK using same logic as train_3dhp.py
        batch_pck = compute_pck_h36m(pck_pred_tensor, pck_gt_tensor, torso_diameters, fixed_threshold=150.0)
        for key in pck_results:
            if key in batch_pck:
                pck_results[key] = batch_pck[key]
        
        # Compute AUC using same logic as train_3dhp.py
        auc_avg = compute_auc_h36m(pck_pred_tensor, pck_gt_tensor)
        valid_samples = len(pck_pred_frames)
    else:
        auc_avg = 0.0
    
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            results_accelaration[action].append(acc_err)
            for joint_idx in range(args.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
    final_result_procrustes = []
    final_result_joints = [[] for _ in range(args.num_joints)]
    final_result_acceleration = []
    final_result = []

    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
        for joint_idx in range(args.num_joints):
            final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))
        print(action,"p1:",np.mean(results[action]),"   p2:",np.mean(results_procrustes[action]))

    joint_errors = []
    for joint_idx in range(args.num_joints):
        joint_errors.append(
            np.mean(np.array(final_result_joints[joint_idx]))
        )
    joint_errors = np.array(joint_errors)
    e1 = np.mean(np.array(final_result))
    assert round(e1, 4) == round(np.mean(joint_errors), 4), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    acceleration_error = np.mean(np.array(final_result_acceleration))
    e2 = np.mean(np.array(final_result_procrustes))
    
    # Print comprehensive results - same format as train_3dhp.py
    print('\n' + '='*70)
    print('COMPREHENSIVE HUMAN3.6M EVALUATION RESULTS')
    print('='*70)
    print('Standard Human3.6M Protocol Results:')
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Acceleration error:', acceleration_error, 'mm/s^2')
    
    print('\nComprehensive Metrics:')
    # Print in ascending order like train_3dhp.py
    print(f'PCK@10%_torso: {pck_results["PCK@10%_torso"]*100:.2f}%')
    print(f'PCK@20%_torso: {pck_results["PCK@20%_torso"]*100:.2f}%')
    print(f'PCK@30%_torso: {pck_results["PCK@30%_torso"]*100:.2f}%')
    print(f'PCK@100%_torso: {pck_results["PCK@100%_torso"]*100:.2f}%')
    print(f'PCK@10%_150mm: {pck_results["PCK@10%_150mm"]*100:.2f}%')
    print(f'PCK@20%_150mm: {pck_results["PCK@20%_150mm"]*100:.2f}%')
    print(f'PCK@30%_150mm: {pck_results["PCK@30%_150mm"]*100:.2f}%')
    print(f'PCK@100%_150mm: {pck_results["PCK@100%_150mm"]*100:.2f}%')
    print(f'AUC: {auc_avg:.4f}')
    
    print('\nPer-action breakdown (Protocol #1):')
    for i, action in enumerate(action_names):
        print(f'  {action}: {final_result[i]:.2f} mm')
    
    print('='*70)
    
    return e1, e2, joint_errors, acceleration_error, pck_results, auc_avg


# Add Human3.6M-specific utility functions - same logic as train_3dhp.py
def calculate_torso_diameter_h36m(gt_3d, left_shoulder_idx=11, right_shoulder_idx=14, 
                                  left_hip_idx=4, right_hip_idx=1):
    """
    Calculate torso diameter using shoulder and hip distances (same as MPI-INF-3DHP standard)
    Human3.6M joint indices:
    0: Hip (root), 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
    7: Spine, 8: Thorax, 9: Neck, 10: Head, 11: LShoulder, 12: LElbow, 13: LWrist,
    14: RShoulder, 15: RElbow, 16: RWrist
    """
    if gt_3d.dim() == 4:
        gt_3d = gt_3d[:, 0]  # Take first frame if temporal
    N = gt_3d.shape[0]
    torso_diameters = torch.zeros(N, device=gt_3d.device)
    
    for i in range(N):
        # Shoulder distance
        left_shoulder = gt_3d[i, left_shoulder_idx]   # (3,)
        right_shoulder = gt_3d[i, right_shoulder_idx] # (3,)
        shoulder_dist = torch.norm(left_shoulder - right_shoulder)
        
        # Hip distance  
        left_hip = gt_3d[i, left_hip_idx]   # (3,)
        right_hip = gt_3d[i, right_hip_idx] # (3,)
        hip_dist = torch.norm(left_hip - right_hip)
        
        # Average of shoulder and hip distance as torso diameter
        torso_diameters[i] = (shoulder_dist + hip_dist) / 2.0
    
    return torso_diameters

def compute_pck_h36m(pred, gt, torso_diameters, fixed_threshold=150.0, pck_thresholds=[0.1, 0.2, 0.3, 1.0]):
    """
    Compute traditional PCK metric: percentage of keypoints within threshold
    Same logic as train_3dhp.py but adapted for Human3.6M
    """
    N, J, _ = pred.shape
    joint_errors = torch.norm(pred - gt, dim=-1)  # (N, J) - per-joint distances
    
    pck_results = {}
    
    # Torso-based PCK: percentage of keypoints within X% of torso diameter
    for thresh_pct in pck_thresholds:
        thresh_pct_int = int(thresh_pct * 100)
        
        # Calculate threshold for each sample: thresh_pct * torso_diameter
        thresholds = torso_diameters.unsqueeze(1) * thresh_pct  # (N, 1)
        
        # Check which keypoints are within threshold
        correct_keypoints = (joint_errors <= thresholds).float()  # (N, J)
        
        # PCK = percentage of ALL keypoints (across all samples and joints) that are correct
        pck = correct_keypoints.mean().item()
        pck_results[f'PCK@{thresh_pct_int}%_torso'] = pck
    
    # Fixed threshold PCK: percentage of keypoints within X% of fixed threshold (150mm)
    for thresh_pct in pck_thresholds:
        thresh_pct_int = int(thresh_pct * 100)
        
        # Calculate threshold: thresh_pct * 150mm (e.g., 10% of 150mm = 15mm, 100% of 150mm = 150mm)
        threshold = fixed_threshold * thresh_pct
        
        # Check which keypoints are within threshold
        correct_keypoints = (joint_errors <= threshold).float()  # (N, J)
        
        # PCK = percentage of ALL keypoints (across all samples and joints) that are correct
        pck = correct_keypoints.mean().item()
        pck_results[f'PCK@{thresh_pct_int}%_150mm'] = pck
    
    return pck_results

def compute_auc_h36m(pred, gt, max_threshold=150, num_steps=50):
    """
    Compute AUC by evaluating PCK over a range of thresholds.
    Same logic as train_3dhp.py
    """
    N, J, _ = pred.shape
    thresholds = np.linspace(0, max_threshold, num_steps)
    pck_values = []
    
    for thresh in thresholds:
        correct_keypoints = (torch.norm(pred - gt, dim=-1) <= thresh).float()
        pck = correct_keypoints.mean().item()  # Average across all keypoints
        pck_values.append(pck)
    
    auc = np.trapz(pck_values, thresholds) / max_threshold
    return auc


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': 8,
        # 'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root='data/motion3d', dt_file=args.dt_file)  # Used for H36m evaluation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_TCPFormer(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        project='MemoryInducedTransformer',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        name=opts.wandb_name,
                        project='MemoryInducedTransformer',
                        settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            with torch.no_grad():
                # Run comprehensive evaluation
                mpjpe, p_mpjpe, joints_error, acceleration_error, pck_results, auc = evaluate(
                    args, model, test_loader, datareader, device)
                print(f"\nFinal Comprehensive Results Summary:")
                print(f"Protocol #1 (MPJPE): {mpjpe:.2f} mm")
                print(f"Protocol #2 (P-MPJPE): {p_mpjpe:.2f} mm")
                print(f"AUC: {auc:.4f}")
                print(f"PCK@10%_torso: {pck_results['PCK@10%_torso']*100:.2f}%")
                print(f"PCK@20%_torso: {pck_results['PCK@20%_torso']*100:.2f}%")
                print(f"PCK@30%_torso: {pck_results['PCK@30%_torso']*100:.2f}%")
                print(f"PCK@100%_torso: {pck_results['PCK@100%_torso']*100:.2f}%")
                print(f"PCK@10%_150mm: {pck_results['PCK@10%_150mm']*100:.2f}%")
                print(f"PCK@20%_150mm: {pck_results['PCK@20%_150mm']*100:.2f}%")
                print(f"PCK@30%_150mm: {pck_results['PCK@30%_150mm']*100:.2f}%")
                print(f"PCK@100%_150mm: {pck_results['PCK@100%_150mm']*100:.2f}%")
                print(f"Acceleration Error: {acceleration_error:.2f} mm/s^2")
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)

        mpjpe, p_mpjpe, joints_error, acceleration_error, pck_results, auc = evaluate(
            args, model, test_loader, datareader, device)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
            print('save the best checkpoint at : {} !'.format(epoch))
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]
        if opts.use_wandb:
            wandb_log_dict = {
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/acceleration_error': acceleration_error,
                'eval/min_mpjpe': min_mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval/auc': auc,
                'eval_additional/upper_body_error': np.mean(joints_error[H36M_UPPER_BODY_JOINTS]),
                'eval_additional/lower_body_error': np.mean(joints_error[H36M_LOWER_BODY_JOINTS]),
                'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                **joint_label_errors
            }
            
            # Add all PCK results to WandB
            for key, value in pck_results.items():
                wandb_log_dict[f'eval/{key}'] = value
            
            wandb.log(wandb_log_dict, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    if opts.use_wandb:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)
    
    train(args, opts)


if __name__ == '__main__':
    main()