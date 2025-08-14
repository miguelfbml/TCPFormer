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
from utils.learning import load_model_TCPFormer, AverageMeter, decay_lr_exponentially, sch_decay
from torch.utils.data import DataLoader
from utils.utils_H3_6 import (AccumLoss, calculate_torso_diameter_h36m, compute_pck_h36m, 
                              compute_auc_h36m, mpjpe_cal, count_param_numbers, 
                              make_root_relative_h36m, H36M_CONNECTIONS, H36M_JOINT_NAMES)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/TCPFormer_h36m_243.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='new checkpoint directory')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def train_one_epoch(args, model, train_loader, optimizer, losses):
    model.train()
    for batch_input, batch_gt in tqdm(train_loader):
        batch_size = batch_input.shape[0]
        if torch.cuda.is_available():
            batch_input, batch_gt = batch_input.cuda(), batch_gt.cuda()

        if args.flip:
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_flip = model(batch_input_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip = flip_data(predicted_3d_pos_flip)
            predicted_3d_pos = model(batch_input)
            predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2.0
        else:
            predicted_3d_pos = model(batch_input)

        optimizer.zero_grad()
        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv = loss_limb_var(predicted_3d_pos)
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a = loss_angle(predicted_3d_pos, batch_gt)
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)

        loss_total = args.lambda_3d_pos * loss_3d_pos + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_3d_velocity * loss_3d_velocity + \
                     args.lambda_lv * loss_lv + \
                     args.lambda_lg * loss_lg + \
                     args.lambda_a * loss_a + \
                     args.lambda_av * loss_av

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, datareader, device):
    print("[INFO] Evaluation with comprehensive metrics (MPJPE, P-MPJPE, PCK, AUC)")
    results_all = []
    model.eval()
    
    # Initialize comprehensive metrics tracking (same as train_3dhp.py)
    error_sum_test = AccumLoss()
    pck_results = {
        'PCK@90%_torso': 0.0, 'PCK@80%_torso': 0.0, 'PCK@70%_torso': 0.0,
        'PCK@90%_150mm': 0.0, 'PCK@80%_150mm': 0.0, 'PCK@70%_150mm': 0.0
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
            
            # Calculate comprehensive metrics for each batch
            N = predicted_3d_pos.shape[0]
            
            # Extract center frame for PCK/AUC calculations (same as train_3dhp.py)
            center_frame_idx = predicted_3d_pos.shape[1] // 2
            pred_frame = predicted_3d_pos[:, center_frame_idx]  # (N, 17, 3)
            gt_frame = y[:, center_frame_idx]  # (N, 17, 3)
            
            # Make root-relative for MPJPE calculation (Human3.6M uses joint 0 as root)
            pred_frame_rel = make_root_relative_h36m(pred_frame, root_joint_idx=0)
            gt_frame_rel = make_root_relative_h36m(gt_frame, root_joint_idx=0)
            
            # Calculate MPJPE using Human3.6M function
            joint_error_test = mpjpe_cal(pred_frame_rel, gt_frame_rel).item()
            error_sum_test.update(joint_error_test * N, N)
            
            # Calculate torso diameters for PCK (use non-root-relative poses)
            torso_diameters = calculate_torso_diameter_h36m(gt_frame)
            
            # Compute PCK for torso-based and 150mm thresholds
            batch_pck = compute_pck_h36m(pred_frame, gt_frame, torso_diameters, fixed_threshold=150.0)
            for key in pck_results:
                pck_results[key] += batch_pck[key] * N
            
            # Compute AUC
            auc = compute_auc_h36m(pred_frame, gt_frame)
            auc_sum += auc * N
            
            valid_samples += N

    results_all = np.concatenate(results_all)

    # Denormalize results for standard H36M evaluation
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

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        oc[frame_list] += 1

        # P-MPJPE
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
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
    
    # Calculate comprehensive metrics averages (same as train_3dhp.py)
    mpjpe_comprehensive = error_sum_test.avg
    for key in pck_results:
        pck_results[key] /= valid_samples
    auc_avg = auc_sum / valid_samples
    
    # Print comprehensive results
    print('\n' + '='*70)
    print('COMPREHENSIVE HUMAN3.6M EVALUATION RESULTS')
    print('='*70)
    print('Standard Human3.6M Protocol Results:')
    print(f'Protocol #1 Error (MPJPE): {e1:.2f} mm')
    print(f'Protocol #2 Error (P-MPJPE): {e2:.2f} mm')
    print(f'Acceleration error: {acceleration_error:.2f} mm/s^2')
    
    print(f'\nComprehensive Metrics (frame-wise evaluation):')
    print(f'Frame-wise MPJPE: {mpjpe_comprehensive:.2f} mm')
    print(f'PCK Results:')
    for key, value in pck_results.items():
        print(f'  {key}: {value*100:.2f}%')
    print(f'AUC: {auc_avg:.4f}')
    
    print(f'\nPer-action breakdown (Protocol #1):')
    for i, action in enumerate(action_names):
        print(f'  {action}: {final_result[i]:.2f} mm')
    
    print(f'\nJoint-wise breakdown (Protocol #1):')
    for joint_idx in range(args.num_joints):
        joint_name = H36M_JOINT_NAMES[joint_idx] if joint_idx < len(H36M_JOINT_NAMES) else f"Joint_{joint_idx}"
        print(f'  {joint_name}: {joint_errors[joint_idx]:.2f} mm')
    
    print('='*70)
    
    return e1, e2, joint_errors, acceleration_error, mpjpe_comprehensive, pck_results, auc_avg


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = load_model_TCPFormer(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    lr_decay = args.lr_decay
    lr_gamma = args.lr_gamma
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr,
                           amsgrad=True)

    epoch_start = 0
    min_mpjpe = float('inf')
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
                          project='TCPFormer',
                          resume="must",
                          settings=wandb.Settings(start_method='fork'))
        else:
            if opts.use_wandb:
                print(f"Run ID: {wandb_id}")
                wandb.init(id=wandb_id,
                          name=opts.wandb_name,
                          project='TCPFormer',
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
                mpjpe, p_mpjpe, joints_error, acceleration_error, mpjpe_comprehensive, pck_results, auc = evaluate(
                    args, model, test_loader, datareader, device)
                print(f"\nFinal Comprehensive Results Summary:")
                print(f"Protocol #1 (MPJPE): {mpjpe:.2f} mm")
                print(f"Protocol #2 (P-MPJPE): {p_mpjpe:.2f} mm")
                print(f"Frame-wise MPJPE: {mpjpe_comprehensive:.2f} mm")
                print(f"AUC: {auc:.4f}")
                print(f"Best PCK@80%_150mm: {pck_results['PCK@80%_150mm']*100:.2f}%")
                print(f"Acceleration Error: {acceleration_error:.2f} mm/s^2")
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_scale', '3d_velocity', 'lv', 'lg', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, losses)
        with torch.no_grad():
            mpjpe, p_mpjpe, joints_error, acceleration_error, mpjpe_comprehensive, pck_results, auc = evaluate(
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
                'train/loss_lv': losses['lv'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/loss_angle_velocity': losses['angle_velocity'].avg,
                'train/loss_total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/p_mpjpe': p_mpjpe,
                'eval/mpjpe_comprehensive': mpjpe_comprehensive,
                'eval/auc': auc,
                'eval/acceleration_error': acceleration_error,
                'eval/min_mpjpe': min_mpjpe,
            }
            
            # Add PCK results
            for key, value in pck_results.items():
                wandb_log_dict[f'eval/{key}'] = value
            
            # Add joint errors
            wandb_log_dict.update(joint_label_errors)
            
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
    torch.backends.cudnn.deterministic = True
    args = get_config(opts.config)

    train(args, opts)


if __name__ == "__main__":
    main()