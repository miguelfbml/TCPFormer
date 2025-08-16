import os
from torch.autograd import Variable
import torch
import numpy as np

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape, f"Predicted shape is {predicted.shape} while target is {target.shape}"
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_p1, wandb_id, last=True):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    file_name = 'last.pth.tr' if last else 'best.pth.tr'
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_p1': min_p1,
        'wandb_id': wandb_id,
    }, os.path.join(checkpoint_path, file_name))


def calculate_torso_diameter(gt_3d, left_shoulder_idx=5, right_shoulder_idx=2, left_hip_idx=11, right_hip_idx=8):
    """
    Calculate torso diameter using shoulder and hip distances (same as MPI-INF-3DHP standard)
    Args:
        gt_3d: Tensor of shape (N, T, J, 3) or (N, J, 3) with 3D ground truth keypoints.
        left_shoulder_idx: Index of left shoulder joint (default: 5)
        right_shoulder_idx: Index of right shoulder joint (default: 2)  
        left_hip_idx: Index of left hip joint (default: 11)
        right_hip_idx: Index of right hip joint (default: 8)
    Returns:
        torso_diameters: Tensor of shape (N,) with torso diameters
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

def compute_pck(pred, gt, torso_diameters, fixed_threshold=150.0, pck_thresholds=[0.1, 0.2, 0.3]):
    """
    Compute traditional PCK metric: percentage of keypoints within threshold
    
    Args:
        pred: Tensor of shape (N, J, 3) with predicted keypoints
        gt: Tensor of shape (N, J, 3) with ground truth keypoints  
        torso_diameters: Tensor of shape (N,) with torso diameters
        fixed_threshold: Fixed threshold in mm (default: 150mm)
        pck_thresholds: List of threshold percentages (default: [0.1, 0.2, 0.3] = 10%, 20%, 30%)
    
    Returns:
        pck_results: Dict with PCK values for each threshold
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
        
        # Calculate threshold: thresh_pct * 150mm (e.g., 10% of 150mm = 15mm)
        threshold = fixed_threshold * thresh_pct
        
        # Check which keypoints are within threshold
        correct_keypoints = (joint_errors <= threshold).float()  # (N, J)
        
        # PCK = percentage of ALL keypoints (across all samples and joints) that are correct
        pck = correct_keypoints.mean().item()
        pck_results[f'PCK@{thresh_pct_int}%_150mm'] = pck
    
    return pck_results

def compute_auc(pred, gt, max_threshold=150, num_steps=50):
    """
    Compute AUC by evaluating PCK over a range of thresholds.
    Args:
        pred: Tensor of shape (N, J, 3) with predicted keypoints.
        gt: Tensor of shape (N, J, 3) with ground truth keypoints.
        max_threshold: Maximum threshold in mm (default: 150).
        num_steps: Number of threshold steps (default: 50).
    Returns:
        auc: Area under the PCK curve.
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

