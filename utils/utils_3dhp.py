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





def calculate_torso_diameter(gt_3d, left_shoulder_idx=5, right_hip_idx=8):
    """
    Calculate torso diameter as Euclidean distance between left shoulder and right hip.
    Returns None if keypoints are invalid (e.g., zeros or NaNs).
    Args:
        gt_3d: Tensor of shape (N, T, J, 3) or (N, J, 3) with 3D ground truth keypoints.
        left_shoulder_idx: Index of left shoulder joint.
        right_hip_idx: Index of right hip joint.
    Returns:
        torso_diameters: Tensor of shape (N,) with torso diameters, or None for invalid samples.
    """
    if gt_3d.dim() == 4:
        gt_3d = gt_3d[:, 0]  # Take first frame if temporal
    N = gt_3d.shape[0]
    torso_diameters = torch.zeros(N, device=gt_3d.device)
    
    for i in range(N):
        ls = gt_3d[i, left_shoulder_idx]  # Left shoulder
        rh = gt_3d[i, right_hip_idx]      # Right hip
        # Check if keypoints are valid (non-zero and non-NaN)
        if (torch.all(ls != 0) and torch.all(rh != 0) and
            not torch.any(torch.isnan(ls)) and not torch.any(torch.isnan(rh))):
            distance = torch.norm(ls - rh)
            torso_diameters[i] = distance
        else:
            torso_diameters[i] = 0  # Mark as invalid
    return torso_diameters

def compute_pck(pred, gt, torso_diameters, threshold_factor=0.1, pck_thresholds=[0.9, 0.8, 0.7]):
    """
    Compute PCK at specified thresholds (90%, 80%, 70%) using 0.1 * torso diameter.
    Args:
        pred: Tensor of shape (N, J, 3) with predicted keypoints.
        gt: Tensor of shape (N, J, 3) with ground truth keypoints.
        torso_diameters: Tensor of shape (N,) with torso diameters.
        threshold_factor: Fraction of torso diameter for error threshold (default: 0.1).
        pck_thresholds: List of PCK thresholds (e.g., [0.9, 0.8, 0.7]).
    Returns:
        pck_results: Dict with PCK values for each threshold.
    """
    N, J, _ = pred.shape
    pck_results = {f'PCK@{int(t*100)}%': 0.0 for t in pck_thresholds}
    valid_samples = 0

    for i in range(N):
        if torso_diameters[i] == 0:
            continue  # Skip samples with invalid torso diameter
        threshold = threshold_factor * torso_diameters[i]
        errors = torch.norm(pred[i] - gt[i], dim=-1)  # Per-joint Euclidean distance
        correct_keypoints = (errors <= threshold).float()
        for t in pck_thresholds:
            pck = (correct_keypoints.mean() >= t).float()
            pck_results[f'PCK@{int(t*100)}%'] += pck.item()
        valid_samples += 1

    if valid_samples > 0:
        for t in pck_thresholds:
            pck_results[f'PCK@{int(t*100)}%'] /= valid_samples
    return pck_results

def compute_auc(pred, gt, max_threshold=500, num_steps=50):
    """
    Compute AUC by evaluating PCK over a range of thresholds.
    Args:
        pred: Tensor of shape (N, J, 3) with predicted keypoints.
        gt: Tensor of shape (N, J, 3) with ground truth keypoints.
        max_threshold: Maximum threshold in mm (default: 500).
        num_steps: Number of threshold steps (default: 50).
    Returns:
        auc: Area under the PCK curve.
    """
    N, J, _ = pred.shape
    thresholds = np.linspace(0, max_threshold, num_steps)
    pck_values = []
    
    for thresh in thresholds:
        correct_keypoints = (torch.norm(pred - gt, dim=-1) <= thresh).float()
        pck = correct_keypoints.mean(dim=-1).mean().item()
        pck_values.append(pck)
    
    auc = np.trapz(pck_values, thresholds) / max_threshold
    return auc




