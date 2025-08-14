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
    """
    Calculate MPJPE for Human3.6M dataset
    Args:
        predicted: (N, 17, 3) or (17, 3) predicted 3D poses
        target: (N, 17, 3) or (17, 3) ground truth 3D poses
    Returns:
        MPJPE in mm
    """
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

def calculate_torso_diameter_h36m(gt_3d, left_shoulder_idx=11, right_shoulder_idx=14, 
                                  left_hip_idx=4, right_hip_idx=1):
    """
    Calculate torso diameter for PCK metric - Human3.6M version
    
    Human3.6M joint indices:
    0: Hip (root), 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
    7: Spine, 8: Thorax, 9: Neck, 10: Head, 11: LShoulder, 12: LElbow, 13: LWrist,
    14: RShoulder, 15: RElbow, 16: RWrist
    
    Args:
        gt_3d: (N, 17, 3) ground truth 3D poses
        left_shoulder_idx: 11 (LShoulder)
        right_shoulder_idx: 14 (RShoulder) 
        left_hip_idx: 4 (LHip)
        right_hip_idx: 1 (RHip)
    
    Returns:
        torso_diameter: (N,) torso diameter for each pose
    """
    if isinstance(gt_3d, torch.Tensor):
        gt_3d = gt_3d.cpu().numpy()
    
    # Shoulder distance
    left_shoulder = gt_3d[:, left_shoulder_idx, :]  # (N, 3)
    right_shoulder = gt_3d[:, right_shoulder_idx, :]  # (N, 3)
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1)  # (N,)
    
    # Hip distance  
    left_hip = gt_3d[:, left_hip_idx, :]  # (N, 3)
    right_hip = gt_3d[:, right_hip_idx, :]  # (N, 3)
    hip_dist = np.linalg.norm(left_hip - right_hip, axis=1)  # (N,)
    
    # Average of shoulder and hip distance as torso diameter
    torso_diameter = (shoulder_dist + hip_dist) / 2.0
    return torso_diameter

def compute_pck_h36m(pred, gt, torso_diameters=None, fixed_threshold=150.0, pck_thresholds=[0.9, 0.8, 0.7]):
    """
    Compute PCK metrics for Human3.6M dataset
    
    Args:
        pred: (N, 17, 3) predicted 3D poses
        gt: (N, 17, 3) ground truth 3D poses  
        torso_diameters: (N,) torso diameter for each pose
        fixed_threshold: Fixed threshold in mm (default 150mm)
        pck_thresholds: List of percentage thresholds
    
    Returns:
        pck_results: Dictionary with PCK results
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # pred, gt shape: (N, 17, 3)
    joint_errors = np.linalg.norm(pred - gt, axis=2)  # (N, 17)
    
    pck_results = {}
    
    # Torso-based thresholds
    if torso_diameters is not None:
        for percentage in pck_thresholds:
            percentage_int = int(percentage * 100)
            threshold = torso_diameters[:, None] * percentage  # (N, 1)
            correct = joint_errors < threshold  # (N, 17)
            pck = np.mean(correct, axis=0)  # (17,) - per joint
            pck_results[f'PCK@{percentage_int}%_torso'] = np.mean(pck)  # Overall average
    
    # Fixed threshold (150mm)
    for percentage in pck_thresholds:
        percentage_int = int(percentage * 100)
        threshold = fixed_threshold * percentage
        correct = joint_errors < threshold
        pck = np.mean(correct, axis=0)
        pck_results[f'PCK@{percentage_int}%_150mm'] = np.mean(pck)
    
    return pck_results

def compute_auc_h36m(pred, gt, max_threshold=150, num_steps=50):
    """
    Compute AUC (Area Under Curve) for Human3.6M dataset
    
    Args:
        pred: (N, 17, 3) predicted 3D poses
        gt: (N, 17, 3) ground truth 3D poses
        max_threshold: Maximum threshold for AUC computation
        num_steps: Number of threshold steps
    
    Returns:
        auc: AUC value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # Calculate joint errors
    joint_errors = np.linalg.norm(pred - gt, axis=2)  # (N, 17)
    
    thresholds = np.linspace(0, max_threshold, num_steps)
    pck_values = []
    
    for threshold in thresholds:
        correct = joint_errors < threshold
        pck = np.mean(correct)
        pck_values.append(pck)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(pck_values, thresholds) / max_threshold
    return auc

def count_param_numbers(model):
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params

# Human3.6M specific joint mappings and constants
H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

# Human3.6M skeleton connections for visualization
H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),  # Right leg
    (0, 4), (4, 5), (5, 6),  # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16),  # Right arm
]

def make_root_relative_h36m(poses_3d, root_joint_idx=0):
    """
    Make Human3.6M poses root-relative by subtracting root joint position
    
    Args:
        poses_3d: (N, 17, 3) or (17, 3) 3D poses
        root_joint_idx: Index of root joint (0 for Hip in Human3.6M)
    
    Returns:
        Root-relative poses
    """
    if isinstance(poses_3d, torch.Tensor):
        return poses_3d - poses_3d[..., root_joint_idx:root_joint_idx+1, :]
    else:
        return poses_3d - poses_3d[..., root_joint_idx:root_joint_idx+1, :]

def denormalize_h36m(poses, scale_factor=1000.0):
    """
    Denormalize Human3.6M poses from normalized units to millimeters
    
    Args:
        poses: Normalized poses
        scale_factor: Scale factor to convert to mm
    
    Returns:
        Denormalized poses in mm
    """
    return poses * scale_factor

def normalize_h36m(poses, scale_factor=1000.0):
    """
    Normalize Human3.6M poses from millimeters to normalized units
    
    Args:
        poses: Poses in mm
        scale_factor: Scale factor to normalize
    
    Returns:
        Normalized poses
    """
    return poses / scale_factor