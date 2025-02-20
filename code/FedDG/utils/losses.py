import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

def dice_loss(score, target):

    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        # print("In loop", type(i), i)
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss

def mixup_dice_loss(dice_loss, score, y_a, y_b, lam):
    return lam * dice_loss(score, y_a) + (1 - lam) * dice_loss(score, y_b)

def kl_divergence_loss(score, target):
    target = target.float()

    log_score = F.log_softmax(score, dim=1)

    target_prob = F.softmax(target, dim=1)

    kl_loss = nn.KLDivLoss(reduction='batchmean')

    loss = kl_loss(log_score, target_prob)

    return loss


def l1_loss(score, target):
    target = target.float()
    loss = 0
    for i in range(target.shape[1]):  # 遍历每个类别或通道
        loss += F.l1_loss(score[:, i, ...], target[:, i, ...], reduction='mean')
    loss = loss / target.shape[1]  # 计算所有类别的平均 L1 损失

    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
