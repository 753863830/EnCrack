import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2,  reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         log_probs = F.logsigmoid(input)
#         probs = torch.exp(log_probs)
#
#         # 计算类别权重
#
#         alpha = torch.tensor(self.alpha).to(input.device)
#         # weights = torch.pow(1 - probs, self.gamma) * alpha
#
#         # 计算 Focal Loss
#         # loss = -weights * target * log_probs - (1 - weights) * (1 - target) * log_probs
#         loss = - alpha * torch.pow(1 - probs, self.gamma) * target * log_probs - \
#                (1 - alpha) * torch.pow(probs, 2) * (1 - target) * torch.log2_(1-probs)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#
#         return loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        # targets=targets.squeeze().long()
        # ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        #单个类被只是用二元交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()