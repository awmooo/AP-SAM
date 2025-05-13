import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    损失函数DiceLoss
    """
    def __init__(self, from_logits=False, smooth=1e-7, eps=1e-7, reduction=None):
        super(DiceLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        if not self.from_logits: y_pred = F.sigmoid(y_pred)
        # flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = torch.sum(y_pred * y_true)
        cardinalities = torch.sum(y_pred + y_true) + self.smooth                           
        dice = (2.0 * intersection + self.smooth) / (cardinalities + self.smooth).clamp_min(self.eps)
        loss = 1 - dice
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction is None:
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
