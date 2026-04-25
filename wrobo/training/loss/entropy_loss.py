import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss.
    KLDivLoss = D_{KL}(target || pred) = sum(target * log(target / pred))
    """
    def __init__(self, reduction: str = "batchmean", apply_nonlin: bool = True, eps: float = 1e-7):
        super(KLDivLoss, self).__init__()
        self.KL_loss = nn.KLDivLoss(reduction=reduction)
        self.apply_nonlin = apply_nonlin
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if self.apply_nonlin:
            pred = pred.softmax(dim=1)
            target = target.softmax(dim=1)

        pred = pred.clip(min=self.eps).log()
        target = target.clip(min=self.eps)

        loss = self.KL_loss(pred, target)

        return loss


class JSDivLoss(nn.Module):
    """
    Jensen-Shannon Divergence Loss.
    JSDivLoss = (D_{KL}(pred_p || pred_m) + D_{KL}(pred_q || pred_m)) / 2
    where pred_m = (pred_p + pred_q) / 2
    """
    def __init__(self, reduction: str = "batchmean", apply_nonlin: bool = True, eps: float = 1e-7):
        super(JSDivLoss, self).__init__()
        self.KL_loss = nn.KLDivLoss(reduction=reduction)
        self.apply_nonlin = apply_nonlin
        self.eps = eps

    def forward(self, pred_p: torch.Tensor, pred_q: torch.Tensor):
        if self.apply_nonlin:
            pred_p = pred_p.softmax(dim=1)
            pred_q = pred_q.softmax(dim=1)

        pred_m = ((pred_p + pred_q) / 2).clip(min=self.eps).log()
        pred_p = pred_p.clip(min=self.eps)
        pred_q = pred_q.clip(min=self.eps)

        loss = (self.KL_loss(pred_m, pred_p) + self.KL_loss(pred_m, pred_q)) / 2

        return loss
