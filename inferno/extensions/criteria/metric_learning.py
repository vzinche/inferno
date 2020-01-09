import torch
import torch.nn as nn

__all__ = ['TripletLoss']


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, swap=False, absolute=False, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.abs = absolute
        if self.abs:
            reduction = 'none'
        if reduction == 'mean':     # pytorch, what's wrong with you
            reduction = 'elementwise_mean'
        self.loss = nn.TripletMarginLoss(margin=margin, p=p,
                                         swap=swap, reduction=reduction)

    def forward(self, triplet, target=None):
        # inferno trainer assumes every loss has a target; target not used here
        # we might or might not have the batch dim here
        assert len(triplet) == 3 or len(triplet[0]) == 3
        if len(triplet) == 3:
            triplet = triplet.unsqueeze(0)
        loss = self.loss(triplet[:, 0], triplet[:, 1], triplet[:, 2])
        if self.abs:
            loss = sum(loss > 0).type(torch.float) / len(triplet)
        return loss
