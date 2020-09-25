import torch
import torch.nn as nn

__all__ = ['TripletLoss', 'ContrastiveLoss', 'PairwiseContrastive']


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, swap=False, absolute=False, reduction='mean'):
        super().__init__()
        self.abs = absolute
        if self.abs:
            reduction = 'none'
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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.m = margin
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, dist, class_):
        # the last dim should match len(class_) for broadcasting
        dist = dist.transpose(0, -1)
        loss = 0.5 * class_ * dist**2 + \
               (1 - class_) * 0.5 * torch.clamp(self.m - dist, min=0)**2
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class PairwiseContrastive(nn.Module):
    # input batch would contain [anchor1, pos1, anchor2, pos2, ...]
    def __init__(self, margin=1, order=1, reduction='mean'):
        super().__init__()
        self.p = order
        self.contr = ContrastiveLoss(margin=margin, reduction=reduction)

    # inferno trainer assumes every loss has a target; target not used here
    def forward(self, inp, target=None):
        inp = inp.flatten(1)
        assert len(inp) % 2 == 0
        samples1, samples2 = inp[::2], inp[1::2]
        pos_dist = (samples1 - samples2).norm(p=self.p, dim=1)
        neg_dist = (torch.roll(samples1, 1, 0) - samples2).norm(p=self.p, dim=1)
        labels = torch.cat((torch.ones(len(pos_dist)), torch.zeros(len(neg_dist)))).to(inp.device)
        loss = self.contr(torch.cat((pos_dist, neg_dist)), labels)
        return loss
