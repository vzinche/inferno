import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.exceptions import assert_


class WeightedMSELoss(nn.Module):
    NEGATIVE_CLASS_WEIGHT = 1.

    def __init__(self, positive_class_weight=1., positive_class_value=1., size_average=True):
        super(WeightedMSELoss, self).__init__()
        assert_(positive_class_weight >= 0,
                "Positive class weight can't be less than zero, got {}."
                .format(positive_class_weight),
                ValueError)
        self.mse = nn.MSELoss(size_average=size_average)
        self.positive_class_weight = positive_class_weight
        self.positive_class_value = positive_class_value

    def forward(self, input, target):
        # Get a mask
        positive_class_mask = target.data.eq(self.positive_class_value).type_as(target.data)
        # Get differential weights (positive_weight - negative_weight,
        # i.e. subtract 1, assuming the negative weight is gauged at 1)
        weight_differential = (positive_class_mask
                               .mul_(self.positive_class_weight - self.NEGATIVE_CLASS_WEIGHT))
        # Get final weight by adding weight differential to a tensor with negative weights
        weights = weight_differential.add_(self.NEGATIVE_CLASS_WEIGHT)
        # `weights` should be positive if NEGATIVE_CLASS_WEIGHT is not messed with.
        sqrt_weights = weights.sqrt_()
        return self.mse(input * sqrt_weights, target * sqrt_weights)


class Norm(nn.Module):
    def __init__(self, order=1, size_average=True):
        super().__init__()
        self.order = order
        self.average = size_average

    def forward(self, inp, target=None):
        if target is not None:
            inp = inp - target
        inp = inp.flatten()
        norm = torch.norm(inp, p=self.order)
        if self.average:
            norm = norm / len(inp)
        return norm


class MaskedBce(nn.Module):
    # Masked binary crossentropy to target
    def __init__(self, mask_value=False):
        super(MaskedBce, self).__init__()
        self.mask_value = mask_value     # the value to ignore for reconstruction loss

    def forward(self, output, target):
        # check if bool False and not int 0
        if not self.mask_value and isinstance(self.mask_value, bool):
            mask = torch.ones(target.shape).bool()
        else:
            mask = (target != self.mask_value)
        bce_loss = F.binary_cross_entropy(output[mask].flatten(),
                                          target[mask].flatten())
        return bce_loss


class SoftCE(nn.Module):
    def __init__(self, smoothing=0.2):
        super().__init__()
        # smoothing can be a fixed value or a range
        assert isinstance(smoothing, float) or len(smoothing) == 2
        self.smoothing = smoothing

    def forward(self, input_, target):
        smooth = self.smoothing if isinstance(self.smoothing, float) \
                 else np.random.uniform(*self.smoothing)
        num_labels = input_.shape[-1]
        target_onehot = torch.eye(num_labels)[target].to(target.device)
        target_onehot[target_onehot == 1] = 1 - smooth
        target_onehot[target_onehot == 0] = smooth / (num_labels - 1)
        logprobs = F.log_softmax(input_, dim=1)
        return  -(target_onehot * logprobs).sum() / input_.shape[0]
