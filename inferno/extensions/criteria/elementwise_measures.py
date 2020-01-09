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


class BceRegularized(nn.Module):
    # Binary crossentropy to target plus norm on embedding
    def __init__(self, norm=2, mask_value=False):
        super(BceRegularized, self).__init__()
        self.norm = norm
        self.mask_value = mask_value     # the value to ignore for reconstruction loss


    def forward(self, output, target):
        # check if bool False and not int 0
        if not self.mask_value and isinstance(self.mask_value, bool):
            mask = torch.ones(target.shape).byte()
        else:
            mask = (target != self.mask_value)
        # The output should include the embedding
        assert len(output) == 2
        reconstruction, embedding = output
        bce_loss = F.binary_cross_entropy(reconstruction[mask].flatten(),
                                          target[mask].flatten())
        reg_loss = embedding.norm(p=self.norm) / len(embedding.flatten())
        return bce_loss + reg_loss
