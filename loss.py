"""!@file losses.py

@brief Custom loss functions for training the model
"""

import torch
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss


class ComboLoss(nn.Module):
    """!
    @brief Combination of soft-dice loss and modified cross entropy loss,

    @details Implementation is exactly as introduced in the original paper
    "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation" (https://arxiv.org/pdf/1805.02798.pdf).
    alpha controls the contribution of the modified cross entropy loss, while beta controls
    penalization of false positives vs false negatives. beta < 0.5 penalizes false positives more.
    alpha = 2/3, beta = 0.5 yields equal weight to the loss terms since then alpha * beta = (1 - alpha).
    The smoothing term serves several purposes:
    - prevents division by zero
    - allows a non-zero derivative when there is no ground truth mask
    - gives rise to a smoother loss surface which helps stabalize the learning process
    """

    def __init__(self, alpha=2 / 3, beta=0.5, smooth=1.0, eps=1e-7):
        """!
        @param alpha: weight of the modified cross entropy loss compared to soft dice loss.
        @param beta: weight for controlling penalization of false positives/negatives within
                    the modified cross entropy loss.
        @param smooth: smoothing term
        @param eps: small constant to prevent numerical issues from log(probs)
        """
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(self, probs, labels):
        """!
        @param probs: predicted probabilities - tensor of shape (batch_size, 1, H, W)
        @param labels: true masks - tensor of shape (batch_size, 1, H, W)

        @return combo: tensor of shape (1, )
        """
        # calculate soft-dice coefficient
        intersection = (probs * labels).sum()
        dice = (2.0 * intersection + self.smooth) / \
            (probs.sum() + labels.sum() + self.smooth)

        # calculate modified cross entropy loss
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        modified_bce = -(self.beta * labels * torch.log(probs) +
                         (1 - self.beta) * (1 - labels) * torch.log(1 - probs)).mean()

        # calculate combo loss
        combo = self.alpha * modified_bce - (1 - self.alpha) * dice

        return combo


class SoftDiceLoss(nn.Module):
    """!
    @brief soft-dice loss for binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, probs, labels):
        """!
        @param probs: predicted probabilities - tensor of shape (batch_size, 1, H, W)
        @param labels: true masks - tensor of shape (batch_size, 1, H, W)

        @return loss: tensor of shape (1, )
        """

        intersection = (probs * labels).sum()
        denom = (probs.pow(self.p) + labels.pow(self.p)).sum()
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)

        return 1.0 - dice


class MLoss(nn.Module):
    """!
    @brief Combination of soft-dice loss and modified cross entropy loss,

    @details Implementation is exactly as introduced in the original paper
    "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation" (https://arxiv.org/pdf/1805.02798.pdf).
    alpha controls the contribution of the modified cross entropy loss, while beta controls
    penalization of false positives vs false negatives. beta < 0.5 penalizes false positives more.
    alpha = 2/3, beta = 0.5 yields equal weight to the loss terms since then alpha * beta = (1 - alpha).
    The smoothing term serves several purposes:
    - prevents division by zero
    - allows a non-zero derivative when there is no ground truth mask
    - gives rise to a smoother loss surface which helps stabalize the learning process
    """

    def __init__(self, alpha=2 / 3, beta=0.5, smooth=1.0, eps=1e-7):
        """!
        @param alpha: weight of the modified cross entropy loss compared to soft dice loss.
        @param beta: weight for controlling penalization of false positives/negatives within
                    the modified cross entropy loss.
        @param smooth: smoothing term
        @param eps: small constant to prevent numerical issues from log(probs)
        """
        super(MLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(self, probs, labels):
        """!
        @param probs: predicted probabilities - tensor of shape (batch_size, 1, H, W)
        @param labels: true masks - tensor of shape (batch_size, 1, H, W)

        @return combo: tensor of shape (1, )
        """
        # calculate soft-dice coefficient
        intersection = (probs * labels).sum()
        dice = (2.0 * intersection + self.smooth) / \
            (probs.sum() + labels.sum() + self.smooth)

        # calculate modified cross entropy loss
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        modified_bce = -(self.beta * labels * torch.log(probs) +
                         (1 - self.beta) * (1 - labels) * torch.log(1 - probs)).mean()

        # calculate combo loss
        combo = self.alpha * modified_bce + (1 - self.alpha) * (1 - dice)

        return combo


class CombinedLoss(nn.Module):


    def __init__(self, alpha=1):
        """!
        @param alpha: weight of the modified cross entropy loss compared to soft dice loss.
        @param beta: weight for controlling penalization of false positives/negatives within
                    the modified cross entropy loss.
        @param smooth: smoothing term
        @param eps: small constant to prevent numerical issues from log(probs)
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, probs, labels):
        return self.alpha * torch.nn.BCELoss()(probs, labels) + (2 - self.alpha) *DiceLoss(mode='binary')(probs, labels) 

