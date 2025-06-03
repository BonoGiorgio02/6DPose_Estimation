import torch
import torch.nn.functional as F
from torch import nn

class PoseLoss(nn.Module):
    """Loss function combines translation and rotation
        returns a weighted sum of the translation and rotation losses.
    """

    def __init__(self, alpha=1.0, beta=1.0):
        """Initialize alpha and beta values.

        Args:
            alpha (float, optional): Weight (importance) to give to translation. Defaults to 1.0.
            beta (float, optional): Weight (importance) to give to rotation. Defaults to 1.0.
        """
        super(PoseLoss, self).__init__()
        self.alpha = alpha  # weight for translation loss
        self.beta = beta    # weight for rotation loss

    def forward(self, pred_trans, pred_rot, gt_trans, gt_rot):
        """Initialize alpha and beta values.

          Args:
              pred_trans (matrix , optional): Predicted translation.
              pred_rot (matrix, optional): Predicted rotation.
              gt_trans (matrix, optional): Ground truth translation.
              gt_rot (matrix, optional): Ground truth rotation.
        """
        # Translation loss (MSE)
        trans_loss = F.mse_loss(pred_trans, gt_trans)

        # Frobenius norm of the differences between matrices
        rot_loss = F.mse_loss(pred_rot, gt_rot.view(-1, 4)) # view(-1,4) for quaternion, view(-1, 3, 3) for rotation matrix
        # rot_loss = self.quaternion_loss(pred_rot, gt_rot.view(-1, 4)) # view(-1,4) for quaternion, view(-1, 3, 3) for rotation matrix

        # Total Loss
        total_loss = self.alpha * trans_loss + self.beta * rot_loss

        return total_loss, trans_loss, rot_loss

    def quaternion_loss(self, pred_q, gt_q):
        pred_q = F.normalize(pred_q, dim=-1)
        gt_q = F.normalize(gt_q, dim=-1)
        dot = torch.sum(pred_q * gt_q, dim=-1).abs()
        return 1 - dot.mean()