import torch
import os
import trimesh
import torch.nn.functional as F
from torch import nn
from utils.models_point_loader import load_models_points

class PoseLossExtension(nn.Module):
    """Loss function combines translation and rotation
        returns a weighted sum of the translation and rotation losses.
    """

    def __init__(self, alpha=1.0, beta=1.0, class_names=None, device=torch.device("cpu")):
        """Initialize alpha and beta values.

        Args:
            alpha (float, optional): Weight (importance) to give to translation. Defaults to 1.0.
            beta (float, optional): Weight (importance) to give to rotation. Defaults to 1.0.
        """
        super(PoseLossExtension, self).__init__()
        self.alpha = alpha  # weight for translation loss
        self.beta = beta    # weight for rotation loss
        self.models_dir = "./datasets/linemod/DenseFusion/Linemod_preprocessed/models"
        # meters
        self.models_dict = load_models_points(self.models_dir, class_names, device)
    
    def forward(self, pixel_rotations_norm, pixel_translations, pixel_confidences, gt_trans, gt_rot, obj_id=None):
        """Initialize alpha and beta values.

          Args:
              pred_trans (matrix , optional): Predicted translation.
              pred_rot (matrix, optional): Predicted rotation.
              gt_trans (matrix, optional): Ground truth translation.
              gt_rot (matrix, optional): Ground truth rotation.
        """

        # gt_rot has shape [batch_size, 3, 3], while pixel_rotations_norm has shape [batch_size, N_valid, 4]
        total_loss = self.dense_loss(pixel_rotations_norm, pixel_translations, pixel_confidences, gt_trans, gt_rot, obj_id)

        return total_loss

    def dense_loss(self, pred_r, pred_t, pred_c, gt_trans, gt_rot, obj_id):

        # pred_r is quaternion ([batch_size, N_valid, 4]), gt_rot is matrix ([batch_size, 3, 3]), gt_trans ([batch_size, 3])
        # pred_c ([batch_size, N_valid, 1]), pred_t ([batch_size, N_valid, 3])
        bs, num_p, _ = pred_c.size()
        # after stack and unsqueeze ([batch_size, 1, points_loaded, 3])
        models_point = torch.stack([self.models_dict[f"{obj:02d}"] for obj in obj_id]).unsqueeze(1)

        # convert pred_r to rotation matrix
        pred_base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]*2 + pred_r[:, :, 3]*2)).view(bs, num_p, 1),\
                                (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                                (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                                (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                                (1.0 - 2.0*(pred_r[:, :, 1]*2 + pred_r[:, :, 3]*2)).view(bs, num_p, 1), \
                                (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                                (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                                (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                                (1.0 - 2.0*(pred_r[:, :, 1]*2 + pred_r[:, :, 2]*2)).view(bs, num_p, 1)), dim=2).view(bs * num_p, 3, 3)
        
        # unsqueeze(1) ([batch_size, 1, 3, 3]), repeat(1, N_valid, 1, 1) ([batch_size, N_valid, 3, 3])
        gt_rot = gt_rot.unsqueeze(1).repeat(1, num_p, 1, 1).view(bs * num_p, 3, 3)
        # unsqueeze(1) ([batch_size, 1, 3]), repeat(1, N_valid, 1) ([batch_size, N_valid, 3])
        gt_trans = gt_trans.unsqueeze(1).repeat(1, num_p, 1).view(bs * num_p, 3)
        # pred_c ([batch_size*N_valid, 1])
        pred_c = pred_c.view(bs* num_p)
        # models_points ([batch_size, 1, points_loaded, 3]), repeat ([[batch_size, N_valid, points_loaded, 3]])
        model_points = models_point.repeat(1, num_p, 1, 1).view(bs * num_p, self.models_dict['01'].shape[0] , 3)
        # predicted points, [batch_size*N_valid, points_loaded, 3] * [batch_size*N_valid, 3, 3] + [batch_size*N_valid, 1, 3]
        pred_points = torch.bmm(model_points, pred_base.transpose(1, 2)) + pred_t.contiguous().view(bs * num_p, 1, 3)
        # ground truth points, [batch_size*N_valid, points_loaded, 3] * [batch_size*N_valid, 3, 3] + [batch_size*N_valid, 1, 3]
        gt_points = torch.bmm(model_points, gt_rot.transpose(1, 2)) + gt_trans.view(bs * num_p, 1, 3)
        # pred_points, gt_points ([batch_size*N_valid, points_loaded, 3])
        # torch.norm((pred_points-gt_points), dim=2) ([batch_size*N_valid, points_loaded])
        # distance ([batch_size*N_valid])
        distance = torch.mean(torch.norm((pred_points-gt_points), dim=2), dim=1)
        # loss depends on confidence
        loss = torch.mean((distance * pred_c - 0.015* torch.log(pred_c)), dim=0)

        # pred_c ([batch_size, N_valid])
        pred_c = pred_c.view(bs, num_p)
        # for each row find index of max, for each element of batch find index of the most confident point
        _, max_conf_index = torch.max(pred_c, dim=1)
        # distance ([batch_size, N_valid])
        distance = distance.view(bs, num_p)

        # get translation of the most confident point in each element of batch
        t = pred_t[torch.arange(bs), max_conf_index]
        pred_base = pred_base.view(bs,num_p,3,3)
        # get rotation (matrix)
        r= pred_base[torch.arange(bs), max_conf_index]

        return loss, r, t