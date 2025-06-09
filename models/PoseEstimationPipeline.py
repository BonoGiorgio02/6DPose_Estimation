import torch
from torch import nn
from models.PixelWiseFusionNetwork import PixelWiseFusionNetwork

class PoseEstimationPipeline(nn.Module):
    """
    Pipeline for pose estimation inspired to DenseFusion:
    - Pixel-wise pose prediction + confidence
    - Argmax for final selection
    """
    def __init__(self,geometric_dims=[64, 128, 256],fx=572.4114, fy=573.57043, cx=325.2611, cy=242.04899):
        super().__init__()

        # fusion network (without output_classes)
        self.fusion_network = PixelWiseFusionNetwork(geometric_dims=geometric_dims,fx=fx, fy=fy, cx=cx, cy=cy)

        # global pooling for reducing spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # final MLP for final pixel-wise prediction
        # Input: fusion_dim (pixel features) + 128 (global features)
        self.final_mlp_quat = nn.Sequential(
            nn.Conv1d(704, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.final_mlp_transl = nn.Sequential(
            nn.Conv1d(704, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.final_mlp_conf = nn.Sequential(
            nn.Conv1d(704, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        # head for rotation (quaternion for each pixel)
        self.rotation_head = nn.Conv1d(128, 4, 1)  # [B, 4, H, W]

        # head for translation (for each pixel)
        self.translation_head = nn.Conv1d(128, 3, 1)  # [B, 3, H, W]

        # head for confidence (for each pixel)
        self.confidence_head = nn.Conv1d(128, 1, 1)  # [B, 1, H, W]

    def forward(self, batch_data):
        """
        Args:
            batch_data: dict containing point_cloud, image, batch
        Returns:
            rotation_quaternion
            translation
            confidence
        """
        B, _, H, W = batch_data['cropped_img'].shape
        # extract pixel-wise feature fuse
        pixel_features = self.fusion_network(batch_data)    # [B, 704, N_valid]

        # pixel-wise prediction
        pixel_rotations = self.rotation_head(self.final_mlp_quat(pixel_features))  # [B, 4, N_valid]
        pixel_translations = self.translation_head(self.final_mlp_transl(pixel_features))  # [B, 3, N_valid]
        pixel_confidences = self.confidence_head(self.final_mlp_conf(pixel_features))  # [B, 1, N_valid]

        # apply sigmoid function to the confidence to normalize to [0,1]
        pixel_confidences = torch.sigmoid(pixel_confidences)

        # normalize quaternion for each pixel
        pixel_rotations_norm = pixel_rotations / torch.norm(pixel_rotations, dim=1, keepdim=True)

        pixel_rotations_norm = pixel_rotations_norm.transpose(2, 1)  # [B, N_valid, 4]
        pixel_translations = pixel_translations.transpose(2, 1)  # [B, N_valid, 3]
        pixel_confidences = pixel_confidences.transpose(2, 1)  # [B, N_valid, 1]

        return pixel_rotations_norm, pixel_translations, pixel_confidences