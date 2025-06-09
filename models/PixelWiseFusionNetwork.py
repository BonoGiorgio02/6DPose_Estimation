import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from models.GeometricFeatureExtractor import GeometricFeatureExtractor
from utils.PointProjector import PointProjector

class PixelWiseFusionNetwork(nn.Module):
    """
    Main network for pixel-wise fusion of geometric and colour features
    """
    def __init__(self,geometric_dims=[64, 128, 256],image_backbone='resnet18',fx=572.4114,fy=573.57043,cx=325.2611,cy=242.04899):
        super().__init__()

        self.geometric_dims = geometric_dims
        self.sample_img_features_dim = 256

        # geometric feature extractor
        self.geometric_extractor = GeometricFeatureExtractor(geometric_dims)

        # point projector with fixed intrinsics
        self.point_projector = PointProjector(fx, fy, cx, cy)

        # image feature extractor (CNN backbone)
        if image_backbone == 'resnet18':
            self.backbone = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
            self.image_encoder = nn.Sequential(*list(self.backbone.children())[:-2])
            image_feat_dim = 512
        else:
            raise NotImplementedError(f"Backbone {image_backbone} not implemented")

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # from 512 to 256
        self.features_reduction = nn.Sequential(
            nn.Conv2d(image_feat_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.sample_img_features_dim, 1),
            nn.BatchNorm2d(self.sample_img_features_dim),
            nn.ReLU(inplace=True)
            )

    def forward(self, batch):
        """
        Args:
            batch: dictionary containing:
                - cropped_img: [B, 3, H, W] RGB images cropped and padded
                - pointcloud: [B, N, 3] 3D coordinates
                - bbox_base: [B, 4] bbox parameters
                - paddings: [B, 4] padding parameters
        Returns:
            fused_features: [B, fusion_dim, H, W] feature fuse pixel-wise
        """
        # cropped and padded image
        B, _, H, W = batch['cropped_img'].shape
        image = batch['cropped_img']
        # extract geometric features for all batch items
        geometric_features_dict = self.geometric_extractor(batch)

        # extract image features
        image_features = self.image_encoder(image)  # [B, 512, H_pad/32, W_pad/32]
        image_features = self.features_reduction(image_features)  # [B, 256, H_pad/32, W_pad/32]
        image_features = F.interpolate(image_features, size=(H, W), mode='bilinear', align_corners=False) # [B, 256, H_pad, W_pad]

        # initialize fusion feature maps for each level
        fusion_maps = []

        # geometric_features = torch.cat([level_data['features'] for level_data in geometric_features_dict.values()], dim=1) # [B*N, 448]
        all_geometric_features = []
        for level_name in ['level1', 'level2', 'level3']:
            all_geometric_features.append(geometric_features_dict[level_name]['features'])

        geometric_features = torch.cat(all_geometric_features, dim=1)  # [B*N, sum(dims)]

        level_data = geometric_features_dict['level1']
        points_3d = level_data['pos']  # all points for all batches [B*N, 3]
        points_batch = level_data['batch']  # batch assignment for each point [B*N]

        # process each batch separately
        for b in range(B):
            # get points belonging to this batch (only 800)
            batch_mask = points_batch == b

            batch_points = points_3d[batch_mask]  # [N_b, 3]
            batch_geom_feats = geometric_features[batch_mask]  # [N_b, 448]
            bbox_base = batch['bbox_base'][b]  # [4]
            padding = batch['paddings'][b]  # [4]

            # project points to padded image coordinates
            pixel_coords, valid_mask = self.point_projector(batch_points, bbox_base, padding)

            # keep only valid points and features
            valid_points = batch_points[valid_mask] # [N_valid, 3]
            valid_geom_feats = batch_geom_feats[valid_mask] # [N_valid, feat_geom]
            valid_pixels = pixel_coords[valid_mask] # [N_valid, 2]

            # get subset of valid points
            indices = torch.randperm(valid_points.size(0))[:600]
            valid_points = valid_points[indices]
            valid_geom_feats = valid_geom_feats[indices]
            valid_pixels = valid_pixels[indices]

            if len(valid_points) == 0:
                print(f"[DEBUG] No valid points for batch {b}")
                continue

            img_feats = image_features[b]  # [C, H, W]
            C, H, W = img_feats.shape

            # valid_pixels: [N_valid, 2] -> [u, v] = [x, y]
            valid_pixels = valid_pixels.long()
            u = valid_pixels[:, 0]  # [N_valid]
            v = valid_pixels[:, 1]  # [N_valid]

            # flatten spatial dimensions: [C, H * W]
            img_feats_flat = img_feats.view(C, -1)

            # convert (v, u) to flat indices: idx = v * W + u
            flat_indices = v * W + u  # [N_valid]

            # expand indices to gather per channel: [C, N_valid]
            idx_expand = flat_indices.unsqueeze(0).expand(C, -1)  # [256, N_valid]

            # gather features
            sampled_feats = torch.gather(img_feats_flat, 1, idx_expand).T  # [N_valid, 256]

            fused_features = torch.cat([valid_geom_feats, sampled_feats], dim=1)  # [N_valid, 704]

            fusion_maps.append(fused_features)

        # for each element of batch, for each point there are 704 features
        fused_features = torch.stack(fusion_maps)  # [B, N_valid, 704]
        return fused_features.transpose(2, 1)  # [B, 704, N_valid]