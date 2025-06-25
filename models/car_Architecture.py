# This is just an example of architecture for applying the pose estimation to the autonomous car settings
# All the imports have to be added
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, use_xyz=True):
        super().__init__(aggr='max')
        self.use_xyz = use_xyz
        mlp_channels = in_channels + 3 if use_xyz else in_channels
        self.mlp = nn.Sequential(
            nn.Linear(mlp_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, h, pos, edge_index):
        h = h.view(-1, h.size(-1))        
        pos = pos.view(-1, 3)
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        geometric_feat = pos_j - pos_i
        if self.use_xyz:
            geometric_feat = torch.cat([h_j, geometric_feat], dim=-1)
        else:
            geometric_feat = h_j
        return self.mlp(geometric_feat)


class SetAbstractionLayer(nn.Module):
    def __init__(self, n_points, radius, n_neighbors, in_channels, out_channels):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.pointnet_layer = PointNetLayer(in_channels, out_channels)

    def forward(self, pos, h=None, batch_indices=None):
        if h is None:
          h = pos

        pos_batch = pos.shape[0]
        pos = pos.view(-1, 3) 
        
        if self.n_points is not None:
            centroids_idx = fps(pos, batch_indices, ratio=self.n_points*pos_batch/pos.shape[0], random_start=False)
            centroids = pos[centroids_idx] 
            centroids_batch = batch_indices[centroids_idx]

        else:
            centroids_idx = torch.arange(pos.size(0), device=pos.device)
            centroids = pos
            centroids_batch = batch_indices

        edge_index = radius(pos, centroids, r=self.radius,
                               batch_x=batch_indices, batch_y=centroids_batch,
                               max_num_neighbors=self.n_neighbors) 

        aggregated_h = self.pointnet_layer(h, pos, edge_index) 
        
        new_h = torch.zeros(centroids.size(0), aggregated_h.size(1),
                           device=aggregated_h.device, dtype=aggregated_h.dtype)

        for i, centroid_idx in enumerate(centroids_idx):
            mask = edge_index[0] == i
            if mask.any():
                neighbor_feats = aggregated_h[edge_index[1][mask]]
                new_h[i] = neighbor_feats.max(dim=0)[0]
        centroids = centroids.view(pos_batch, -1, 3)
        new_h = new_h.view(pos_batch, -1, new_h.size(-1))
        return centroids, new_h, centroids_batch, edge_index


class GeometricFeatureExtractor(nn.Module):
    def __init__(self, feature_dims=[64, 128, 256]):
        super().__init__()
        self.feature_dims = feature_dims

        self.sa1 = SetAbstractionLayer(
            n_points=800, radius=0.05, n_neighbors=32,
            in_channels=3, out_channels=feature_dims[0]
        )
        self.sa2 = SetAbstractionLayer(
            n_points=200, radius=0.1, n_neighbors=64,
            in_channels=feature_dims[0], out_channels=feature_dims[1]
        )
        self.sa3 = SetAbstractionLayer(
            n_points=50, radius=0.2, n_neighbors=64,
            in_channels=feature_dims[1], out_channels=feature_dims[2]
        )
        
        self.feat_proj1 = nn.Sequential(
            nn.Linear(feature_dims[0], feature_dims[0]),
            nn.BatchNorm1d(feature_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.feat_proj2 = nn.Sequential(
            nn.Linear(feature_dims[1], feature_dims[1]),
            nn.BatchNorm1d(feature_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.feat_proj3 = nn.Sequential(
            nn.Linear(feature_dims[2], feature_dims[2]),
            nn.BatchNorm1d(feature_dims[2]),
            nn.ReLU(inplace=True)
        )

    def upsampling_features(self, pos, h, edge_index):
      all_feats = torch.zeros(pos.shape[0], h.shape[1], device=pos.device, dtype=h.dtype) 

      counts = torch.zeros(pos.shape[0], 1, device=pos.device, dtype=h.dtype) 

      all_feats.index_add_(0, edge_index[1], h[edge_index[0]])
      counts.index_add_(0, edge_index[1], torch.ones_like(h[edge_index[0]][:, :1]))
      
      counts = counts.clamp(min=1)
      all_feats = all_feats / counts
      return all_feats

    def forward(self, batch):
        pointcloud = batch['pointcloud'] 
        B, N, _ = pointcloud.shape

        pos = pointcloud 
        batch_indices = torch.arange(B, device=pos.device).repeat_interleave(N) 

        pos1, h1, batch1, edge_index1 = self.sa1(pos, batch_indices=batch_indices)      
        pos2, h2, batch2, edge_index2 = self.sa2(pos1, h1, batch_indices=batch1)        
        pos3, h3, batch3, edge_index3 = self.sa3(pos2, h2, batch_indices=batch2)             

        h1 = h1.view(-1, h1.shape[-1])
        h2 = h2.view(-1, h2.shape[-1])
        h3 = h3.view(-1, h3.shape[-1])

        pos1 = pos1.view(-1, 3)
        pos2 = pos2.view(-1, 3)
        pos3 = pos3.view(-1, 3)

        h2_up = self.upsampling_features(pos1, h2, edge_index2) 
        h3_up = self.upsampling_features(pos2, h3, edge_index3) 
        h3_up = self.upsampling_features(pos1, h3_up, edge_index2) 

        return {
            'level1': {'pos': pos1, 'features': h1, 'batch': batch1},  
            'level2': {'pos': pos1, 'features': h2_up, 'batch': batch1}, 
            'level3': {'pos': pos1, 'features': h3_up, 'batch': batch1},
        }


class PointProjector(nn.Module):
    def __init__(self, fx=1108.5094, fy=1108.5094, cx=640.0, cy=360.0):
        super().__init__()
        self.register_buffer('camera_intrinsics', torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]))

    def forward(self, points_3d, bbox_base, padding, filtering=True):
        device = points_3d.device

        x_min, y_min, crop_width, crop_height = bbox_base.int()
        pad_left, pad_right, pad_top, pad_bottom = padding.int()

        points_3d_homo = torch.hstack([points_3d, torch.ones((points_3d.shape[0], 1)).to(device)]).to(device)

        points_camera_homo = (torch.tensor([
            [-0.4226, -0.9063, 0.     , -0.0453],
            [0.     ,  0     , -1.0000,  0.0100],
            [0.9063 , -0.4226, 0.     , -0.0661],
            [0.     ,  0.    ,  0.    ,  1.0000]
        ]).to(device) @ points_3d_homo.T).T
        points_camera = points_camera_homo[:, :3]  
        
        points_camera_t = points_camera.T
        points_2d_homo = (torch.tensor([
            [1108.5094, 0.0, 640.0],
            [0.0, 1108.5094, 360.0],
            [0.0, 0.0, 1.0]
        ]).to(device) @ points_camera_t)
        pixel_coords_original = points_2d_homo / (points_2d_homo[2:3,:]+1e-10)
        pixel_coords_original = pixel_coords_original[:2,:].T

        pixel_coords_crop = torch.zeros_like(pixel_coords_original)
        pixel_coords_crop[:, 0] = pixel_coords_original[:, 0] - x_min 
        pixel_coords_crop[:, 1] = pixel_coords_original[:, 1] - y_min 

        pixel_coords_padded = torch.zeros_like(pixel_coords_crop)
        pixel_coords_padded[:, 0] = pixel_coords_crop[:, 0] + pad_left
        pixel_coords_padded[:, 1] = pixel_coords_crop[:, 1] + pad_top

        if filtering:
            valid_depth = points_camera[:, 2] > 1e-8 
            
            valid_in_original_crop = (
                (pixel_coords_original[:, 0] >= x_min) &
                (pixel_coords_original[:, 0] < x_min + crop_width) &
                (pixel_coords_original[:, 1] >= y_min) &
                (pixel_coords_original[:, 1] < y_min + crop_height)
            )
            
            padded_width = crop_width + pad_left + pad_right
            padded_height = crop_height + pad_top + pad_bottom

            valid_in_padded = (
                (pixel_coords_padded[:, 0] >= 0) &
                (pixel_coords_padded[:, 0] < padded_width) &
                (pixel_coords_padded[:, 1] >= 0) &
                (pixel_coords_padded[:, 1] < padded_height)
            )

            valid_mask = valid_depth & valid_in_original_crop & valid_in_padded
        else:
            valid_mask = torch.ones(points_3d.shape[0], dtype=torch.bool, device=device)

        return pixel_coords_padded, valid_mask


class PixelWiseFusionNetwork(nn.Module):
    def __init__(self,
                 geometric_dims=[64, 128, 256],
                 image_backbone='resnet18',
                 fx=1108.5094, fy=1108.5094, cx=640.0, cy=360.0):
        super().__init__()

        self.geometric_dims = geometric_dims
        self.sample_img_features_dim = 256

        self.geometric_extractor = GeometricFeatureExtractor(geometric_dims)

        self.point_projector = PointProjector(fx, fy, cx, cy)

        if image_backbone == 'resnet18':
            self.backbone = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
            self.image_encoder = nn.Sequential(*list(self.backbone.children())[:-2])
            image_feat_dim = 512
        else:
            raise NotImplementedError(f"Backbone {image_backbone} not implemented")

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.features_reduction = nn.Sequential(
            nn.Conv2d(image_feat_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.sample_img_features_dim, 1),
            nn.BatchNorm2d(self.sample_img_features_dim),
            nn.ReLU(inplace=True)
            )

    def forward(self, batch):
        B, _, H, W = batch['cropped_img'].shape
        image = batch['cropped_img']
        geometric_features_dict = self.geometric_extractor(batch)

        image_features = self.image_encoder(image) 
        image_features = self.features_reduction(image_features) 
        image_features = F.interpolate(image_features, size=(H, W), mode='bilinear', align_corners=False) 

        fusion_maps = []

        all_geometric_features = []
        for level_name in ['level1', 'level2', 'level3']:
            all_geometric_features.append(geometric_features_dict[level_name]['features'])

        geometric_features = torch.cat(all_geometric_features, dim=1) 

        level_data = geometric_features_dict['level1']
        points_3d = level_data['pos'] 
        points_batch = level_data['batch'] 

        for b in range(B):
            batch_mask = points_batch == b

            batch_points = points_3d[batch_mask] 
            batch_geom_feats = geometric_features[batch_mask]
            bbox_base = batch['bbox_base'][b]
            padding = batch['paddings'][b]

            pixel_coords, valid_mask = self.point_projector(batch_points, bbox_base, padding)

            valid_points = batch_points[valid_mask]
            valid_geom_feats = batch_geom_feats[valid_mask]
            valid_pixels = pixel_coords[valid_mask]

            temp_batch = torch.ones(valid_points.size(0), dtype=torch.long, device=valid_points.device)

            indices = fps(valid_points, temp_batch, ratio=600.0/valid_points.size(0), random_start=False)
            valid_points = valid_points[indices]
            valid_geom_feats = valid_geom_feats[indices]
            valid_pixels = valid_pixels[indices]

            if len(valid_points) == 0:
                print(f"[DEBUG] No valid points for batch {b}")
                return torch.empty(1, 704, 0, device=image.device)

            img_feats = image_features[b] 
            C, H, W = img_feats.shape

            valid_pixels = valid_pixels.long()
            u = valid_pixels[:, 0]
            v = valid_pixels[:, 1]

            img_feats_flat = img_feats.view(C, -1)

            flat_indices = v * W + u

            idx_expand = flat_indices.unsqueeze(0).expand(C, -1) 

            sampled_feats = torch.gather(img_feats_flat, 1, idx_expand).T 

            fused_features = torch.cat([valid_geom_feats, sampled_feats], dim=1)

            fusion_maps.append(fused_features)

        if not fusion_maps:
            print("[DEBUG PixelWiseFusionNetwork] fusion_maps is empty. Cannot stack.")
            return torch.empty(B, 704, 0, device=image.device)


        fused_features = torch.stack(fusion_maps)
        return fused_features.transpose(2, 1) 

class PoseEstimationPipeline(nn.Module):
    def __init__(self,
                 geometric_dims=[64, 128, 256],
                 fx=1108.5094, fy=1108.5094, cx=640.0, cy=360.0):
        super().__init__()

        self.fusion_network = PixelWiseFusionNetwork(
            geometric_dims=geometric_dims,
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

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

        self.rotation_head = nn.Conv1d(128, 4, 1) 

        self.translation_head = nn.Conv1d(128, 3, 1) 

        self.confidence_head = nn.Conv1d(128, 1, 1)

    def forward(self, batch_data):
        B, _, H, W = batch_data['cropped_img'].shape
        pixel_features = self.fusion_network(batch_data) 

        if pixel_features.size(2) == 0:
             print("[DEBUG PoseEstimationPipeline] Fusion network returned empty tensors.")
             return torch.empty(B, 4, device=pixel_features.device), torch.empty(B, 3, device=pixel_features.device), torch.empty(B, 1, device=pixel_features.device)

        pixel_rotations = self.rotation_head(self.final_mlp_quat(pixel_features)) 
        pixel_translations = self.translation_head(self.final_mlp_transl(pixel_features)) 
        pixel_confidences = self.confidence_head(self.final_mlp_conf(pixel_features)) 

        pixel_confidences = torch.sigmoid(pixel_confidences)

        pixel_rotations_norm = pixel_rotations / torch.norm(pixel_rotations, dim=1, keepdim=True)

        pixel_rotations_norm = pixel_rotations_norm.transpose(2, 1)
        pixel_translations = pixel_translations.transpose(2, 1)
        pixel_confidences = pixel_confidences.transpose(2, 1)

        return pixel_rotations_norm, pixel_translations, pixel_confidences

    def get_pose_at_pixel(self, batch_data, pixel_coords):
        with torch.no_grad():
            B = batch_data['cropped_img'].shape[0]

            _, _, pixel_predictions = self.forward(batch_data)

            pixel_rotations = pixel_predictions['rotations'] 
            pixel_translations = pixel_predictions['translations'] 

            rotations = torch.zeros(B, 4, device=pixel_rotations.device)
            translations = torch.zeros(B, 3, device=pixel_translations.device)

            for b in range(B):
                u, v = pixel_coords[b]
                u, v = int(u.clamp(0, pixel_rotations.shape[3]-1)), int(v.clamp(0, pixel_rotations.shape[2]-1))
                rotations[b] = pixel_rotations[b, :, v, u]
                translations[b] = pixel_translations[b, :, v, u]

            return rotations, translations