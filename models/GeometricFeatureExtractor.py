import torch
from torch import nn
from models.SetAbstractionLayer import SetAbstractionLayer

class GeometricFeatureExtractor(nn.Module):
    """
    Extract geometric features from point cloud and produce embeddings for fusion
    """
    def __init__(self, feature_dims=[64, 128, 256]):
        super().__init__()
        self.feature_dims = feature_dims

        # multi-scale feature extraction
        # given the input pointcloud, sample n_points by using FPS, radius specifies the region to find the nearest (at most) n_neighbours
        # in_channels=3 as input pointcloud has for each point 3 coordinates (x,y,z)
        # in summary this layer takes in a batch of shape [batch_size,n_points,3], it samples 800 points (centroid) [batch_size,800,3]
        # then for each point find at most n_neighbours nearest neighbours [batch_size,800,32,3], then there is MLP and pooling to learn local features [batch_size, 800, feature_dims[0]]
        # it returns centroids, new_h, centroids_batch, edge_index ([N_centroids, 3] [N_centroids, out_channels] [N_centroids], [2, E])
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

    def upsampling_features(self, pos, h, edge_index):
        # pos [B*N, 3], h [M, C], edge:index [2, E]

        # edge_index has shape [2,E]
        # edge_index[1] -> idx of original points (pos)
        # edge_index[0] -> idx of centroids

        # create an empty tensor for original points
        all_feats = torch.zeros(pos.shape[0], h.shape[1], device=pos.device, dtype=h.dtype) # [B*N, C]

        # count how many centroids are related to each point
        counts = torch.zeros(pos.shape[0], 1, device=pos.device, dtype=h.dtype) # [B*N, 1]

        # sum features of each centroid to all neighbour points
        # accumulate the elements of h[edge_index[0]] by adding to the indices in the order given by edge_index[1]
        # for each point in edge_index[1] assign the corresponding centroid feature, if same index there are more features do the sum
        all_feats.index_add_(0, edge_index[1], h[edge_index[0]])
        counts.index_add_(0, edge_index[1], torch.ones_like(h[edge_index[0]][:, :1]))

        # average (avoid zero division)
        counts = counts.clamp(min=1) # set minimum value 1
        all_feats = all_feats / counts
        return all_feats

    def forward(self, batch):
        """
        Args:
            batch: dict containing 'pointcloud': [B, N, 3] point cloud coordinates
        Returns:
            multi_scale_features: dict con features a diverse risoluzioni
        """
        pointcloud = batch['pointcloud']  # [B, N, 3]
        B, N, _ = pointcloud.shape

        # flatten pointcloud and create batch indices
        pos = pointcloud # [B, N, 3]
        # repeat batch_id for each point
        batch_indices = torch.arange(B, device=pos.device).repeat_interleave(N)  # [B*N]

        # extract multi-scale features with proper batch tracking
        # [B, 800, 3] [B, 800, 64] [B*800] [2, E] output shapes
        # centroids, new_h (feature vector), centroids_batch (batch_id for each point), edge_index
        pos1, h1, batch1, edge_index1 = self.sa1(pos, batch_indices=batch_indices)
        # [B, 200, 3] [B, 200, 128] [B*200] [2, E] output shapes
        pos2, h2, batch2, edge_index2 = self.sa2(pos1, h1, batch_indices=batch1)
        # [B, 50, 3] [B, 50, 256] [B*50] [2, E] output shapes
        pos3, h3, batch3, edge_index3 = self.sa3(pos2, h2, batch_indices=batch2)

        h1 = h1.view(-1, h1.shape[-1]) # [B*800, 64]
        h2 = h2.view(-1, h2.shape[-1]) # [B*200, 128]
        h3 = h3.view(-1, h3.shape[-1]) # [B*50, 256]

        pos1 = pos1.view(-1, 3) # [B*800, 64]
        pos2 = pos2.view(-1, 3) # [B*200, 128]
        pos3 = pos3.view(-1, 3) # [B*50, 256]

        # upsample such that feature vectors are [B*800,features] and they can be stacked
        # receives [B*800, 3], [B*200, 128], [2, E], E=B*200*n_centroids, centroids_id in edge_index2[0] are 0 to 199
        h2_up = self.upsampling_features(pos1, h2, edge_index2)  # [B*800, 128]

        # receives [B*200, 3], [B*50, 256], [2, E]
        h3_up = self.upsampling_features(pos2, h3, edge_index3) # [B*200, 256]
        # receives [B*800, 3], [B*200, 256], [2, E]
        h3_up = self.upsampling_features(pos1, h3_up, edge_index2) # [B*800, 256]

        return {
            'level1': {'pos': pos1, 'features': h1, 'batch': batch1},  # Fine details h1 [B*800, 64]
            'level2': {'pos': pos1, 'features': h2_up, 'batch': batch1},  # Medium details h2 [B*800, 128]
            'level3': {'pos': pos1, 'features': h3_up, 'batch': batch1},  # Coarse details h3 [B*800, 256]
        }