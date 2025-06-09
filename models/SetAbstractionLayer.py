import torch
from torch import nn
from models.PointNetLayer import PointNetLayer
from torch_geometric.nn.pool import fps, radius

class SetAbstractionLayer(nn.Module):
    """Set Abstraction for PointNet++"""
    def __init__(self, n_points, radius, n_neighbors, in_channels, out_channels):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.pointnet_layer = PointNetLayer(in_channels, out_channels)

    def forward(self, pos, h=None, batch_indices=None):
        # if no extra feature provided, use coordinates (x,y,z)
        if h is None:
            h = pos

        pos_batch = pos.shape[0]
        pos = pos.view(-1, 3)  # -> [B, N, 3] -> [B*N, 3]
        # sampling with proper batch handling
        if self.n_points is not None:
            # N can be n_pointcloud_points, 800, 200
            # sample n_points using FPS, self.n_points can be 800, 200, 50, ratio=self.n_points/N, so 800/n_pointcloud_points,
            # 200/800, 50/200
            # order of batch_indices is important, if it 0,0,...,0,1,...,1,2,... and so on, then the results are all points for batch 0
            # then all points for the following batch and so on
            centroids_idx = fps(pos, batch_indices, ratio=self.n_points*pos_batch/pos.shape[0]) # for each batch_id n_centroids=800, 200, 50
            centroids = pos[centroids_idx] # for each batch_id [N_centroids, 3]
            # for each element in centroids, get batch_id
            centroids_batch = batch_indices[centroids_idx]# for each batch_id [N_centroids]
        
        else:
            # if downsampling not needed
            centroids_idx = torch.arange(pos.size(0), device=pos.device)
            centroids = pos
            centroids_batch = batch_indices
        
        # batch_x specifies the batch_id of each point in pos, while batch_y batch_id of each centroid
        # the shape is [2,E] where E is number of edges, it can be at most B*self.n_centroids*self.n_neighbours, first row contains indices of centroids (indices of rows),
        # other one points
        # it is batch_id aware thanks to batch_x and batch_y, the output has as second row the indices of rows of pos, which has shape [B*N,3]
        # edge_index has all centroids of a batch first, then another one
        edge_index = radius(pos, centroids, r=self.radius,
                                batch_x=batch_indices, batch_y=centroids_batch,
                                max_num_neighbors=self.n_neighbors)
        
        # PointNet
        aggregated_h = self.pointnet_layer(h, pos, edge_index) # for each batch_id [self.n_points, out_channels], so it has shape [B*self.n_points, out_channels]
        # aggregate per centroids
        new_h = torch.zeros(centroids.size(0), aggregated_h.size(1),
                           device=aggregated_h.device, dtype=aggregated_h.dtype) # [B*N_centroids, out_channels]

        # for each centroid
        for i, centroid_idx in enumerate(centroids_idx): # centroids_idx is the list of centroids (values can be from 0 to self.n_points-1)
            mask = edge_index[0] == i # consider edges connected to centroid i (index of centroids)
            if mask.any():
                # if there are edges, edge_index[1][mask] return the selected points index, which can be from 0 to B*self.n_points-1
                neighbor_feats = aggregated_h[edge_index[1][mask]]
                # for each centroid, take max feature
                new_h[i] = neighbor_feats.max(dim=0)[0] # each element has out_channels elements
        centroids = centroids.view(pos_batch, -1, 3) # it returns for each batch_id the centroids, as the above 
        new_h = new_h.view(pos_batch, -1, new_h.size(-1)) # it can be done, just because centroids_idx has points grouped by batch_id
        return centroids, new_h, centroids_batch, edge_index # [B, N_centroids, 3] [B, N_centroids, out_channels] [B, N_centroids], [2, E]