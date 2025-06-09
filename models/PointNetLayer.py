import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class PointNetLayer(MessagePassing):
    """PointNet layer for extracting geometric features"""
    def __init__(self, in_channels, out_channels, use_xyz=True):
        super().__init__(aggr='max') # for each node among all the received messages, take the max
        self.use_xyz = use_xyz # include or not the x,y,z coordinates
        mlp_channels = in_channels + 3 if use_xyz else in_channels
        # out_channels is number of features to output
        self.mlp = nn.Sequential(
            nn.Linear(mlp_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, h, pos, edge_index):
        h = h.view(-1, h.size(-1)) # h: [B, 1, N, C] or [B, N, C] -> Flatten to [B*N, C]
        pos = pos.view(-1, 3)
        # compute messages from each node j to its neighbour i, then aggregate by using aggr, it calls message() for each edge
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        # create message
        geometric_feat = pos_j - pos_i
        if self.use_xyz:
            geometric_feat = torch.cat([h_j, geometric_feat], dim=-1)
        else:
            geometric_feat = h_j
        return self.mlp(geometric_feat) # produce message vector