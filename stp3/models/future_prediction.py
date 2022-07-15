import torch
import torch.nn as nn

from stp3.layers.convolutions import Bottleneck, Block, DeepLabHead
from stp3.layers.temporal import SpatialGRU, Dual_GRU, BiGRU

class FuturePrediction(nn.Module):
    def __init__(self, in_channels, latent_dim, n_future, mixture=True, n_gru_blocks=2, n_res_layers=1):
        super(FuturePrediction, self).__init__()
        self.n_spatial_gru = n_gru_blocks

        gru_in_channels = latent_dim
        self.dual_grus = Dual_GRU(gru_in_channels, in_channels, n_future=n_future, mixture=mixture)
        self.res_blocks1 = nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)])

        self.spatial_grus = []
        self.res_blocks = []
        for i in range(self.n_spatial_gru):
            self.spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                self.res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
            else:
                self.res_blocks.append(DeepLabHead(in_channels, in_channels, 128))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)


    def forward(self, x, state):
        # x has shape (b, 1, c, h, w), state: torch.Tensor [b, n_present, hidden_size, h, w]
        x = self.dual_grus(x, state)

        b, n_future, c, h, w = x.shape
        x = self.res_blocks1(x.view(b * n_future, c, h, w))
        x = x.view(b, n_future, c, h, w)

        x = torch.cat([state, x], dim=1)

        hidden_state = x[:, 0]
        for i in range(self.n_spatial_gru):
            x = self.spatial_grus[i](x, hidden_state)

            b, s, c, h, w = x.shape
            x = self.res_blocks[i](x.view(b*s, c, h, w))
            x = x.view(b, s, c, h, w)

        return x