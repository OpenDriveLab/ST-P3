import torch
import torch.nn as nn

from stp3.layers.convolutions import Bottleneck


class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
        self, in_channels, latent_dim, method="GAUSSIAN"):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.method = method

        if method == 'GAUSSIAN':
            self.encoder = DistributionEncoder(in_channels, self.compress_dim)
            self.decoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
            )
        elif method == 'MIXGAUSSIAN':
            self.encoder = DistributionEncoder(in_channels, self.compress_dim)
            self.decoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.compress_dim, out_channels=6 * self.latent_dim + 3, kernel_size=1)
            )
        elif method == 'BERNOULLI':
            self.encoder = nn.Sequential(
                Bottleneck(in_channels, self.latent_dim)
            )
            self.decoder = nn.LogSigmoid()
        else:
            raise NotImplementedError

    def forward(self, s_t):
        b, s = s_t.shape[:2]
        assert s == 1
        encoding = self.encoder(s_t[:, 0])

        if self.method == 'GAUSSIAN':
            decoder = self.decoder(encoding).view(b, 1, 2 * self.latent_dim)
        elif self.method == 'MIXGAUSSIAN':
            decoder = self.decoder(encoding).view(b, 1, 6 * self.latent_dim + 3)
        elif self.method == 'BERNOULLI':
            decoder = self.decoder(encoding)
        else:
            raise NotImplementedError

        return decoder


class DistributionEncoder(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            Bottleneck(in_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
        )

    def forward(self, s_t):
        return self.model(s_t)
