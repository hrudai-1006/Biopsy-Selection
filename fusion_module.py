import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, in_channels=3, base_filters=16, out_features=128, is_3d=False):
        """
        Fuses T2W, DWI, and DCE modalities.

        Args:
            in_channels (int): Number of input modalities (default: 3).
            base_filters (int): Number of filters in the first conv layer.
            out_features (int): Dimension of the output feature vector.
            is_3d (bool): If True, uses 3D convolutions. If False, uses 2D.
        """
        super(MultimodalFusion, self).__init__()

        self.is_3d = is_3d

        if is_3d:
            Conv = nn.Conv3d
            MaxPool = nn.MaxPool3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        else:
            Conv = nn.Conv2d
            MaxPool = nn.MaxPool2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d

        self.features = nn.Sequential(
            Conv(in_channels, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(kernel_size=2, stride=2),

            Conv(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(kernel_size=2, stride=2),

            Conv(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(kernel_size=2, stride=2),
        )

        self.global_pool = AdaptiveAvgPool(1)

        # Projection to fixed feature size
        self.projection = nn.Sequential(
            nn.Linear(base_filters * 4, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, H, W) or (B, 3, D, H, W)
        Returns:
            fused_features: Tensor of shape (B, out_features)
            spatial_features: Intermediate spatial features before pooling
        """
        spatial_features = self.features(x)
        pooled = self.global_pool(spatial_features)
        pooled = pooled.view(pooled.size(0), -1)
        fused_features = self.projection(pooled)
        return fused_features, spatial_features
