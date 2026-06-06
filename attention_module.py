import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Computes spatial attention from intermediate feature maps.
    This helps highlight suspicious lesion regions before global pooling.
    """
    def __init__(self, in_channels, is_3d=False):
        super(SpatialAttention, self).__init__()

        self.is_3d = is_3d

        if is_3d:
            Conv = nn.Conv3d
        else:
            Conv = nn.Conv2d

        self.attention_conv = Conv(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (B, C, D, H, W)
        Returns:
            weighted_x: Attention-weighted features
            attention_map: The raw attention map for visualization
        """
        attention_logits = self.attention_conv(x)
        attention_map = self.sigmoid(attention_logits)

        weighted_x = x * attention_map
        return weighted_x, attention_map

class IntegratedFusionAttention(nn.Module):
    """
    Combines the fusion module and attention module.
    Outputs a single feature vector and stores the attention map.
    """
    def __init__(self, fusion_module, attention_module):
        super(IntegratedFusionAttention, self).__init__()
        self.fusion = fusion_module
        self.attention = attention_module

        if self.fusion.is_3d:
            self.pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        _, spatial_features = self.fusion(x)
        weighted_spatial, attention_map = self.attention(spatial_features)

        pooled = self.pool(weighted_spatial)
        pooled = pooled.view(pooled.size(0), -1)

        fused_features = self.fusion.projection(pooled)
        return fused_features, attention_map
