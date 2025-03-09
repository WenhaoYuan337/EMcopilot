import torch
import torch.nn as nn
from segmentation_models_pytorch import UnetPlusPlus
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None
        self.channels = channels
        self.reduction = reduction

    def forward(self, x):
        batch, channels, _, _ = x.size()
        if self.fc is None or self.channels != channels:
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // self.reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // self.reduction, channels, bias=False),
                nn.Sigmoid()
            ).to(x.device)
            self.channels = channels

        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_attention(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_map)
        return x * spatial_out

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((np.log2(channels) / gamma) + b))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class AttentionUNetPlusPlus(nn.Module):
    def __init__(self, encoder_name="resnet18", classes=2, attention_type="SE"):
        super(AttentionUNetPlusPlus, self).__init__()
        self.unet = UnetPlusPlus(encoder_name=encoder_name, classes=classes)
        self.attention_type = attention_type

        if self.attention_type == "SE":
            self.attention = nn.ModuleList([SEBlock(ch) for ch in [64, 128, 256, 512]])
        elif self.attention_type == "CBAM":
            self.attention = nn.ModuleList([CBAM(ch) for ch in [64, 128, 256, 512]])
        elif self.attention_type == "ECA":
            self.attention = nn.ModuleList([ECABlock(ch) for ch in [64, 128, 256, 512]])
        else:
            raise NotImplementedError(f"Attention type {self.attention_type} not implemented.")

    def forward(self, x):
        features = self.unet.encoder(x)
        for i in range(len(features)):
            if i < len(self.attention):
                features[i] = self.attention[i](features[i])
        output = self.unet.decoder(*features)
        return self.unet.segmentation_head(output)
