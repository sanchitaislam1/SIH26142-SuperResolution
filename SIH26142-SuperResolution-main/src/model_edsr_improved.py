import torch
import torch.nn as nn


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, features=64, reduction=16):

        super().__init__()

        hidden = max(features // reduction, 4)

        self.attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                features,
                hidden,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                features,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.attention(x)

        return x * weights


# ============================================================
# IMPROVED RESIDUAL BLOCK
# ============================================================

class ImprovedResidualBlock(nn.Module):

    def __init__(
        self,
        features=64,
        residual_scale=0.1
    ):

        super().__init__()

        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        # Channel attention
        self.channel_attention = ChannelAttention(
            features=features,
            reduction=16
        )

    def forward(self, x):

        residual = self.conv1(x)

        residual = self.relu(residual)

        residual = self.conv2(residual)

        # Channel attention
        residual = self.channel_attention(residual)

        # Residual scaling
        residual = residual * self.residual_scale

        return x + residual


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class UpsampleBlock(nn.Module):

    def __init__(
        self,
        features=64,
        scale=4
    ):

        super().__init__()

        layers = []

        if scale == 4:

            for _ in range(2):

                layers.append(
                    nn.Conv2d(
                        features,
                        features * 4,
                        kernel_size=3,
                        padding=1
                    )
                )

                layers.append(
                    nn.PixelShuffle(2)
                )

        elif scale == 2:

            layers.append(
                nn.Conv2d(
                    features,
                    features * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

        else:

            raise ValueError(
                "Only scale 2 and scale 4 are supported."
            )

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):

        return self.upsample(x)


# ============================================================
# IMPROVED EDSR
# ============================================================

class EDSRImproved(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale=4
    ):

        super().__init__()

        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # RESIDUAL BODY
        # ----------------------------------------------------

        blocks = [
            ImprovedResidualBlock(
                features=features,
                residual_scale=0.1
            )
            for _ in range(num_blocks)
        ]

        self.body = nn.Sequential(
            *blocks,

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            )
        )

        # ----------------------------------------------------
        # UPSAMPLING
        # ----------------------------------------------------

        self.upsample = UpsampleBlock(
            features=features,
            scale=scale
        )

        # ----------------------------------------------------
        # TAIL
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # Head
        x = self.head(x)

        # Residual body
        residual = self.body(x)

        # Global residual connection
        x = x + residual

        # Upsampling
        x = self.upsample(x)

        # Output
        x = self.tail(x)

        return x