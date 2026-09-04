import torch
import torch.nn as nn


# ============================================================
# RESIDUAL BLOCK
# Standard EDSR residual block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, features=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.block(x)

        return x + residual


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class UpsampleBlock(nn.Module):

    def __init__(self, features=64, scale=4):

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
# STANDARD EDSR BASELINE
# ============================================================

class EDSRBaseline(nn.Module):

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
        # BODY
        # ----------------------------------------------------

        blocks = [
            ResidualBlock(features)
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