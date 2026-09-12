import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class EDSR(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale=4
    ):
        super().__init__()

        # First convolution
        self.head = nn.Conv2d(
            in_channels,
            features,
            3,
            padding=1
        )

        # Residual blocks
        self.body = nn.Sequential(
            *[
                ResidualBlock(features)
                for _ in range(num_blocks)
            ]
        )

        # Reconstruction
        self.body_conv = nn.Conv2d(
            features,
            features,
            3,
            padding=1
        )

        # 4x upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                features,
                features * (scale ** 2),
                3,
                padding=1
            ),
            nn.PixelShuffle(scale)
        )

        # Final output
        self.tail = nn.Conv2d(
            features,
            out_channels,
            3,
            padding=1
        )

    def forward(self, x):

        # Feature extraction
        x = self.head(x)

        # Residual learning
        residual = x
        x = self.body(x)
        x = self.body_conv(x)
        x = x + residual

        # 32x32 → 128x128
        x = self.upsample(x)

        # Produce 4-band HR image
        x = self.tail(x)

        return x


if __name__ == "__main__":

    model = EDSR()

    # Test input:
    # batch = 2
    # channels = 4
    # height = 32
    # width = 32

    test_input = torch.randn(2, 4, 32, 32)

    output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)