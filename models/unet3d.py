import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SE3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHWD
        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)

        x1 = nn.functional.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3DTimeAsChannel(nn.Module):
    """
    时间维度视为通道维度：
    输入形状: (B, C_in_total, Z, Y, X)
    其中 C_in_total = input_steps * in_channels_per_step
    输出形状: (B, H, Z, Y, X), H = pred_steps * 6 (T, U, V, W, K, NUT)
    """

    def __init__(
        self,
        in_channels_total: int,
        pred_steps: int,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels_total = in_channels_total
        self.pred_steps = pred_steps

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels_total, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8 // factor)
        self.se_down1 = SE3D(base_channels * 2)
        self.se_down2 = SE3D(base_channels * 4)
        self.se_down3 = SE3D(base_channels * 8 // factor)

        bottleneck_channels = base_channels * 8 // factor
        self.bottleneck = DoubleConv(bottleneck_channels, bottleneck_channels)
        self.se_bottleneck = SE3D(bottleneck_channels)

        self.up1 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up2 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up3 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, pred_steps * 6)  # 输出 pred_steps 帧，每帧 6 通道

    def forward(self, x):
        # x: (B, C_in_total, Z, Y, X)
        x1 = self.inc(x)
        x2 = self.se_down1(self.down1(x1))
        x3 = self.se_down2(self.down2(x2))
        x4 = self.se_down3(self.down3(x3))
        xb = self.se_bottleneck(self.bottleneck(x4))

        x = self.up1(xb, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)  # (B, pred_steps * 6, Z, Y, X)
        return logits