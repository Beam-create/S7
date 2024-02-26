import torch
import torch.nn as nn
from torch import Tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        down = self.conv(x)
        pool = self.pool(down)
        return down, pool

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, img_channels: int=1, num_classes: int=4):
        super(Unet, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.down5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
        )

    def pad_to(self, x: Tensor, stride: int=16):
        h, w = x.shape[-2:]
        if h % stride > 0:
            new_h = h + stride - h % stride
        else:
            new_h = h
        if w % stride > 0:
            new_w = w + stride - w % stride
        else:
            new_w = w
        lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
        lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
        pads = (lw, uw, lh, uh)

        # zero-padding by default.
        # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
        out = nn.functional.pad(x, pads, "constant", 0)
        return out, pads

    # def pad_to(self, x: Tensor, new_size: int):
    #     new_img = torch.zeros((x.shape[0], x.shape[1], new_size, new_size)).to(x.device)
    #     new_img[:, :, :x.shape[2], :x.shape[3]] = x
    #     return new_img

    # def unpad(self, x: Tensor, original_size: int):
    #     return x[:, :, :original_size, :original_size]

    def unpad(self, x: Tensor, pad: tuple):
        if pad[2] + pad[3] > 0:
            x = x[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            x = x[:, :, :, pad[0]:-pad[1]]
        return x

    def forward(self, x: Tensor):
        x1 = self.down1(x)

        x2 = self.down2(x1)

        x3 = self.down3(x2)

        x4 = self.down4(x3)

        x5 = self.down5(x4)

        u1 = torch.cat([x5, x4], 1)
        u1 = self.up1(u1)

        u2 = torch.cat([u1, x3], dim=1)
        u2 = self.up2(u2)

        u3 = torch.cat([u2, x2], dim=1)
        u3 = self.up3(u3)

        u4 = torch.cat([u3, x1], dim=1)
        u4 = self.up4(u4)

        y = self.sigmoid(u4)

        return y


class SegmentationLoss(nn.Module):
    def init(self):
        super(SegmentationLoss, self).init()

    def forward(self, output, target):
        one_hot_encoded = torch.nn.functional.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
        loss = nn.BCELoss()(output, one_hot_encoded)
        return loss
