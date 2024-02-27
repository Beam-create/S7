import torch
import torch.nn as nn
from torch import Tensor


class Unet(nn.Module):
    def __init__(self, img_channels: int=1, num_classes: int=4):
        super(Unet, self).__init__()
        self.hidden = None
        self.sz = 3

        self.down1 = nn.Sequential(
            # Down 1
            nn.Conv2d(img_channels, 2**self.sz, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** self.sz),
            nn.ReLU(),
            nn.Conv2d(2**self.sz, 2**self.sz, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** self.sz),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            # Down 2
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(2**self.sz, 2**(self.sz+1), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+1)),
            nn.ReLU(),
            nn.Conv2d(2**(self.sz+1), 2**(self.sz+1), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+1)),
            nn.ReLU()
        )

        self.down3 = nn.Sequential(
            # Down 3
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(2**(self.sz+1), 2**(self.sz+2), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+2)),
            nn.ReLU(),
            nn.Conv2d(2**(self.sz+2), 2**(self.sz+2), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+2)),
            nn.ReLU()
        )

        self.down4 = nn.Sequential(
            # Down 4
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(2**(self.sz+2), 2**(self.sz+3), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+3)),
            nn.ReLU(),
            nn.Conv2d(2**(self.sz+3), 2**(self.sz+3), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+3)),
            nn.ReLU()
        )

        self.down5 = nn.Sequential(
            # Down 5
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(2**(self.sz+3), 2**(self.sz+4), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+4)),
            nn.ReLU(),
            nn.Conv2d(2 ** (self.sz + 4), 2 ** (self.sz + 3), kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** (self.sz + 3)),
            nn.ReLU()
        )

        self.upsample_1 = nn.ConvTranspose2d(2**(self.sz+3), 2**(self.sz+3), kernel_size=2, padding=0, stride=2)
        self.up1 = nn.Sequential(
            # Down 6
            nn.Conv2d(2**(self.sz+4), 2**(self.sz+3), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+3)),
            nn.ReLU(),
            nn.Conv2d(2 ** (self.sz + 3), 2 ** (self.sz + 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** (self.sz + 2)),
            nn.ReLU()
        )

        self.upsample_2 = nn.ConvTranspose2d(2 ** (self.sz + 2), 2 ** (self.sz + 2), kernel_size=2, padding=0, stride=2)
        self.up2 = nn.Sequential(
            # Down 7
            nn.Conv2d(2**(self.sz+3), 2**(self.sz+2), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+2)),
            nn.ReLU(),
            nn.Conv2d(2 ** (self.sz + 2), 2 ** (self.sz + 1), kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** (self.sz + 1)),
            nn.ReLU()
        )

        self.upsample_3 = nn.ConvTranspose2d(2 ** (self.sz + 1), 2 ** (self.sz + 1), kernel_size=2, padding=0, stride=2)
        self.up3 = nn.Sequential(
            # Down 8
            nn.Conv2d(2**(self.sz+2), 2**(self.sz+1), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz+1)),
            nn.ReLU(),
            nn.Conv2d(2 ** (self.sz + 1), 2 ** (self.sz), kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** (self.sz)),
            nn.ReLU()
        )

        self.upsample_4 = nn.ConvTranspose2d(2 ** (self.sz), 2 ** (self.sz), kernel_size=2, padding=0, stride=2)
        self.up4 = nn.Sequential(
            # Down 9
            nn.Conv2d(2**(self.sz+1), 2**(self.sz), kernel_size=3, padding=1),
            nn.BatchNorm2d(2**(self.sz)),
            nn.ReLU(),
            nn.Conv2d(2 ** (self.sz), 2 ** (self.sz), kernel_size=3, padding=1),
            nn.BatchNorm2d(2 ** (self.sz)),
            nn.ReLU(),
            nn.Conv2d(2 ** (self.sz), 4, kernel_size=3, padding=1)
        )

    # def pad_to(self, x: Tensor, stride: int=16):
    #     h, w = x.shape[-2:]
    #     if h % stride > 0:
    #         new_h = h + stride - h % stride
    #     else:
    #         new_h = h
    #     if w % stride > 0:
    #         new_w = w + stride - w % stride
    #     else:
    #         new_w = w
    #     lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    #     lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    #     pads = (lw, uw, lh, uh)
    #
    #     # zero-padding by default.
    #     # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    #     out = nn.functional.pad(x, pads, "constant", 0)
    #     return out, pads

    def pad_to(self, x: Tensor, new_size: int):
        new_img = torch.zeros((x.shape[0], x.shape[1], new_size, new_size)).to(x.device)
        new_img[:, :, :x.shape[2], :x.shape[3]] = x
        return new_img

    def unpad(self, x: Tensor, original_size: int):
        return x[:, :, :original_size, :original_size]

    # def unpad(self, x: Tensor, pad: tuple):
    #     if pad[2] + pad[3] > 0:
    #         x = x[:, :, pad[2]:-pad[3], :]
    #     if pad[0] + pad[1] > 0:
    #         x = x[:, :, :, pad[0]:-pad[1]]
    #     return x

    def forward(self, x: Tensor):
        # Pad image to fit to size 64x64
        # y, pads = self.pad_to(x)
        y = self.pad_to(x, 64)
        # Down 1
        y1 = self.down1(y)
        y2 = self.down2(y1)
        y3 = self.down3(y2)
        y4 = self.down4(y3)
        y = self.down5(y4)
        y = self.upsample_1(y)
        y = torch.cat([y, y4], 1)
        y = self.up1(y)
        y = self.upsample_2(y)
        y = torch.cat([y, y3], 1)
        y = self.up2(y)
        y = self.upsample_3(y)
        y = torch.cat([y, y2], 1)
        y = self.up3(y)
        y = self.upsample_4(y)
        y = torch.cat([y, y1], 1)
        y = self.up4(y)
        y = nn.Sigmoid()(y)
        # y = self.unpad(y, pads)
        y = self.unpad(y, 53)
        # return None
        return y

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

    def forward(self, output, target):
        one_hot_encoded = torch.nn.functional.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
        loss = nn.BCELoss()(output, one_hot_encoded)
        return loss