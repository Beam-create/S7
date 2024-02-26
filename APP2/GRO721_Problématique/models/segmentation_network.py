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
        self.hidden = None
        # Image padding
        self.target_size = (64, 64)
        # Calculate the padding values
        self.pad_height = max(0, self.target_size[0] - 53)
        self.pad_width = max(0, self.target_size[1] - 53)
        self.sz = 4
        self.relu = nn.ReLU()
        # Down 1
        self.conv_1_1 = nn.Conv2d(img_channels, 2**self.sz, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(2**self.sz, 2**self.sz, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2**self.sz)

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv_2_1 = nn.Conv2d(2**(self.sz), 2**(self.sz+1), kernel_size=3, padding=1)
        self.conv_2_2 = nn.Conv2d(2**(self.sz+1), 2**(self.sz+1), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2**(self.sz+1))

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv_3_1 = nn.Conv2d(2**(self.sz+1), 2**(self.sz+2), kernel_size=3, padding=1)
        self.conv_3_2 = nn.Conv2d(2**(self.sz+2), 2**(self.sz+2), kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2**(self.sz+2))

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv_4_1 = nn.Conv2d(2**(self.sz+2), 2**(self.sz+3), kernel_size=3, padding=1)
        self.conv_4_2 = nn.Conv2d(2**(self.sz+3), 2**(self.sz+3), kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2**(self.sz+3))

        # Down 5
        self.maxpool_5 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv_5_1 = nn.Conv2d(2**(self.sz+3), 2**(self.sz+4), kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(2 ** (self.sz + 4))
        self.conv_5_2 = nn.Conv2d(2**(self.sz+4), 2**(self.sz+3), kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(2 ** (self.sz + 3))

        # Up 6
        self.upsample_6 = nn.ConvTranspose2d(2**(self.sz+3), 2**(self.sz+3), kernel_size=2, padding=0, stride=2)
        self.conv_6_1 = nn.Conv2d(2**(self.sz+4), 2**(self.sz+3), kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(2 ** (self.sz + 3))
        self.conv_6_2 = nn.Conv2d(2**(self.sz+3), 2**(self.sz+2), kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(2 ** (self.sz + 2))

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(2**(self.sz+2), 2**(self.sz+2), kernel_size=2, padding=0, stride=2)
        self.conv_7_1 = nn.Conv2d(2**(self.sz+3), 2**(self.sz+2), kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(2 ** (self.sz + 2))
        self.conv_7_2 = nn.Conv2d(2**(self.sz+2), 2**(self.sz+1), kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(2 ** (self.sz + 1))

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(2**(self.sz+1), 2**(self.sz+1), kernel_size=2, padding=0, stride=2)
        self.conv_8_1 = nn.Conv2d(2**(self.sz+2), 2**(self.sz+1), kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(2**(self.sz+1))
        self.conv_8_2 = nn.Conv2d(2**(self.sz+1), 2**self.sz, kernel_size=3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(2 ** (self.sz))

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(2**(self.sz), 2**(self.sz), kernel_size=2, padding=0, stride=2)
        self.conv_9_1 = nn.Conv2d(2**(self.sz+1), 2**(self.sz), kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(2 ** (self.sz))
        self.conv_9_2 = nn.Conv2d(2**(self.sz), 2**(self.sz), kernel_size=3, padding=1)
        self.conv_end = nn.Conv2d(2**(self.sz), num_classes, kernel_size=1, padding=0)

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
        # Pad image to fit to size 64x64
        y, pads = self.pad_to(x)
        # y = self.pad_to(x, 64)
        # Down 1
        # y = self.conv_1_1(padded_image)
        y = self.conv_1_1(y)
        y = self.bn1(y)
        # print(y.size())
        y = self.relu(y)
        y = self.conv_1_2(y)
        y = self.bn1(y)
        y_1 = self.relu(y)

        # Down 2
        y = self.maxpool_2(y_1)
        y = self.conv_2_1(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv_2_2(y)
        y = self.bn2(y)
        y_2 = self.relu(y)

        # Down 3
        y = self.maxpool_3(y_2)
        y = self.conv_3_1(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv_3_2(y)
        y = self.bn3(y)
        y_3 = self.relu(y)

        # Down 4
        y = self.maxpool_4(y_3)
        y = self.conv_4_1(y)
        y = self.bn4(y)
        y = self.relu(y)
        y = self.conv_4_2(y)
        y = self.bn4(y)
        y_4 = self.relu(y)

        # Down 5
        y = self.maxpool_5(y_4)

        y = self.conv_5_1(y)
        y = self.bn5_1(y)
        y = self.relu(y)
        y = self.conv_5_2(y)
        y = self.bn5_2(y)
        y = self.relu(y)

        # Up 6
        y = self.upsample_6(y)
        y = torch.cat([y, y_4], 1)
        y = self.conv_6_1(y)
        y = self.bn6_1(y)
        y = self.relu(y)
        y = self.conv_6_2(y)
        y = self.bn6_2(y)
        y = self.relu(y)

        # Up 7
        y = self.upsample_7(y)
        y = torch.cat([y, y_3], 1)
        y = self.conv_7_1(y)
        y = self.bn7_1(y)
        y = self.relu(y)
        y = self.conv_7_2(y)
        y = self.bn7_2(y)
        y = self.relu(y)

        # Up 8
        y = self.upsample_8(y)
        y = torch.cat([y, y_2], 1)
        y = self.conv_8_1(y)
        y = self.bn8_1(y)
        y = self.relu(y)
        y = self.conv_8_2(y)
        y = self.bn8_2(y)
        y = self.relu(y)

        # Up 9
        y = self.upsample_9(y)
        y = torch.cat([y, y_1], 1)
        y = self.conv_9_1(y)
        y = self.bn9(y)
        y = self.relu(y)
        y = self.conv_9_2(y)
        y = self.bn9(y)
        y = self.relu(y)
        y = self.conv_end(y)
        y = nn.Sigmoid()(y)
        y = self.unpad(y, pads)
        # y = self.unpad(y, 53)
        return y