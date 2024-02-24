import torch.nn as nn
import torch

from torch import Tensor

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: nn.Module=None, stride:int =1)->None:
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor)-> Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x
class ResNet(nn.Module): #[3, 2, 2, 2]
    def __init__(self, img_channels: int=1, num_classes: int= 3)->None:
        super(ResNet, self).__init__()
        self.in_channels = 9
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, padding=0, stride=1)
        self.conv_end = nn.Conv2d(self.in_channels*8, num_classes, kernel_size=1, padding=0, stride=1)

        self.block1 = BasicBlock(self.in_channels*1, self.in_channels*1)
        self.block2 = BasicBlock(self.in_channels*1, self.in_channels*1)
        self.block3 = BasicBlock(self.in_channels*1, self.in_channels*2, stride=2,
                       downsample=self.identity_downsample(self.in_channels*1, self.in_channels*2))
        self.block4 = BasicBlock(self.in_channels*2, self.in_channels*2)
        self.block5 = BasicBlock(self.in_channels*2, self.in_channels*4, stride=2,
                       downsample=self.identity_downsample(self.in_channels*2, self.in_channels*4))
        self.block6 = BasicBlock(self.in_channels*4, self.in_channels*4)
        self.block7 = BasicBlock(self.in_channels*4, self.in_channels*8, stride=2,
                       downsample=self.identity_downsample(self.in_channels*4, self.in_channels*8))
        self.block8 = BasicBlock(self.in_channels*8, self.in_channels*8)
    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        )

    def forward(self, x: Tensor)-> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        self.relu(x)
        x = self.maxpool(x)
        # print(f'maxpool {x.shape}')

        # Block1
        x = self.block1.forward(x)
        x = self.relu(x)
        # print(f'Block1 {x.shape}')

        # Block2
        x = self.block2.forward(x)
        x = self.relu(x)
        # print(f'Block2 {x.shape}')

        # Block3
        x = self.block3.forward(x)
        x = self.relu(x)
        # print(f'Block3 {x.shape}')

        # Block4
        x = self.block4.forward(x)
        x = self.relu(x)
        # print(f'Block4 {x.shape}')

        # Block5
        x = self.block5.forward(x)
        x = self.relu(x)
        # print(f'Block5 {x.shape}')

        # Block6
        x = self.block6.forward(x)
        x = self.relu(x)
        # print(f'Block6 {x.shape}')

        # Block7
        x = self.block7.forward(x)
        x = self.relu(x)
        # print(f'Block7 {x.shape}')

        # Block8
        x = self.block8.forward(x)
        x = self.relu(x)
        # print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = self.conv_end(x)
        # print(f'conv_end {x.shape}')
        x = torch.flatten(x, 1)
        return x

# if __name__ == '__main__':
#     tensor = torch.rand([1, 1, 53, 53])
#     model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=3)
#     print(model)
#
#     # Total parameters and trainable parameters.
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"{total_params:,} total parameters.")
#     total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{total_trainable_params:,} training parameters.")
#     output = model(tensor)