import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(AlexNet, self).__init__()

        dropout_prob = 0

        self.conv_1 = nn.Conv2d(input_channels, 32, 11, 2, 3)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout2d(dropout_prob)
        self.maxpool_1 = nn.MaxPool2d(3, 1, 1)

        self.conv_2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout2d(dropout_prob)
        self.maxpool_2 = nn.MaxPool2d(3, 2, 1)

        self.conv_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.relu_3 = nn.ReLU(inplace=True)
        self.dropout_3 = nn.Dropout2d(dropout_prob)
        self.conv_4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.relu_4 = nn.ReLU(inplace=True)
        self.dropout_4 = nn.Dropout2d(dropout_prob)
        self.conv_5 = nn.Conv2d(64, 16, 3, 1, 1)
        self.bn_5 = nn.BatchNorm2d(16)
        self.relu_5 = nn.ReLU(inplace=True)
        self.dropout_5 = nn.Dropout2d(dropout_prob)
        self.maxpool_3 = nn.MaxPool2d(2, 2, 0)

        self.linear_1 = nn.Linear(576, 96)
        self.bnlin_1 = nn.BatchNorm1d(96)
        self.relulin_1 = nn.ReLU(inplace=True)
        self.dropoutlin_1 = nn.Dropout(dropout_prob)
        self.linear_2 = nn.Linear(96, 32)
        self.bnlin_2 = nn.BatchNorm1d(32)
        self.relulin_2 = nn.ReLU(inplace=True)
        self.dropoutlin_2 = nn.Dropout(dropout_prob)
        self.linear_3 = nn.Linear(32, n_classes)


    def forward(self, x):
        # AlexNet
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.relu_5(x)

        x = self.maxpool_3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.linear_1(x)
        x = self.bnlin_1(x)
        x = self.relulin_1(x)

        x = self.linear_2(x)
        x = self.bnlin_2(x)
        x = self.relulin_2(x)

        x = self.linear_3(x)

        return x
