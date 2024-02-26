# À compléter
import torch
from torch import nn

class AlexNetDetect(nn.Module):
    def __init__(self):
        super(AlexNetDetect, self).__init__()

        dropout_prob = 0
        self.conv_1 = nn.Conv2d(1, 32, 5, 1, 1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(3, 2, 0)

        self.conv_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(3, 2, 0)

        self.conv_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.relu_3 = nn.ReLU()

        self.conv_4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.relu_4 = nn.ReLU()

        self.conv_5 = nn.Conv2d(64, 24, 3, 1, 1)
        self.bn_5 = nn.BatchNorm2d(24)
        self.relu_5 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool2d(2, 2, 0)

        self.linear_1 = nn.Linear(864, 254)
        self.bnlin_1 = nn.BatchNorm1d(254)
        self.relulin_1 = nn.ReLU()

        self.linear_2 = nn.Linear(254, 96)
        self.bnlin_2 = nn.BatchNorm1d(96)
        self.relulin_2 = nn.ReLU()

        self.linear_3 = nn.Linear(96, 21)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        N = x.shape[0]

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

        box = x[:, :12]
        class_score = x[:, 12:]

        box = self.sigmoid(torch.reshape(box, (N, 3, 4)))
        class_score = self.softmax(torch.reshape(class_score, (N, 3, 3)))

        x = torch.cat((box, class_score), 2)

        return x

class detectionLoss(nn.Module):
    def __init__(self):
        super(detectionLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        LossBox = self.mse(output[:, :, 0:4], target[:, :, 0:4])

        onehot = torch.nn.functional.one_hot(target[:, :, 4].to(torch.int64), num_classes=3).float()
        LossClass = self.ce(output[:, :, 4:], onehot)

        return LossBox + LossClass
