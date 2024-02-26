import torch.nn as nn
from torch import flatten


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        self.fc1 = nn.Linear(28 * 28, 10)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.mise_comm1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(2)
        self.mise_comm2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lin = nn.Linear(2*7*7, 10)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # output = self.fc1(x.view(x.shape[0], -1))
        # output = self.fc2(x).view(x.shape[0], -1)
        a = self.mise_comm1(self.relu(self.batch_norm1(self.conv1(x))))
        b = self.mise_comm2(self.relu(self.batch_norm2(self.conv2(a))))
        output = self.lin(flatten(b, start_dim=1))
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
