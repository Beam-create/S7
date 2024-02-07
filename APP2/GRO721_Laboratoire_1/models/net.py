import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        self.fc1 = nn.Linear(28 * 28, 10)
        self.fc2 = nn.Conv2d(1, 10, kernel_size=28)
        self.c1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1)
        self.c2 = nn.BatchNorm2d(4)
        self.c3 = nn.ReLU()
        self.c4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(4,2, kernel_size=3, padding=1, stride=1)
        self.c6 = nn.BatchNorm2d(2)
        self.c7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c8 = nn.Linear(2*7*7, 10)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        a = self.c1(x)
        b = self.c2(a)
        c = self.c3(b)
        d = self.c4(c)
        e = self.c5(d)
        f = self.c6(e)
        g = self.c7(f)
        #print("Shape de g", g.shape)
        h = torch.flatten(g, start_dim=1)
        #print("Shape de h", h.shape)
        output = self.c8(h.view(x.shape[0], -1))

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
