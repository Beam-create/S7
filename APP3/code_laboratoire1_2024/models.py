# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        #self.rnn = nn.RNN(1, n_hidden, n_layers, batch_first=True)
        #self.rnn = nn.GRU(1, n_hidden, n_layers, batch_first=True)
        self.rnn = nn.LSTM(1, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, 1)
        self.activation = nn.Tanh()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
    
    def forward(self, x, h=None):

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        x = x.unsqueeze(-1)
        x ,h = self.rnn(x, h)
        x = self.linear(x)
        x = self.activation(x)

        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
        return x, h

if __name__ == '__main__':
    x = torch.zeros((100,2,1)).float()
    model = Model(25)
    print(model(x))