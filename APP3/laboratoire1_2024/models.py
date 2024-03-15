# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        self.batch_first = True
        # self.rnn = nn.RNN(1, n_hidden, n_layers, batch_first=self.batch_first)
        self.rnn = nn.GRU(input_size=1, hidden_size=n_hidden, num_layers=n_layers, batch_first=self.batch_first)
        # self.rnn = nn.LSTM(input_size=1, hidden_size=n_hidden, num_layers=n_layers, batch_first=self.batch_first)
        self.fc = nn.Linear(25, 1)
        self.activation = nn.Tanh()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
    
    def forward(self, x, h=None):

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        x = x.unsqueeze(-1)
        x, h = self.rnn(x, h)
        out = self.fc(x)
        out = self.activation(out)
        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
        # print(out.size(), h.size())
        return out, h

# if __name__ == '__main__':
    # x = torch.zeros((50,199,2)).float()
    # model = Model(25)
    # out, h = model(x)