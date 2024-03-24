# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen: dict):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        self.encoder_layer = nn.GRU(self.maxlen['seq'], self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc_enc2decode = nn.Linear(2*self.hidden_dim, self.hidden_dim)

        self.decode_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        self.decoder_layer = nn.RNN(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        # Couches pour attention
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.to(self.device)

    def attnModule(self, query, values):
        query = self.hidden2query(query)
        N = values.shape[0]
        He = values.shape[-1]
        vec_dot = torch.bmm(values, query.view(N, He, 1))
        vec_dot = vec_dot.squeeze(-1)
        att_weights  = torch.softmax(vec_dot,dim=-1)
        att_weights = att_weights.unsqueeze(1)
        att_out = torch.bmm(att_weights, values)

    def decoder(self, encoder_outs, hidden):
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, self.maxlen['trad'], self.dict_size)).to(self.device) # Vecteur de sortie du décodage
        hidden = self.fc_enc2decode(hidden.view(batch_size, self.hidden_dim*2))
        hidden = self.relu(hidden).unsqueeze(0)
        for i in range(self.maxlen['trad']):
            vec_emb = self.decode_embedding(vec_in)
            vec_rnn, hidden = self.decoder_layer(vec_emb, hidden)
            vec_fully = self.fc(vec_rnn)
            vec_fully = self.relu(vec_fully)
            vec_sm = self.sm(vec_fully)
            idx = torch.argmax(vec_sm, dim=-1)
            vec_in = idx
            vec_out[:, i, :] = vec_sm.squeeze(1)
        return vec_out, hidden, None

    def forward(self, x):
        # Input encoding
        out, h = self.encoder_layer(x.type(torch.float32))
        out, hidden, attn = self.decoder(out, h)

        return out, hidden, attn
    

