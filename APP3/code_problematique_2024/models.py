# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class trajectory2seq_attn_bi(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_attn_bi, self).__init__()
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
        self.encoder_rnn = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=0.35)

        self.word_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        self.decoder_rnn = nn.RNN(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Couches pour attention
        self.hidden2query = nn.Linear(self.hidden_dim, 2*self.hidden_dim)

        # Couche encoder 2 decoder
        self.fc_enc2dec = nn.Linear(2*self.hidden_dim*self.n_layers, self.hidden_dim*self.n_layers)

        # Couche dense pour la sortie
        self.fc_att_comb = nn.Linear(3*self.hidden_dim, self.dict_size)
        self.to(self.device)

    def encoder(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.shape[0]
        # y, (h, c) = self.encoder_rnn(x.type(torch.float32))
        y, h = self.encoder_rnn(x.type(torch.float32))
        h = h.view(batch_size, -1)
        h = self.fc_enc2dec(h)
        h = h.view(self.n_layers, batch_size, self.hidden_dim)
        return y, h

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query.squeeze(1)).unsqueeze(-1)

        # Attention
        vec_dot = torch.bmm(values, query) # Produit vectoriel
        attention_weights = torch.softmax(vec_dot, 1)
        attention_output = torch.bmm(attention_weights.permute(0,2,1), values) # Multiplication matrice

        return attention_output, attention_weights

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['trad']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage
        att_w = torch.zeros((batch_size, max_len, self.maxlen['seq'])).to(self.device)  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            vec_emb = self.word_embedding(vec_in)
            vec_rnn, hidden = self.decoder_rnn(vec_emb, hidden)

            att_out, att_weight = self.attentionModule(vec_rnn, encoder_outs)
            vec_concat = torch.cat((vec_rnn, att_out), dim=2)
            vec_fully = self.fc_att_comb(vec_concat)

            idx = torch.argmax(vec_fully, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_fully.squeeze(1)
            att_w[:, i, :] = att_weight.squeeze(-1)

        return vec_out, hidden, att_w

    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn



