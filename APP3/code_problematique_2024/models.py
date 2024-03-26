# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
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
        self.word_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        self.encoder_rnn = nn.GRU(2, hidden_dim, n_layers, batch_first=True)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim, self.dict_size)
        self.sm = nn.Softmax(dim=2)
        self.to(device)

    def encoder(self, x):

        y, h = self.encoder_rnn(x.type(torch.float32))

        return y, h

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['out']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
            vec_emb = self.word_embedding(vec_in)
            vec_rnn, hidden = self.decoder_rnn(vec_emb, hidden)
            vec_fully = self.fc(vec_rnn)
            vec_sm = self.sm(vec_fully)

            idx = torch.argmax(vec_sm, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_sm.squeeze(1)

        return vec_out, hidden, None


    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden


class trajectory2seq_attn(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_attn, self).__init__()
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
        self.word_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        self.encoder_rnn = nn.GRU(2, hidden_dim, n_layers, batch_first=True)
        self.decoder_rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Couches pour attention
        self.att_combine = nn.Linear(2 * hidden_dim, hidden_dim)
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim, self.dict_size)
        self.sm = nn.Softmax(dim=2)
        self.to(device)

    def encoder(self, x):

        x = x.permute(0, 2, 1)
        y, h = self.encoder_rnn(x.type(torch.float32))

        return y, h

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # Attention

        vec_dot = torch.bmm(values, query.permute(0, 2, 1))
        vec_dot = vec_dot.squeeze(-1)
        attention_weights = torch.softmax(vec_dot, 1)

        attention_weights = attention_weights.unsqueeze(1)
        attention_output = torch.bmm(attention_weights, values)

        return attention_output, attention_weights

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['out']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage
        att_w = torch.zeros((batch_size, self.maxlen['in'], self.dict_size)).to(self.device)  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            vec_emb = self.word_embedding(vec_in)
            vec_rnn, hidden = self.decoder_rnn(vec_emb, hidden)

            att_out, att_w = self.attentionModule(vec_rnn, encoder_outs)
            vec_concat = torch.cat((vec_rnn, att_out), dim=2)
            vec_fully = self.att_combine(vec_concat)
            vec_fully = self.fc(vec_fully)

            idx = torch.argmax(vec_fully, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_fully.squeeze(1)

        return vec_out, hidden, att_w

    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn


class trajectory2seq_bi(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_bi, self).__init__()
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
        self.word_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        self.encoder_rnn = nn.GRU(maxlen['in'], hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_hidden = nn.Linear(2*hidden_dim, hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim, self.dict_size)
        self.sm = nn.Softmax(dim=2)
        self.to(device)

    def encoder(self, x):

        y, h = self.encoder_rnn(x.type(torch.float32))

        h2 = h.view(-1, 2*self.hidden_dim)
        h_out = self.fc_hidden(h2)
        h_out = h_out.view(1, y.shape[0], self.hidden_dim)

        return y, h_out

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['out']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
            vec_emb = self.word_embedding(vec_in)
            vec_rnn, hidden = self.decoder_rnn(vec_emb, hidden)
            vec_fully = self.fc(vec_rnn)
            vec_sm = self.sm(vec_fully)

            idx = torch.argmax(vec_sm, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_sm.squeeze(1)

        return vec_out, hidden, None


    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden


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
        self.word_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        self.encoder_rnn = nn.LSTM(2, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_hidden = nn.Linear(2 * hidden_dim * n_layers, n_layers * hidden_dim)

        # Couches pour attention
        self.att_combine = nn.Linear(2 * hidden_dim, hidden_dim)
        self.hidden2value = nn.Linear(2*hidden_dim, hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim, self.dict_size)
        self.sm = nn.Softmax(dim=2)
        self.to(device)

    def encoder(self, x):
        x = x.permute(0, 2, 1)
        y, h = self.encoder_rnn(x.type(torch.float32))

        h2 = h[0].view(y.shape[0], -1)

        h_out = self.fc_hidden(h2)
        h_out = h_out.view(self.n_layers, y.shape[0], self.hidden_dim)

        return y, h_out

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        v2 = values.reshape(-1, 2*self.hidden_dim)
        v_out = self.hidden2value(v2)
        v_out = v_out.view(values.shape[0], -1, self.hidden_dim)
        # Attention

        vec_dot = torch.bmm(v_out, query.permute(0, 2, 1))
        vec_dot = vec_dot.squeeze(-1)
        attention_weights = torch.softmax(vec_dot, 1)

        attention_weights = attention_weights.unsqueeze(1)
        attention_output = torch.bmm(attention_weights, v_out)
        return attention_output, attention_weights

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['out']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage
        att_w = torch.zeros((batch_size, self.maxlen['in'], self.dict_size)).to(self.device)  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            vec_emb = self.word_embedding(vec_in)
            vec_rnn, hidden = self.decoder_rnn(vec_emb, hidden)

            att_out, att_w = self.attentionModule(vec_rnn, encoder_outs)
            vec_concat = torch.cat((vec_rnn, att_out), dim=2)
            vec_fully = self.att_combine(vec_concat)
            vec_fully = self.fc(vec_fully)

            idx = torch.argmax(vec_fully, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_fully.squeeze(1)

        return vec_out, hidden, att_w

    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn

