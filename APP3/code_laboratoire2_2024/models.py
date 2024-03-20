# GRO722 Laboratoire 2
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Seq2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        # Encodeur

        # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        x_emb_fr = self.fr_embedding(x)
        x, h = self.encoder_layer(x_emb_fr)

        out = x
        hidden = h

        # ---------------------- Laboratoire 2 - Question 3 - Fin de la section à compléter -----------------

        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------   
            vec_emb = self.en_embedding(vec_in)
            vec_rnn, hidden = self.decoder_layer(vec_emb, hidden)
            vec_fully = self.fc(vec_rnn)

            idx = torch.argmax(vec_fully, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_fully.squeeze(1)

            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        return vec_out, hidden, None

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


class Seq2seq_attn(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq_attn, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        #Encodeur

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
        
        x_emb_fr = self.fr_embedding(x)
        x, h = self.encoder_layer(x_emb_fr)

        out = x
        hidden = h
        
        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # Attention

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        N = values.shape[0]
        He = values.shape[-1]
        vec_dot = torch.bmm(values, query.view(N, He, 1))
        vec_dot = vec_dot.squeeze(-1)
        #vec_dot = torch.tensordot(values, query, dims=[[2], [2]])
        #vec_dot = vec_dot.squeeze(dim=-1)
        #vec_dot = torch.sum(vec_dot, dim=2)
        attention_weights = torch.softmax(vec_dot, 1)

        attention_weights = attention_weights.unsqueeze(1)
        attention_output = torch.bmm(attention_weights, values)

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.max_len['fr'], self.max_len['en'])).to(self.device) # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

            vec_emb = self.en_embedding(vec_in)
            vec_rnn, hidden = self.decoder_layer(vec_emb, hidden)

            att_out, att_w = self.attentionModule(vec_rnn, encoder_outs)
            vec_concat = torch.cat((vec_rnn, att_out), dim=2)
            vec_fully = self.att_combine(vec_concat)
            vec_fully = self.fc(vec_fully)

            idx = torch.argmax(vec_fully, dim=-1)

            vec_in = idx
            vec_out[:, i, :] = vec_fully.squeeze(1)

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return vec_out, hidden, attention_weights


    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out,h)
        return out, hidden, attn
