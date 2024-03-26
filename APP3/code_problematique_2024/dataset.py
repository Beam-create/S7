import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename, device='cuda'):
        self.device = device

        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        self.symb2int = {start_symbol:0, stop_symbol:1, pad_symbol:2}
        cpt_symb = 3

        # Création du dictionnaire
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        for i, letter in enumerate(alphabet, start=3):
            self.symb2int[letter] = i
            cpt_symb += 1

        # Extraction des symboles
        for i, data in enumerate(self.data):
            word = data[0]
            car_seq = []
            for symb in word:
                car_seq.extend(symb)
            data[0] = car_seq

        self.int2symb = dict()
        self.int2symb = {v: k for k, v in self.symb2int.items()}

        self.dict_size = len(self.int2symb)

        # Ajout du padding aux séquences
        self.max_len_word = 0
        self.max_len_seq = 0

        # find the max length
        for i, data in enumerate(self.data):
            word = data[0]
            seq = data[1]

            # Update word max length
            if len(word) > self.max_len_word:
                self.max_len_word = len(word)

            # Update sequence max length
            if seq.shape[1] > self.max_len_seq:
                self.max_len_seq = seq.shape[1]

        self.max_len_word += 1
        self.max_len_seq += 1

        pad_word = ['<eos>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']

        # Pad to length
        for i, data in enumerate(self.data):
            word = data[0]
            seq = data[1]

            # apply padding to word
            word.extend((pad_word[0:(self.max_len_word - len(word))]))
            self.data[i][0] = word

            # normalise input between 0 and 1
            min_val = np.min(seq)
            max_val = np.max(seq)
            scaled_matrix = (seq - min_val) / (max_val - min_val)

            # Apply padding to input
            seq_x = scaled_matrix[0]
            seq_y = scaled_matrix[1]

            # Create padding np array
            pad_seq_y = np.ones(self.max_len_seq - len(seq_y)) * 2
            pad_seq_x = np.ones(self.max_len_seq - len(seq_x)) * 2
            seq_x = np.append(seq_x, pad_seq_x)
            seq_y = np.append(seq_y, pad_seq_y)

            self.data[i][1] = np.stack((seq_x, seq_y), axis=0)

        # Return only 5 word
        #data = self.data[0:3]
        #self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputTensor = torch.tensor(item[1]).to(self.device)
        targetList = item[0]
        targetList = [self.symb2int[i] for i in targetList]
        targetTensor = torch.tensor(targetList).to(self.device)
        return inputTensor, targetTensor

    def visualisation(self, idx):
        input_sequence, target_sequence = self[idx]
        input_sequence = input_sequence.cpu().numpy()
        word = [self.int2symb[i] for i in target_sequence.cpu().tolist()]
        plt.plot(input_sequence[0], input_sequence[1], label='input sequence')
        plt.title('Visualization of sample ' + str(idx) + ' : ' + str(word) )
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(3):
        a.visualisation(np.random.randint(0, len(a)))