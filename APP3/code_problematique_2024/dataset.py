import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        self.symb2int = {start_symbol:0, stop_symbol:1, pad_symbol:2}
        cpt_symb = 3

        # Extraction des symboles
        for i, data in enumerate(self.data):
            word = data[0]
            car_seq = []
            for symb in word:
                if symb not in self.symb2int:
                    self.symb2int[symb] = cpt_symb
                    cpt_symb += 1
                car_seq.extend(symb)
            data[0] = car_seq

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
            data[0] = word

            # apply padding to points
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputTensor = torch.tensor(item[1])
        targetTensor = item[0]
        return inputTensor, targetTensor

    def visualisation(self, idx):
        input_sequence, target_sequence = self[idx]
        input_sequence = input_sequence.numpy()

        plt.plot(input_sequence[0], input_sequence[1], label='input sequence')
        plt.title('Visualization of sample ' + str(idx) + ' : ' + str(target_sequence) )
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))