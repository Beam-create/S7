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

        # Dict int2symb
        self.int2symb = dict()
        self.int2symb = {v:k for k, v in self.symb2int.items()}
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

        pad_word = ['<eos>']
        [pad_word.append('<pad>') for i in range(1, self.max_len_word)]
        pad_sequence = np.full((2, self.max_len_seq), -2)
        pad_sequence[0][0] = -1
        pad_sequence[1][0] = -1
        # print(len(pad_sequence))

        # Pad to length
        for i, data in enumerate(self.data):
            word = data[0]
            seq = data[1]

            # apply padding to word
            word.extend((pad_word[0:(self.max_len_word - len(word))]))
            self.data[i][0] = word

            # apply padding to points
            seq_min_val = -1*(np.floor(seq.min()))
            # offset all vals in seq of seq_min_val
            positive_seq = seq + seq_min_val
            self.data[i][1] = np.concatenate((positive_seq, pad_sequence[:, :(self.max_len_seq - len(seq[1]))]), axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_tensor = torch.tensor(item[1])
        target_tensor = item[0]
        return input_tensor, target_tensor

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