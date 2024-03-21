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

            # shift points to positive
            seq_x = seq[0]
            seq_y = seq[1]
            if seq.min() < 0:
                offset = abs(np.floor(seq.min()))
                seq_x = [x + offset for x in seq_x]
                seq_y = [x + offset for x in seq_y]

            # Create padding np array
            pad_seq = np.zeros(self.max_len_seq - len(seq_x))
            #pad_seq[0] = -1
            seq_x = np.append(seq_x, pad_seq)
            seq_y = np.append(seq_y, pad_seq)

            self.data[i][1] = np.stack((seq_x, seq_y), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputTensor = torch.tensor(item[1])
        targetList = item[0]
        targetList = [self.symb2int[i] for i in targetList]
        targetTensor = torch.tensor(targetList)
        return inputTensor, targetTensor

    def visualisation(self, idx):
        input_sequence, target_sequence = self[idx]
        input_sequence = input_sequence.numpy()
        word = [self.int2symb[i] for i in target_sequence.tolist()]
        plt.plot(input_sequence[0], input_sequence[1], label='input sequence')
        plt.title('Visualization of sample ' + str(idx) + ' : ' + str(word) )
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))