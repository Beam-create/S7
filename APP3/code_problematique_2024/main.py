# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    training = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 3221                # Pour répétabilité
    n_workers = 2           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    batch_size = 32
    lr = 0.01
    n_epochs = 50

    n_hidden = 5
    n_layers = 1

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print('Using device:', device)

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords(filename='data_trainval.p')
    shuffle_dataset = True
    validation_split = 0.8
    n_train_samp = int(len(dataset) * validation_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    # Séparation de l'ensemble de données (entraînement et validation)

    print('Number of training examples:', len(dataset_train))
    print('Number of validation examples:', len(dataset_val))
    print(f'Epochs: {n_epochs}, Batch size: {batch_size}, Learning rate: {lr}')

    # Instanciation du model
    model = trajectory2seq(n_layers=n_layers, \
                           hidden_dim=n_hidden, device=device, symb2int=dataset.symb2int,\
                           int2symb=dataset.int2symb, dict_size=dataset.dict_size, \
                           maxlen={'seq':dataset.max_len_seq,'trad':dataset.max_len_word})
    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))

    # Initialisation des variables
    best_val_loss = np.inf # pour sauvegarder le meilleur model

    if training:
        if learning_curves:
            val_loss = [] # Historique des couts
            train_loss = [] # Historique des couts
            fig, ax = plt.subplots(1) # Init figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            model.train()
            for batch_idx, data in enumerate(train_loader):
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                optimizer.zero_grad()
                # Forward
                output, hidden, attn = model(input_seq)
                loss = criterion(output.view(-1, model.dict_size['trad']), target_seq.view(-1))
                running_loss_train += loss.item()

                # Backward
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset),
                                        100. * batch_idx / len(train_loader), running_loss_train / (batch_idx + 1)), end='\r')

                # calcul distance Levenshtein
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    dist += edit_distance(a[:Ma], b[:Mb])/batch_size

            # Validation
            running_loss_val = 0
            model.eval()
            for batch_idx, data in enumerate(val_loader):
                input_seq, target_seq = data
                input_seq, target_seq = input_seq.to(device).long(), target_seq.to(device).long()

                output, hidden, attn = model(input_seq)
                loss = criterion(output.view(-1, model.maxlen['trad']), target_seq.view(-1))
                running_loss_val += loss.item()
            print('\nValidation - Average loss: {:.4f}'.format(running_loss_val / len(val_loader)))
            print('')

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model, 'model.pt')


            # Affichage
            if learning_curves:
                train_loss.append(running_loss_train/len(train_loader))
                val_loss.append(running_loss_val/len(val_loader))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(val_loss, label='validation loss')
                ax.legend()
                plt.draw()
                plt.pause(0.01)
        if learning_curves:
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass