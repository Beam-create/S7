# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    visualise = True

    batch_size = 64
    n_epochs = 50
    lr = 0.005

    n_hidden = 9
    n_layers = 1
    train_val_split = 0.7

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')
    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    maxlen = {'in':dataset.max_len_seq, 'out':dataset.max_len_word}
    model = trajectory2seq_attn(n_hidden, n_layers, dataset.int2symb, dataset.symb2int, dataset.dict_size,device, maxlen)

    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters() ]))

    # Initialisation des variables
    best_val_loss = np.inf # pour sauvegarder le meilleur model

    if trainning:

        # Initialisation affichage
        if learning_curves:
            train_dist = [] # Historique des distances
            train_loss = [] # Historique des coûts
            val_dist = []
            val_loss = []
            fig, ax = plt.subplots(1, 2) # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2) # ignore_index=2 ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):

            running_loss_train = 0
            running_loss_val = 0
            epoch_dist_train = 0
            epoch_dist_val = 0

            # Training
            model.train()
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                seq, target = data

                optimizer.zero_grad()

                #  Train Forward
                output, hidden = model(seq)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                running_loss_train += loss.item()

                # Train Backward
                loss.backward()
                optimizer.step()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    epoch_dist_train += edit_distance(a[:Ma], b[:Mb]) / batch_size

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_size, len(dataload_train.dataset),
                    100. *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    epoch_dist_train/len(dataload_train)), end='\r')
            print("")

            # Validation
            model.eval()
            for batch_idx, data in enumerate(dataload_val):
                # Formatage des données
                seq, target = data

                # Forward
                output, hidden = model(seq)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                running_loss_val += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    epoch_dist_val += edit_distance(a[:Ma], b[:Mb]) / batch_size

            print('Valid - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, batch_size, len(dataload_val.dataset),
                                 100. * batch_size / len(dataload_val.dataset),
                                 running_loss_val / (batch_idx + 1),
                                 epoch_dist_val / len(dataload_val)), end='\r')
            print('\n')

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model, 'model.pt')


            # Affichage
            if learning_curves:
                train_loss.append(running_loss_train / len(dataload_train))
                train_dist.append(epoch_dist_train / len(dataload_train))
                val_loss.append(running_loss_val / len(dataload_val))
                val_dist.append(epoch_dist_val / len(dataload_val))
                ax[0].cla()
                ax[0].set_title("Loss")
                ax[0].plot(train_loss, label='training loss')
                ax[0].plot(val_loss, label='valid loss')
                ax[0].legend()
                ax[1].cla()
                ax[1].set_title("Distance")
                ax[1].plot(train_dist, label='training distance')
                ax[1].plot(val_dist, label='valid distance')
                ax[1].legend()
                #plt.show()
                plt.tight_layout()
                plt.pause(0.01)

        # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close('all')

        if visualise:
            model.eval()
            with torch.no_grad():
                for i in range(3):
                    # Find a random data in dataset
                    idx = np.random.randint(0, len(dataset))

                    input, target = dataset[idx]

                    # Forward, loss and distance
                    output, hidden = model(input.unsqueeze(0))
                    loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                    a = target.tolist()
                    b = torch.argmax(output, dim=-1).tolist()
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    epoch_dist_train = edit_distance(a[:Ma], b[:Mb])

                    # Show results
                    output = output.squeeze(0)
                    word_pred = [dataset.int2symb[j] for j in torch.argmax(output, dim=-1).tolist()]
                    word_target = [dataset.int2symb[j] for j in target.tolist()]
                    print("Index        :", idx)
                    print("Prediction   :", word_pred)
                    print("target       :", word_target)
                    print("Loss         :", loss.item())
                    print("Distance     :", epoch_dist_train)
                    print("")
                    dataset.visualisation(idx)



    if test:
        with torch.no_grad():
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