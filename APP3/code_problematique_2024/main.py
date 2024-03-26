# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import matplotlib.pyplot as plt
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
    training = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    display_attention = False
    seed = 3221                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    batch_size = 64
    lr = 0.014
    n_epochs = 150

    n_hidden = 14
    n_layers = 2
    validation_split = 0.8


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
    model = trajectory2seq_attn_bi(n_layers=n_layers, \
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
            train_dist = []
            val_dist = []
            val_loss = [] # Historique des couts
            train_loss = [] # Historique des couts
            fig, ax = plt.subplots(1, 2) # Init figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2) # ignore les symbole de padding, sinon résultats biaisées
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            running_loss_val = 0
            epoch_dist_train = 0
            epoch_dist_val = 0
            M = 0
            model.train()
            for batch_idx, data in enumerate(train_loader):
                input_seq, target_seq = data
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                optimizer.zero_grad()
                # Forward
                output, hidden, attn = model(input_seq)
                loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))
                running_loss_train += loss.item()

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                # calcul distance Levenshtein
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.to(device).tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    epoch_dist_train += edit_distance(a[:Ma], b[:Mb])/M

            # Validation
            model.eval()
            for batch_idx, data in enumerate(val_loader):
                input_seq, target_seq = data
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                output, hidden, attn = model(input_seq)
                loss = criterion(output.view(-1, model.dict_size), target_seq.view(-1))
                running_loss_val += loss.item()

                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    epoch_dist_val += edit_distance(a[:Ma], b[:Mb])/M
            print('Validation - Epoch: {}/{} Average Loss: {:.6f} Average Edit Distance: {:.6f})'.format(
                epoch, n_epochs, running_loss_val/len(val_loader),
                epoch_dist_val/len(val_loader)), end='\r')
            print('\n')

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model, 'model.pt')

            # Affichage
            if learning_curves:
                train_loss.append(running_loss_train/len(train_loader))
                train_dist.append(epoch_dist_train/len(train_loader))

                val_loss.append(running_loss_val/len(val_loader))
                val_dist.append(epoch_dist_val/len(val_loader))

                ax[0].cla()
                ax[0].set_title('Loss')
                ax[0].plot(train_loss, label='training loss')
                ax[0].plot(val_loss, label='validation loss')
                ax[0].legend()
                ax[1].cla()
                ax[1].set_title('Edit distance')
                ax[1].plot(train_dist, label='training distance')
                ax[1].plot(val_dist, label='validation distance')
                ax[1].legend()
                plt.tight_layout()
                plt.pause(0.01)
        if learning_curves:
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        with torch.no_grad():
            model = torch.load('model.pt')
            criterion = nn.CrossEntropyLoss(ignore_index=2)
            test_dataset = []
            test_dataset =  HandwrittenWords(filename='data_test.p', is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
            running_loss_test = 0
            test_dist = 0
            pred_list = []
            target_list =[]
            for batch_idx, data  in enumerate(test_loader):
                input, target = data
                input, target = input.to(device), target.to(device)
                output, hidden, attn = model(input)
                # output = output.squeeze(0)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                running_loss_test += loss.item()

                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    target_list.append(a)
                    b = output_list[i]
                    pred_list.append(b)
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    test_dist += edit_distance(a[:Ma], b[:Mb]) / M
            print('Test - Average Loss: {:.6f} Average Edit Distance: {:.6f})'.format(
                running_loss_test / len(test_loader), test_dist / len(test_loader)), end='\r')
            print('\n')

            conf_matrix = confusion_matrix(target_list, pred_list, dataset.dict_size, ignore=[0,1,2])
            # Plotting the matrix
            plt.imshow(conf_matrix, cmap='binary', interpolation='nearest')
            plt.colorbar()
            # Set the labels using the dictionary
            plt.xticks(range(dataset.dict_size-3), [dataset.int2symb[i] for i in range(3, dataset.dict_size)], rotation=45, ha="right")
            plt.yticks(range(dataset.dict_size-3), [dataset.int2symb[i] for i in range(3, dataset.dict_size)])

            # Set title and labels
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

            np.random.seed(None)
            for i in range(5):
                rdm_idx = np.random.randint(len(test_dataset))
                input, target = test_dataset[rdm_idx]
                input, target = input.to(device).unsqueeze(0), target.to(device)


                output, hidden, attn = model(input)
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_list = target.cpu().tolist()
                pred_word = []
                target_word = []
                pred_word.append([dataset.int2symb[i] for i in output_list[0]])
                target_word.append([dataset.int2symb[i] for i in target_list])
                M = len(output_list)
                for i in range(M):
                    a = target_list
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    test_distance = edit_distance(a[:Ma], b[:Mb])
                print(f"Prediction: {pred_word}")
                print(f"Target: {target_word}")
                print(f'Distance: {test_distance}')
