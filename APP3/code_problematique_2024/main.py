# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import matplotlib.pyplot as plt
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
    display_attention = True
    seed = 3221                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    batch_size = 32
    lr = 0.01
    n_epochs = 50

    n_hidden = 15
    n_layers = 3
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
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset),
                                        100. * batch_idx / len(train_loader), running_loss_train / (batch_idx + 1)), end='\r')

                # calcul distance Levenshtein
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    epoch_dist_train += edit_distance(a[:Ma], b[:Mb])/batch_size
            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:/6f} Average Edit Distance: {:.6f})'.format(
                epoch, n_epochs, batch_size, len(train_loader.dataset),
                100.* batch_size / len(train_loader.dataset), running_loss_train / (batch_idx + 1),
                epoch_dist_train/len(train_loader)), end='\r')


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
                    epoch_dist_val += edit_distance(a[:Ma], b[:Mb])/batch_size
            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:/6f} Average Edit Distance: {:.6f})'.format(
                epoch, n_epochs, batch_size, len(val_loader.dataset),
                100. * batch_size / len(val_loader.dataset), running_loss_val / (batch_idx + 1),
                epoch_dist_train / len(val_loader)), end='\r')
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

            test_dataset = dict()
            testset_filename = 'test.p'
            with open(testset_filename, 'rb') as fp:
                test_dataset = pickle.load(fp)

        # Charger les données de tests
        running_test_loss = 0
        running_test_dist = 0
        for batch_idx, data in enumerate(test_dataset):
            seq, label = data
            output, hidden, attn = model(seq)

            loss = criterion(output.view((-1, model.dict_size)), label.view(-1))
            a = label.cpu().tolist()
            b = torch.argmax(output, dim=-1).cpu().tolist()
            Ma = a.index(1) # longueur mot a
            Mb = b.index(1) if 1 in b else len(b) #longueur mot b
            test_dist = edit_distance(a[:Ma], b[:Mb])

            output = output.squeeze(0)
            word_pred = [dataset.int2symb[j] for j in torch.argmax(output, dim=-1).tolist()]
            word_target = [dataset.int2symb[j] for j in label.tolist()]

        # Affichage de l'attention
        # À compléter (si nécessaire)

            # Affichage des résultats de test
            print("Prediction: ", word_pred)
            print("Target: ", word_target)
            print("Loss: ", loss.item())
            print("Distance: ", test_dist)
            print("\n")

            if display_attention:
                # For each letter of the output, display the attention of the model on the input trajectory
                # The input trajectory is the same for each letter of the output, but where the attention is the highest, darker the points and the line connecting them are
                #  The attention weights are a tensor of the length of the input trajectory
                # Attention weights are a tensor of shape (1, len(input), len(output)) -> (1, len(output), len(input))
                # attn = attn.permute(0, 2, 1)

                # Choose the size of the figure depending on the number of letters in the output
                plt.figure(figsize=(1, 1 * len(output)))
                # Create a subplot for each letter of the output
                word = [dataset.int2symb[i] for i in b]
                for i in range(len(output)):
                    plt.subplot(len(output), 1, i + 1)
                    # Get the attention weights for the i-th letter
                    colors = attn[:, i, :].detach().cpu().numpy()
                    colors = 0.2 + 0.8 * (1 - colors)
                    # Get the trajectory
                    traj = seq.detach().cpu().numpy()
                    # Plot the trajectory
                    # plt.plot(traj[0], traj[1], 'b', linewidth=0.5)
                    # Plot the attention weights on the trajectory (darker points and lines where the attention is higher)
                    plt.scatter(traj[0], traj[1], c=colors, cmap='gray')

                    plt.title(word[i])

                plt.show()
        
        # Affichage de la matrice de confusion
        # À compléter

        pass