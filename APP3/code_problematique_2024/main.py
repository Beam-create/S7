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
    trainning = False          # Entrainement?
    test = True                # Test?
    display_attention = True
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    batch_size = 64
    n_epochs = 150
    lr = 0.014

    n_hidden = 10
    n_layers = 2
    train_val_split = 0.8

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p', device)
    
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
    model = trajectory2seq_attn_bi(n_hidden, n_layers, dataset.int2symb, dataset.symb2int, dataset.dict_size,device, maxlen)

    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters() ]))

    # Initialisation des variables
    best_val_loss = np.inf # pour sauvegarder le meilleur model
    pred_val_list = []
    true_val_list = []

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
                output, hidden, attn = model(seq)

                loss = criterion(output.view((-1, dataset.dict_size)), target.view(-1))
                running_loss_train += loss.item()

                # Train Backward
                loss.backward()

                torch.nn.utils.clip_grad_norm(model.parameters(), 1)
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
                    epoch_dist_train += edit_distance(a[:Ma], b[:Mb]) / M

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_size, len(dataload_train.dataset),
                    100. *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    epoch_dist_train/ len(dataload_train)), end='\r')
            print("")

            # Validation
            model.eval()
            for batch_idx, data in enumerate(dataload_val):
                # Formatage des données
                seq, target = data

                # Forward
                output, hidden, attn = model(seq)
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
                    epoch_dist_val += edit_distance(a[:Ma], b[:Mb]) / M

            print('Valid - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, batch_size, len(dataload_val.dataset),
                                 100. * batch_size / len(dataload_val.dataset),
                                 running_loss_val / (batch_idx + 1),
                                 epoch_dist_val /  len(dataload_val)), end='\r')
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

    if test:
        with torch.no_grad():

            # Instanciation de l'ensemble de données
            dataset_test = HandwrittenWords('data_test.p', device)
            print('Test data : ', len(dataset_test))

            # Instanciation des dataloaders
            dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=n_workers)

            model = torch.load('model.pt')
            criterion = nn.CrossEntropyLoss(ignore_index=2)

            true_val_list = []
            pred_val_list = []
            running_loss_test = 0
            dist_test = 0

            # Évaluation des données de test
            for batch_idx, data in enumerate(dataload_test):
                # Formatage des données
                seq, target = data

                # Forward
                output, hidden, attn = model(seq)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                running_loss_test += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist_test += edit_distance(a[:Ma], b[:Mb]) / M

                    # Add output and target to confusion matrix list
                    pred_val_list.append(b)
                    true_val_list.append(a)

            print('Test - Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                     running_loss_test / (batch_idx + 1),
                    dist_test/ len(dataload_test)), end='\r')
            print("")

            # Affichage des résultats de test
            for i in range(5):
                # Find a random data in dataset
                idx = np.random.randint(0, len(dataset_test))

                input, target = dataset_test[idx]

                # Forward, loss and distance
                output, hidden, attn = model(input.unsqueeze(0))
                output = output.squeeze(0)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))

                # Compute distance
                a = target.cpu().tolist()
                b = torch.argmax(output, dim=-1).cpu().tolist()
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

                # Affichage de l'attention
                if display_attention:
                    # For each letter of the output, display the attention of the model on the input trajectory
                    # The input trajectory is the same for each letter of the output, but where the attention is the highest, darker the points and the line connecting them are
                    #  The attention weights are a tensor of the length of the input trajectory
                    # Attention weights are a tensor of shape (1, len(input), len(output)) -> (1, len(output), len(input))
                    #attn = attn.permute(0, 2, 1)

                    # Choose the size of the figure depending on the number of letters in the output
                    plt.figure(figsize=(1, 1 * len(output)))
                    # Create a subplot for each letter of the output
                    word = [dataset.int2symb[i] for i in b]
                    for i in range(len(output)):
                        plt.subplot(len(output), 1, i + 1)
                        # Get the attention weights for the i-th letter
                        colors = attn[0, 0, :].detach().cpu().numpy()
                        colors = 0.2 + 0.8*(1 - colors)
                        # Get the trajectory
                        traj = input.detach().cpu().numpy()
                        # Plot the trajectory
                        # plt.plot(traj[0], traj[1], 'b', linewidth=0.5)
                        # Plot the attention weights on the trajectory (darker points and lines where the attention is higher)
                        plt.scatter(traj[0], traj[1], c=colors, cmap='gray')

                        plt.title(word[i])

                    plt.show()

            # Affichage de la matrice de confusion
            matrix = confusion_matrix(true_val_list, pred_val_list, dataset.int2symb, [0, 1, 2])

            # Display the array as an image using Matplotlib
            plt.imshow(matrix, cmap='binary', interpolation='nearest')

            # Set custom tick labels for both axes
            plt.xticks(range(26), [dataset.int2symb[i ] for i in range(3, 29)])
            plt.yticks(range(26), [dataset.int2symb[i ] for i in range(3, 29)])

            plt.ylabel('True ')
            plt.xlabel('Predicted ')
            plt.title('Confusion Matrix')
            plt.colorbar()  # Add color bar for reference
            plt.show()