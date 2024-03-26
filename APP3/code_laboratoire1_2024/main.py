# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Inclure la base de donnéess
from dataset import *

# Inclure le modèle
from models import *

if __name__ =="__main__":

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = True            # Forcer l'utilisation du CPU (si un GPU est disponible)
    training = True             # Faire l'entrainement sur l'ensemble de donnees
    learning_curves = True      # Visualiser les courbes d'apprentissage pendant l'entrainement
    test_tagging = False         # Visualiser l'annotation sur des echantillons de validation
    test_generation = True     # Visualiser la generation sur des echantillons de validation

    batch_size = 10             # Taille des lots
    n_epochs = 50               # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur

    n_hidden = 25               # Nombre de neurones caches par couche 
    n_layers = 1                # Nombre de de couches

    n_workers = 0               # Nombre de fils pour charger les donnees
    train_val_split = .7        # Ratio des echantillions pour l'entrainement
    seed = 1                 # Pour repetabilite
    # ------------ Fin des paramètres et hyperparamètres -----------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = SignauxDataset()

    # Séparation du dataset (entraînement et validation)
    n_train_samp = int(len(dataset)*train_val_split)
    n_val_samp = len(dataset)-n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    model = Model(n_hidden,  n_layers=n_layers)
    model = model.to(device)

    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    
    # Initialisation des variables
    best_val_loss = np.inf # pour sauvegarder le meilleur model

    if training:

        # Initialisation affichage
        if learning_curves:
            val_loss =[] # Historique des coûts
            train_loss=[] # Historique des coûts
            fig, ax = plt.subplots(1) # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            model.train()
            for batch_idx, data in enumerate(dataload_train):
                in_seq, target_seq = [obj.to(device).float() for obj in data]

                # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------
                optimizer.zero_grad()
                # Forward
                output, h = model(in_seq)
                output = output.squeeze(-1)
                loss = criterion(output, target_seq)
                running_loss_train += loss.item()

                # Backward
                loss.backward()
                optimizer.step()
                
                # ---------------------- Laboratoire 1 - Question 3 - Fin de la section à compléter ------------------
            
                # Affichage pendant l'entraînement
                if batch_idx % 10 == 0:
                    print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * len(data), len(dataload_train.dataset),
                                        100. * batch_idx / len(dataload_train), running_loss_train / (batch_idx + 1)), end='\r')

            # Validation
            running_loss_val = 0
            model.eval()
            for data in dataload_val:
                in_seq, target_seq = [obj.to(device).float() for obj in data]

                # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------
                output, h = model(in_seq)
                output = output.squeeze(-1)
                loss = criterion(output, target_seq)
                running_loss_val += loss.item()


                # ---------------------- Laboratoire 1 - Question 3 - Fin de la section à compléter ------------------

            print('\nValidation - Average loss: {:.4f}'.format(running_loss_val/len(dataload_val)))
            print('')
            
            # Affichage
            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                val_loss.append(running_loss_val/len(dataload_val))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(val_loss, label='validation loss')
                ax.legend()
                plt.draw()
                plt.pause(0.01)
            
            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model,'model.pt')

        # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close('all')


    if test_tagging:
        with torch.no_grad():
            # Évaluation étiquettage
            model = torch.load('model.pt', map_location=lambda storage, loc: storage)
            model = model.to(device)
            model.eval()
            for num in range(10):
                # Extraction d'une séquence du dataset de validation
                input_sequence, target_sequence = dataset_val[np.random.randint(0,len(dataset_val))]

                # Initialisation de la prédiction de sortie
                prediction_sequence = np.zeros(len(input_sequence))


                # ---------------------- Laboratoire 1 - Question 4 - Début de la section à compléter ------------------


                output, h = model(input_sequence.type(torch.float32))
                output = output.squeeze(-1)
                prediction_sequence = output

                # ---------------------- Laboratoire 1 - Question 4 - Fin de la section à compléter ------------------

                plt.title("Tagged data")
                plt.plot(target_sequence, label='target')
                plt.plot(prediction_sequence, label='prediction')
                plt.legend()
                plt.show()


    if test_generation:
        with torch.no_grad():
            # Évaluation génération
            model = torch.load('model.pt', map_location=lambda storage, loc: storage)
            model = model.to(device)
            model.eval()
            for num in range(10):
                # Extraction d'une séquence du dataset de validation
                input_sequence, target_sequence = dataset_val[np.random.randint(0,len(dataset_val))]

                # Calcul du nombre de prédictions à générer
                usable_input_sequence_len = len(input_sequence)>>1
                nb_predictions_to_generate = len(input_sequence)-usable_input_sequence_len

                # Initialisation de la prédiction de sortie
                prediction_sequence = np.zeros(nb_predictions_to_generate)


                # ---------------------- Laboratoire 1 - Question 5 - Début de la section à compléter ------------------

                input = input_sequence[0:usable_input_sequence_len].type(torch.float32).unsqueeze(0)
                print(input.shape)
                output, h = model(input)
                print(output.shape)
                for i in range(nb_predictions_to_generate-1):
                    generation_input = output[:,-1]
                    output, h = model(generation_input, h)
                    #newoutput = output.squeeze(-1)

                    prediction_sequence[i] = output.item()

                # ---------------------- Laboratoire 1 - Question 5 - Fin de la section à compléter ------------------


                prediction_t = [i+usable_input_sequence_len for i in range(nb_predictions_to_generate)]
                plt.plot(target_sequence)
                plt.plot(prediction_t,prediction_sequence)
                plt.title("Generated data")
                plt.show()