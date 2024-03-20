# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition
    
    if len(x) == 0:
        return len(y)

    elif len(y) == 0:
        return len(x)

    elif x[0] == y[0]:
        return edit_distance(x[1:], y[1:])

    else:
        return 1 + min(edit_distance(x[1:], y), edit_distance(x, y[1:]), edit_distance(x[1:], y[1:]))


def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion

    # À compléter

    return None
