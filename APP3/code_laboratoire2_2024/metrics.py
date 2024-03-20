import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition
    # Fonction récursive 7 des notes de cours
    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
    if len(a) == 0:
        return len(b)

    elif len(b) == 0:
        return len(a)

    elif a[0] == b[0]:
        return edit_distance(a[1:], b[1:])

    else:
        return 1 + min(edit_distance(a[1:], b), edit_distance(a, b[1:]), edit_distance(a[1:], b[1:]))

    
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
    