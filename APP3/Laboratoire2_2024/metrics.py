import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition

    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
    N, M = len(a), len(b)
    # Create array of size NxM
    dp = [[0 for i in range(M+1)] for j in range(N+1)]
    # Base case: When N = 0
    for j in range(M+1):
        dp[0][j] = j
    # Base case: When M = 0
    for i in range(N+1):
        dp[i][0] = i
    # Transition
    for i in range(1, N+1):
        for j in range(1, M+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], # Insertion
                                   dp[i][j-1], # Deletion
                                   dp[i-1][j-1] # Replacement
                                   )
    return dp[N][M]
    
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
    