import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
# slow ver.
def cross_entropy(Y, P):
    Y_arr = np.asarray(Y)
    P_arr = np.asarray(P)
    cross_entropy_arr = []
    i = 0
    
    while i < len(Y_arr):
        y = Y_arr[i]
        p = P_arr[i]
        cross_entropy_arr.append(-y*np.log(p) -(1-y)*np.log(1-p))
        i += 1
    
    cross_entropy = np.asarray(cross_entropy_arr).sum()
    
    return cross_entropy

#quick ver.
def cross_entropy(Y, P):
    #convert to float array so that you can do log
    Y = np.float_(Y)
    P = np.float_(P)
    #np.method() works on the entire array~
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
