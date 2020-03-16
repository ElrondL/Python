import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    List = np.asarray(L)
    print(List)
    L_exp = np.exp(List)
    print(L_exp)
    L_sum = L_exp.sum()
    print(L_sum)
    result = []
    for l in np.nditer(L_exp):
        result.append(l/L_sum)
    return result
