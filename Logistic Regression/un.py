import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def linear_Classifier(X_i, grad):
    # this function return a value to show if the points are above or below the plane(w_i, w_0)
    w_i = grad[0]
    w_0 = grad[1]
    return np.dot(w_i.T, X_i) + w_0

A = np.array([
    [1,2,3,4,5],
    [2,3,4,5,6],
    [4,5,6,7,8]
])

B = np.array([5,4,3,2,1])

for i in A:
    print(np.dot(B, i))

print(np.dot(B, A.T))