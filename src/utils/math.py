import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(z))

def bernouli(q, x):
    return (q**x)*((1-q)**(1-x))