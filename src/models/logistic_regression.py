from src.utils.math import sigmoid
import math as mt
import numpy as np

class LogisticRegression:
    w  = []
    S  = []
    m  = []
    H = []
    iterations_irls = 15
    model_name = 'Regressao Logistica Bayesiana'
    
    def __init__(self, w0=[], S0=[], m0=[], iterations_irls=15):
        self.w = w0
        self.S = S0
        self.m = m0
        self.iterations_irls = iterations_irls

    def train(self, input, output):

        # initial values
        self.m = np.array([np.zeros(input.shape[1])])
        self.S = np.identity(input.shape[1])
        self.w = np.array([np.ones(input.shape[1])])

        # IRLS algorithm
        for i in range(self.iterations_irls):
            R = sigmoid(np.dot(self.w, input.T))
            R = R * (1 - R)
            R = np.diag(R[0])

            A = np.dot(input.T, R)
            A = np.dot(A, input) + np.linalg.inv(self.S)

            w_t = output - sigmoid(np.dot(input, self.w[0]))
            w_t = np.dot(np.linalg.inv(A), np.dot(input.T, w_t))
            w_t = w_t - np.dot(np.linalg.inv(self.S), self.w.T - self.m.T).T
            self.w = self.w + w_t

        R_hat = sigmoid(np.dot(self.w, input.T))
        R_hat = R_hat * (1 - R_hat)
        R_hat = np.diag(R_hat[0])

        self.H = np.dot(np.dot(input.T, R_hat), input) + np.linalg.inv(self.S)
        
    def predict(self, input):
        output = []
        for x in input:
            # Probit
            mu_a = np.dot(self.w, x)[0]
            s2   = np.dot(np.dot(x.T, np.linalg.inv(self.H)), x)
            output.append(sigmoid( mu_a * (1 + (mt.pi * s2)/8) ** 0.5))
        return np.array(output)