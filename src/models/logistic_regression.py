from src.utils.math import sigmoid
import numpy as np

class LogisticRegression:
    w  = []
    S  = []
    m  = []
    iterations_irls = 15

    def __init__(self, w0=[], S0=[], m0=[]):
        self.w = w0
        self.S = S0
        self.m = m0

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

    def predict(self, input):
        y = sigmoid(np.dot(self.w, input.T))
        return 1 - y[0]
        

        
        
       