import numpy as np
from src.utils.math import bernouli
class NaiveBayes:
    theta_d_c1 = []
    theta_d_c2 = []
    prior_c1 = 0
    prior_c2 = 0

    def __init__(self):
        pass

    def train(self, inp, out):

        inp_c1 = []
        inp_c2 = []
        out_c1 = []
        out_c2 = []
        for i in range(inp.shape[0]):
            if (out[i] == 1):
                inp_c1.append(inp[i, :])
                out_c1.append(out[i])
            else:
                inp_c2.append(inp[i, :])
                out_c2.append(out[i])
        inp_c1 = np.array(inp_c1)
        inp_c2 = np.array(inp_c2)
        out_c1 = np.array(out_c1)
        out_c2 = np.array(out_c2)

        self.theta_d_c1 = np.sum(inp_c1, axis=0) / inp_c1.shape[0]
        self.theta_d_c2 = np.sum(inp_c2, axis=0) / inp_c2.shape[0]
        self.prior_c1 = inp_c1.shape[0] / float(inp.shape[0])
        self.prior_c2 = inp_c2.shape[0] / float(inp.shape[0])

    def predict(self, inp):
        
        prob_c1 = []
        prob_c2 = []

        for i in range(inp.shape[1]):
            prob_c1.append(bernouli(self.theta_d_c1[i], inp[:, i]))
            prob_c2.append(bernouli(self.theta_d_c2[i], inp[:, i]))
        
        prob_c1 = np.prod(np.array(prob_c1), axis=0) * self.prior_c1
        prob_c2 = np.prod(np.array(prob_c2), axis=0) * self.prior_c2
        
        probs = np.concatenate([[prob_c2], [prob_c1]])
        return np.argmax(probs, axis=0)
        