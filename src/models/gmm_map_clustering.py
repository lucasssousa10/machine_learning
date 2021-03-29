import numpy as np
from scipy.spatial.distance import cdist, euclidean
from src.models.kmeans_clustering import KmeansClustering
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

class GmmMapClustering():
    name = 'Gaussian Mixture Model - MAP'
    centers = []
    pis = []
    covs = {}
    def __init__(self, k, max_iter=10):
        self.k = k
        self.max_iter = max_iter

    def run(self, data):
    
        N = data.shape[0]
        d = data.shape[1]
        S = np.cov(data.T) * 1/(self.k ** (2/d))
        v = d + 2
        kapa = 0.05
        mu = np.mean(data, axis=0)
        alphas_k = np.zeros(self.k)
        pis = np.zeros(self.k)
        covs = {}

        kmeans = KmeansClustering(self.k, verbose=False)
        centers, labels = kmeans.run(data)

        for i in range(self.k):
            inds = np.where(labels == i)
            pis[i] = inds[0].shape[0]/N
            alphas_k[i] = inds[0].shape[0]/N
            samples = data[inds, :][0]
            if samples.shape[0] > 0:
                covs[i] = np.cov(samples.T)
            else:
                covs[i] = np.eye(d)

        iter = 0
        while self.max_iter > iter:

            # passo E
            riks = {}
            for i in range(self.k):
                rik = pis[i] * multivariate_normal.pdf(data, mean=centers[i, :], cov=covs[i])
                
                rik_d = 0
                for j in range(self.k):
                    rik_d = rik_d + pis[j] * multivariate_normal.pdf(data, mean=centers[j, :], cov=covs[j])
                rik = rik / rik_d
                riks[i] = rik

            # passo M
            
            for i in range(self.k):
                sum_riks = np.sum(riks[i])

                pis[i] = alphas_k[i] - 1 + sum_riks
                pis[i] = pis[i] / (N - self.k + np.sum(alphas_k))

                xk = np.zeros(d)
                for j in range(data.shape[0]):
                    xk = xk + riks[i][j] * data[j, :]
                
                xk = xk / sum_riks

                centers[i, :] = kapa * centers[i, :] + xk * sum_riks
                centers[i, :] = centers[i, :] / (kapa + sum_riks)
                
                sa = 0
                for j in range(data.shape[0]):
                    sa = sa + riks[i][j] * np.dot(np.array([data[j, :] - xk]).T, np.array([data[j, :] - xk]))
                
                sa = S + sa
                sb = (kapa * sum_riks) / (kapa + sum_riks)
                sb = sb * np.dot(np.array([xk - centers[i, :]]).T, np.array([xk - centers[i, :]]))


                sa = sa + sb
                sa = sa / (v + d + 2 + sum_riks)
                
                covs[i] = sa
  
            iter = iter + 1

            self.centers = centers
            self.covs = covs
            self.pis = pis

        self.plot(data)

        

    def prob(self, data):
        probs = np.zeros(data.shape[0])
        
        for k in range(self.k):
            probs = probs + self.pis[k] * multivariate_normal.pdf(data, mean=self.centers[k, :], cov=self.covs[k])
            
        return probs

    def plot(self, data):
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.suptitle(self.name, fontsize=16)
        N = 100
        mins = np.amin(data, axis=0)
        maxs = np.amax(data, axis=0)
        x = np.linspace(mins[0], maxs[0], N)
        y = np.linspace(mins[1], maxs[1], N)
        X, Y = np.meshgrid(x, y)
        old_shape = X.shape
        X = np.reshape(X, [np.product(old_shape), 1])
        Y = np.reshape(Y, [np.product(old_shape), 1])
        samples = np.concatenate((X, Y), axis=1)

        Z = self.prob(samples)
        X = np.reshape(X, old_shape)
        Y = np.reshape(Y, old_shape)
        Z = np.reshape(Z, old_shape)

        ax.plot(data[:, 0], data[:, 1], 'o', color='#dfc27d')

        cs = ax.contourf(X, Y, Z, cmap=cm.PuBu_r)

        plt.savefig('surface.pdf') 
        
        