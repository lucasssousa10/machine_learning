import numpy as np
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class KmeansClustering():
    name = 'K-Means Clustering'
    
    def __init__(self, k, episilon=0.03, max_iter=100, verbose=True):
        self.k = k
        self.episilon = episilon
        self.max_iter = max_iter
        self.verbose = verbose

        self.color_map = cm.get_cmap('Blues', self.k+1)

    def run(self, data):

        # init random centers among minimal and maximum bounds
        mins = np.amin(data, axis=0)
        maxs = np.amax(data, axis=0)
        centers = np.zeros([mins.shape[0], self.k])

        for i in range(mins.shape[0]):
            r = np.random.uniform(mins[i], maxs[i], self.k)
            centers[i, :] = r

        centers = centers.T
        dists = cdist(data, centers, 'euclidean')
        labels = np.argmin(dists, axis=1)
        variations = np.ones([centers.shape[0]]) * 1000000000
        
        iter = 0
        while np.amax(variations) > self.episilon and self.max_iter > iter:

            for i in range(centers.shape[0]):
                inds = np.where(i == labels)
                
                if data[inds, :].shape[1] == 0:
                    continue

                center = np.mean(data[inds, :], axis=1)[0]

                dist = euclidean(centers[i, :], center)
                variations[i] = dist
                centers[i, :] = center
            
            dists = cdist(data, centers, 'euclidean')
            labels = np.argmin(dists, axis=1)
            iter = iter + 1
        
        if self.verbose:
            self.plot(data, centers, labels)
        
        return centers, labels
    
    def plot(self, data, centers, labels):
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.suptitle(self.name, fontsize=16)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        for i in range(data.shape[0]):
            ax.plot(data[i, 0], data[i, 1], 'o', color=self.color_map(labels[i]+1))
        
        for i in range(centers.shape[0]):
            ax.plot(centers[i, 0], centers[i, 1], 'x', color='#000000')
        
        plt.show()



        

            

        
        


        