from src.utils import math, dataset, plot, metrics
from src.models.gmm_map_clustering import GmmMapClustering
import numpy as np

data = dataset.Dataset('./datasets/clustering_data.csv')
data.split()    

gmm = GmmMapClustering(3)
gmm.run(data.x)