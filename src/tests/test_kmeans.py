from src.utils import math, dataset, plot, metrics
from src.models.naive_bayes import NaiveBayes
from src.models.kmeans_clustering import KmeansClustering

import numpy as np

data = dataset.Dataset('./datasets/clustering_data.csv')
data.split()    

kmeans = KmeansClustering(3)
kmeans.run(data.x)