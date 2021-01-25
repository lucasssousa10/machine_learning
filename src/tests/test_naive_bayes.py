from src.utils import math, dataset, plot, metrics
from src.models.naive_bayes import NaiveBayes


import numpy as np

data = dataset.Dataset('./datasets/votesDataset.csv')
model = NaiveBayes()

data.split()    

model.train(data.x_train, data.t_train)
out = model.predict(data.x_test)

print(out)
print(data.t_test)
conf_mat = metrics.confusion_matrix_binary(out, data.t_test)
print(conf_mat)