from src.utils import math, dataset, plot
from src.models.logistic_regression import LogisticRegression

import numpy as np

data = dataset.Dataset('./datasets/logistic_regression_data.csv')
model = LogisticRegression()

for r in range(1):
    data.split()    
    model.train(data.x_train, data.t_train)
    out = model.predict(data.x_test)

plot.plot_classification(data.x, data.t, model)