from src.utils import math, dataset, plot
from src.models.logistic_regression import LogisticRegression

import numpy as np

data = dataset.Dataset('./datasets/logistic_regression_data.csv')

model = LogisticRegression()
model.train(data.x, data.t)
out = model.predict(data.x)

plot.plot_classification(data.x, data.t, model)