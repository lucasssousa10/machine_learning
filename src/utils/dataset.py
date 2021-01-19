from numpy import genfromtxt
import numpy as np


class Dataset:
    default_delimiter = ','
    file_path = ""
    raw_data = []
    x = []
    t = []
    train_perc = 0.8
    test_perc = 0.2
    validation_perc = 0
    x_train = []
    t_train = []
    x_text = []
    t_test = []
    x_validation = []
    t_validation = []

    def __init__(self, path):
        self.file_path = path
        self.raw_data = genfromtxt(self.file_path, delimiter=self.default_delimiter)
        self.t = self.raw_data[:, -1]
        self.x = self.raw_data[:, :-1]

    def shuffle(self):
        np.random.shuffle(self.raw_data)
        self.t = self.raw_data[:, -1]
        self.x = self.raw_data[:, :-1]

    def split(self):
        self.shuffle()
        n = self.raw_data.shape[0]
        
        n_train = int(round(n * self.train_perc))
        n_test  = int(round(n * self.test_perc))
        n_valid = int(round(n * self.validation_perc))

        self.x_train = self.x[0:n_train,:]
        self.t_train = self.t[0:n_train]

        self.x_test = self.x[n_train:n_train + n_test,:]
        self.t_test = self.t[n_train:n_train + n_test]

        self.x_validation = self.x[n_train + n_test:n_train + n_test + n_valid, :]
        self.t_validation = self.t[n_train + n_test:n_train + n_test + n_valid]