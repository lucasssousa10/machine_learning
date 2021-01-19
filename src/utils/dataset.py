from numpy import genfromtxt

class Dataset:
    default_delimiter = ','
    file_path = ""
    raw_data = []
    x = []
    t = []

    def __init__(self, path):
        self.file_path = path
        self.raw_data = genfromtxt(self.file_path, delimiter=self.default_delimiter)
        self.t = self.raw_data[:, -1]
        self.x = self.raw_data[:, :-1]

    def shuffle(self):
        self.raw_data = self.raw_data.shuffle()
        self.t = self.raw_data[:, -1]
        self.x = self.raw_data[:, :-1]
