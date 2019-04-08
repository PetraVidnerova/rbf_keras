from keras.initializers import Initializer
import numpy as np


class InitFromFile(Initializer):
    """ Initialize the weights by loading from file.

    # Arguments
        filename: name of file, should by .npy file
    """
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, shape, dtype=None):
        with open(self.filename, "rb") as f:
            X = np.load(f)
        assert shape == X.shape
        return X

    def get_config(self):
        return {
            'filename': self.filename
        }
