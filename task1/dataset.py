# reference from cs231n
import numpy as np
import tensorflow as tf


class MNIST(object):

    def __init__(self, batch_size, shuffle=False):

        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32) / 255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle
        print('MNIST dateset loaded completely!')

    def __iter__(self):

        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))
