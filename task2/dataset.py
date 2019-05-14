import numpy as np
import os
import platform
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt


class Cir10:

    # reference from cs321n
    def load_pickle(self, f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return pickle.load(f)
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    # reference from cs321n
    def load_CIFAR_batch(self, filename):
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int")
            Y = np.array(Y)
            return X, Y

    # reference from cs321n
    def load_CIFAR10(self, ROOT):
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # reference from cs321n
    def get_CIFAR10_data(self, num_training=49000, num_validation=1000, num_test=1000):
        cifar10_dir = './datasets/cifar-10-batches-py'

        X_train, y_train, X_test, y_test = self.load_CIFAR10(cifar10_dir)

        # Subsample the data
        mask = list(range(num_training, num_training + num_validation))
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = list(range(num_training))
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = list(range(num_test))
        X_test = X_test[mask]
        y_test = y_test[mask]

        return X_train, y_train, X_val, y_val, X_test, y_test


cir = Cir10()
X_train, y_train, _, _, _, _ = cir.get_CIFAR10_data()
for i in range(10):
    image = X_train[i, :, :, :]
    plt.figure(i)
    plt.imshow(image)
    plt.show()
