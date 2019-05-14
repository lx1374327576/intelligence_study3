# runtime
from dataset import Cir10
from cnn import Cnn
from solver import Solver

# debug
# from .dataset import Cir10
# from .cnn import Cnn
# from .solver import Solver

import numpy as np


def tran(x):
    x = x.transpose(0, 3, 2, 1)
    return x


cir10 = Cir10()
X_train, y_train, X_val, y_val, X_test, y_test = cir10.get_CIFAR10_data(tran=tran)
arr = np.random.permutation(range(X_train.shape[0]))
data = dict()
data['X_train'] = X_train
data['y_train'] = y_train
data['X_val'] = X_val
data['y_val'] = y_val
print(X_train.shape)
model = Cnn()
solver = Solver(model, data, learning_rate=6e-1, batch_size=50, times=120)
solver.train()
model = solver.get_best_model
scores = model.loss(X_test)
y_pred = np.argmax(scores, axis=1)
acc = np.mean(y_pred == y_test)
print('test_acc:', acc)
