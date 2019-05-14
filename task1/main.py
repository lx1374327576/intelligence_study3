# runtime
from dataset import MNIST
from cnn import Cnn
from solver import Solver

# debug
# from .dataset import MNIST
# from .cnn import Cnn
# from .solver import Solver

mnist = MNIST(shuffle=True)
data = dict()
data['X_train'] = mnist.X[:100].reshape((100, 1, 28, 28))
data['y_train'] = mnist.y[:100]
data['X_val'] = mnist.X[100:200].reshape((100, 1, 28, 28))
data['y_val'] = mnist.y[100:200]
model = Cnn()
solver = Solver(model, data, learning_rate=1e-3, batch_size=50, times=50)
solver.train()
