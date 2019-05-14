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
data['X_train'] = mnist.X[:500].reshape((500, 1, 28, 28))
data['y_train'] = mnist.y[:500]
data['X_val'] = mnist.X[500:600].reshape((100, 1, 28, 28))
data['y_val'] = mnist.y[500:600]
model = Cnn()
solver = Solver(model, data, learning_rate=6e-1, batch_size=50, times=120)
solver.train()
