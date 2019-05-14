import numpy as np


class Solver:

    def __init__(self, model, data, learning_rate, batch_size, times):

        self.model = model
        self.best_model = model
        self.best_acc = 0
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.times = times

    def step(self):

        batch_mask = np.random.choice(self.X_train.shape[0], self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        loss, grads = self.model.loss(X_batch, y_batch)

        print('loss: ', loss)
        for key, _ in self.model.params.items():
            dw = grads[key]
            # print(key, dw)
            self.model.params[key] -= self.learning_rate * dw

    def train(self):

        for i in range(self.times):
            self.step()
            if i % 10 == 0:
                print('repeat ', i, ' times:')
                self.check_acc()
            if i % 5 == 0:
                self.learning_rate *= 0.9
        self.check_acc()
        print('best_val_acc:', self.best_acc)

    def check_acc(self, iters=100):

        batch_mask = np.random.choice(self.X_train.shape[0], iters)
        X = self.X_train[batch_mask]
        y = self.y_train[batch_mask]
        scores = self.model.loss(X)
        y_pred = np.argmax(scores, axis=1)
        train_acc = np.mean(y == y_pred)

        batch_mask = np.random.choice(self.X_val.shape[0], iters)
        X = self.X_val[batch_mask]
        y = self.y_val[batch_mask]
        scores = self.model.loss(X)
        y_pred = np.argmax(scores, axis=1)
        val_acc = np.mean(y == y_pred)

        print('train_acc', train_acc, 'val_acc', val_acc)
        if self.best_acc < val_acc:
            self.best_acc = val_acc
            self.best_model = self.model

    def get_best_model(self):
        return self.best_model
