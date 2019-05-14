import numpy as np
# runtime
import ts_lib
# debug
# import task1.ts_lib


# conv ->relu ->max_pool -> fc ->relu ->fc
class Cnn(object):

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-2, reg=0.01):

        self.params = {}
        self.reg = reg
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(H//2*W//2*num_filters, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_stride = 1
        conv_pad = (filter_size - 1) // 2
        pool_height = 2
        pool_width = 2
        pool_stride = 2

        layer1, cache1 = ts_lib.conv(X, W1, b1, conv_stride, conv_pad)
        layer1_relu, cache1_relu = ts_lib.relu(layer1)
        layer1_pool, cache1_pool = ts_lib.max_pool(layer1_relu, pool_height, pool_width, pool_stride)
        layer2, cache2 = ts_lib.fc(layer1_pool, W2, b2)
        layer2_relu, cache2_relu = ts_lib.relu(layer2)
        scores, cache3 = ts_lib.fc(layer2_relu, W3, b3)

        # print('X:', X)
        # print('layer1:', layer1)
        # print('layer1_relu', layer1_relu)
        # print('layer1_pool', layer1_pool)
        # print('layer2', layer2)
        # print('layer2_relu', layer2_relu)
        # print('scores', scores)

        if y is None:
            return scores

        grads = {}
        loss, dscores = ts_lib.softmax(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        dlayer2_relu, grads['W3'], grads['b3'] = ts_lib.fc_back(dscores, cache3)
        dlayer2 = ts_lib.relu_back(dlayer2_relu, cache2_relu)
        dlayer1_pool, grads['W2'], grads['b2'] = ts_lib.fc_back(dlayer2, cache2)
        dlayer1_relu = ts_lib.max_pool_back(dlayer1_pool, cache1_pool)
        dlayer1 = ts_lib.relu_back(dlayer1_relu, cache1_relu)
        _, grads['W1'], grads['b1'] = ts_lib.conv_back(dlayer1, cache1)

        # print('dlayer2_relu:', dlayer2_relu[0])
        # print('w3:', grads['W3'][0])
        # print('b3:', grads['b3'][0])
        # print('dlayer2:', dlayer2[0])
        # print('dlayer1_pool:', dlayer1_pool[0])
        # print('w2:', grads['W2'][0])
        # print('b2:', grads['b2'][0])
        # print('dlayer1_relu:', dlayer1_relu[0])
        # print('dlayer1:', dlayer1[0])
        # print('w1:', grads['W1'][0])
        # print('b1:', grads['b1'][0])

        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads
