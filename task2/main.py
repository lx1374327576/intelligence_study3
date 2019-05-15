# runtime
from dataset import Cir10

import numpy as np
import tensorflow as tf


# def tran(x):
#     x = x.transpose(0, 3, 1, 2)
#     return x


cir10 = Cir10()
X_train, y_train, X_val, y_val, X_test, y_test = cir10.get_CIFAR10_data()

tf.reset_default_graph()

with tf.device('/cpu:0'):
    sc_init = tf.variance_scaling_initializer(scale=2.0)

    optimizer = tf.keras.optimizers.SGD(lr=1e-2)

    layers = [
        tf.keras.layers.Conv2D(128, (5, 5), (1, 1), "same", activation='relu',
                               use_bias=True, bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=sc_init, input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(strides=2),
        tf.keras.layers.Conv2D(128, (5, 5), (1, 1), "same", activation='relu',
                               use_bias=True, bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=sc_init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(strides=2),
        tf.keras.layers.Conv2D(128, (5, 5), (1, 1), "same", activation='relu',
                               use_bias=True, bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=sc_init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(strides=2),
        tf.keras.layers.Conv2D(128, (5, 5), (1, 1), "same", activation='relu',
                               use_bias=True, bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=sc_init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, kernel_initializer=sc_init,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Softmax(),
    ]

    model = tf.keras.Sequential(layers)

    model.compile(optimizer, "sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
