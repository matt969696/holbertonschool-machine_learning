#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf
import numpy as np


def forward_prop(prev, layers, activations, epsilon):
    """forward propagation"""
    res = prev
    for i in range(len(layers) - 1):
        winit = tf.keras.initializers.VarianceScaling(mode='fan_avg')
        baselayer = tf.keras.layers.Dense(layers[i],
                                          kernel_initializer=winit)
        mu, sig = tf.nn.moments(baselayer(res), axes=[0])
        gamma = tf.Variable(tf.ones([layers[i]]), trainable=True)
        beta = tf.Variable(tf.zeros([layers[i]]), trainable=True)
        layer = tf.nn.batch_normalization(baselayer(res), mu,
                                          sig, beta, gamma, epsilon)
        res = activations[i](layer)

    winit = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    flayer = tf.keras.layers.Dense(layers[-1], activation=None,
                                   kernel_initializer=winit, name='layer')
    res = flayer(res)
    return res


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """builds the model"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder("float", shape=(None, X_train.shape[1]))
    y = tf.placeholder("float", shape=(None, Y_train.shape[1]))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    equality = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32), name='Mean')
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0)
    alphadecay = tf.train.inverse_time_decay(alpha, global_step,
                                             1, decay_rate,
                                             staircase=True)

    opt = tf.train.AdamOptimizer(alphadecay, beta1, beta2, epsilon=epsilon)
    train_op = opt.minimize(loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        nbbatch = m // batch_size + 1
        saver = tf.train.Saver()

        for i in range(epochs + 1):
            costt = loss.eval({x: X_train,
                               y: Y_train})
            accut = accuracy.eval({x: X_train,
                                   y: Y_train})
            costv = loss.eval({x: X_valid,
                               y: Y_valid})
            accuv = accuracy.eval({x: X_valid,
                                   y: Y_valid})
            print("After {} epochs:\n".format(i) +
                  "\tTraining Cost: {}\n".format(costt) +
                  "\tTraining Accuracy: {}\n".format(accut) +
                  "\tValidation Cost: {}\n".format(costv) +
                  "\tValidation Accuracy: {}".format(accuv))
            if i == epochs:
                break
            rand_x, rand_y = shuffle_data(X_train, Y_train)
            for j in range(nbbatch):
                start = batch_size * j
                end = min(m, batch_size * (j + 1))
                Xb = rand_x[start:end]
                Yb = rand_y[start:end]
                sess.run(train_op, {x: Xb, y: Yb})
                if (j + 1) % 100 == 0:
                    costs = loss.eval({x: Xb, y: Yb})
                    accus = accuracy.eval({x: Xb, y: Yb})
                    print("\tStep {}:\n".format(j + 1) +
                          "\t\tCost: {}\n".format(costs) +
                          "\t\tAccuracy: {}".format(accus))
            sess.run(tf.assign(global_step, global_step + 1))
        save_path = saver.save(sess, save_path)

    return save_path
