#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a loaded NN model using mini-batch gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection('train_op')[0]
        m = X_train.shape[0]
        nbbatch = m // batch_size + 1

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

        return saver.save(sess, save_path)
