r"""
TensorFlow Eager Execution Example: Linear Regression.
"""

from __future__ import absolute_import   # 绝对路径的引入
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import tensorflow.contrib.eager as tfe

class LinearModel(tf.keras.Model):
    """A tensorflow linear regression model"""

    def __init__(self):
        super(LinearModel, self).__init__()
        self._hidden_layer = tf.layers.Dense(1)

    def call(self, xs):
        """Invoke the linear model

        :param xs: input features, as a tensor of size [batch_size, ndims].
        :return:  the predictions of the linear mode, as a tensor of size [batch_size]
        """
        return self._hidden_layer(xs)


def mean_square_loss(model, xs, ys):
    return tf.reduce_mean(tf.square(tf.subtract(model(xs), ys)))


def fit(model, dataset, optimizer, verbose=False, logdir=None):
    """Fit the linear-regression model

    :param model:
    :param dataset: The tf.data.Dataset to use for training data.
    :param optimizer:
    :param verbose: If true, will print out loss values at every iteration
    :param logdir: The directory in which summaries will be written for TensorBoard
      (optional)
    :return:
    """

    # the loss function to optimize.
    mse = lambda xs, ys: mean_square_loss(model, xs, ys)
    loss_and_grads = tfe.implicit_value_and_gradients(mse)

    if logdir:
        summary_writer = tf.contrib.summary.create_file_writer(logdir)

    for i,(xs, ys) in enumerate(tfe.Iterator(dataset)):
        loss, grads = loss_and_grads(xs, ys)
        if verbose:
            print("Iteration {}: loss = {}".format(i, loss.numpy()))

        optimizer.apply_gradients(grads)

        if logdir:
            with summary_writer.as_default():
                tf.contrib.summary.scalar("loss", loss, step=i)
                tf.contrib.summary.scalar("step", i, step=i)

def synthetic_dataset(w,b, noise_level, batch_size, num_batches):
    """tf.data.Dataset that yields synthetic data for linear regression."""
    return synthetic_dataset_helper(w, b,
                                    tf.shape(w)[0], noise_level, batch_size,
                                    num_batches)

def synthetic_dataset_helper(w, b, num_features, noise_level, batch_size,
                                num_batches):
    """

    # w is a matrix with shape [N, M]
    # b is a vector with shape [M]
    # So:
    # - Generate x's as vectors with shape [batch_size N]
    # - y = tf.matmul(x, W) + b + noise
    """
    def batch(_):
        x = tf.random_normal([batch_size, num_features])
        y = tf.matmul(x, w) + b + noise_level * tf.random_normal([])

        return x, y

    with tf.device("/device:GPU:0"):
        return tf.data.Dataset.range(num_batches).map(batch)

def main(_):
    tf.enable_eager_execution()
    print(tfe.executing_eagerly())   # True
    # Ground truth constants.
    true_w = [[-2.0],[4.0],[1.0]]
    true_b = [0.5]
    noise_level = 0.01

    # Training constants
    batch_size = 64
    learning_rate = 0.1

    print("True w: %s" % true_w)
    print("True b: %s\n" % true_b)

    model = LinearModel()
    dataset = synthetic_dataset(true_w, true_b, noise_level, batch_size, 20)

    device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    print("Using device: %s" % device)
    with tf.device(device):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        fit(model, dataset, optimizer, verbose=True, logdir=FLAGS.logdir)

    print("\nAfter training: w=%s" % model.variables[0].numpy())
    print("\nAfter training: b=%s" % model.variables[1].numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="./graph",
        help="logdir in which Tensorboard summaries will be written (optional).")
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


















