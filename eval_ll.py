import argparse
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
from abstract_network import *
import tensorflow as tf
from scipy.stats import entropy


def variational_posterior(x, z_dim):
    with tf.variable_scope('vi_posterior'):
        conv = conv2d_bn_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
        conv = conv2d_bn_lrelu(conv, 128, 4, 2)   # None x 7 x 7 x 128
        conv = conv2d_bn_lrelu(conv, 128, 4, 1)
        conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # None x (7x7x128)
        fc1 = fc_bn_lrelu(conv, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.05)
        mean = tf.maximum(tf.minimum(mean, 10.0), -10.0)
        return mean, stddev


# Evaluate the log likelihood on test data
def compute_log_sum(val):
    min_val = np.min(val, axis=0, keepdims=True)
    return np.mean(min_val - np.log(np.mean(np.exp(-val + min_val), axis=0)))


class LLEvaluator:
    def __init__(self, model):
        """ model must have the following attributes:
            model.get_generator(z): given input z produce the corresponding generated samples/sample mean
            model.sess: holds the parameters of the model
            model.dataset
            model.data_dims, model.z_dim: hold the dimensionality of x and z
        """
        self.model = model
        self.sess = model.sess
        self.x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        z_mean, z_stddev = variational_posterior(self.x, model.z_dim)
        z_sample = z_mean + tf.multiply(z_stddev,
                                        tf.random_normal(tf.stack([tf.shape(self.x)[0], model.z_dim])))
        x_logit = model.get_generator(z_sample)

        nll_per_sample = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=x_logit), axis=(1, 2, 3))
        self.loss_nll = tf.reduce_mean(nll_per_sample)

        # ELBO loss divided by input dimensions
        elbo_per_sample = tf.reduce_sum(-tf.log(z_stddev) + 0.5 * tf.square(z_stddev) + 0.5 * tf.square(z_mean) - 0.5, axis=1)
        self.loss_elbo = tf.reduce_mean(elbo_per_sample)

        self.is_estimator = nll_per_sample + elbo_per_sample

        vi_var = [var for var in tf.global_variables() if 'vi_posterior' in var.name]
        with tf.variable_scope('adam_train'):
            self.vi_train = tf.train.AdamOptimizer(1e-4).minimize(self.loss_nll + self.loss_elbo, var_list=vi_var)
        adam_var = [var for var in tf.global_variables() if 'adam_train' in var.name]

        self.init_op = tf.variables_initializer(vi_var + adam_var)

    def train(self):
        self.sess.run(self.init_op)
        # Learn variational inference on test set
        for iter in range(10000):
            bx = self.model.dataset.next_test_batch(100)
            nll, elbo, _ = self.sess.run([self.loss_nll, self.loss_elbo, self.vi_train], feed_dict={self.x: bx})
            if iter % 2000 == 0:
                print("VI iteration %d, nll=%.4f, elbo=%.4f" % (iter, nll, elbo))

    def compute_ll(self, num_batch=50):
        train_nll = []
        test_nll = []
        start_time = time.time()
        for i in range(num_batch):
            batch_x = self.model.dataset.next_test_batch(100)
            nll = self.compute_nll_by_is(batch_x)
            test_nll.append(nll)
            batch_x = self.model.dataset.next_batch(100)
            nll = self.compute_nll_by_is(batch_x)
            train_nll.append(nll)
            if i % 5 == 0:
                print("Nll is %.4f/%.4f, time elapsed %.2f" % (np.mean(train_nll), np.mean(test_nll), time.time() - start_time))
        return np.mean(train_nll), np.mean(test_nll)

    def compute_nll_by_is(self, batch_x, verbose=False):
        start_time = time.time()
        nll_list = []
        num_iter = 2000
        for k in range(num_iter):
            nll = self.sess.run(self.is_estimator, feed_dict={self.x: batch_x})
            nll_list.append(nll)
            if verbose and k % 500 == 0:
                print("Iter %d, current value %.4f, time used %.2f" % (
                    k, compute_log_sum(np.stack(nll_list)), time.time() - start_time))
        return compute_log_sum(np.stack(nll_list))

