import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import os, time
import subprocess
import argparse
from scipy import misc as misc
from limited_mnist import LimitedMnist
from abstract_network import *

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-n', '--train_size', type=int, default=50000, help='Number of samples for training')
parser.add_argument('-e1', '--epsilon1', type=float, default=114.0)
parser.add_argument('-e2', '--epsilon2', type=float, default=0.15)
parser.add_argument('-m', '--mi', type=float, default=-5.0, help='Information Preference')
parser.add_argument('-l1', '--lambda1', type=float, default=1.0)
parser.add_argument('-l2', '--lambda2', type=float, default=10.0)
parser.add_argument('--lagrangian', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 200


def make_model_path(name):
    if args.lagrangian:
        log_path = os.path.join('log/lagrangian', name)
    else:
        log_path = os.path.join('log/infovae', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path


if args.lagrangian:
    name_append = '_lag'
else:
    name_append = ''

if args.lagrangian:
    log_path = make_model_path('%.2f_%.2f_%.2f%s' % (args.mi, args.epsilon1, args.epsilon2, name_append))
else:
    log_path = make_model_path('%.2f_%.2f_%.2f%s' % (args.mi, args.lambda1, args.lambda2, name_append))

# Encoder and decoder use the DC-GAN architecture
# 28 x 28 x 1
def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        conv = conv2d_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
        conv = conv2d_lrelu(conv, 128, 4, 2)   # None x 7 x 7 x 128
        conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # None x (7x7x128)
        fc = fc_lrelu(conv, 1024)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.05)
        mean = tf.maximum(tf.minimum(mean, 10.0), -10.0)
        return mean, stddev


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 1024)
        fc = fc_relu(fc, 7*7*128)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 7, 7, 128]))
        conv = conv2d_t_relu(conv, 64, 4, 2)
        mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.sigmoid)
        mean = tf.maximum(tf.minimum(mean, 0.995), 0.005)
        return mean


# def encoder(x, z_dim):
#     with tf.variable_scope('encoder'):
#         x = tf.reshape(x, [-1, 784])
#         fc1 = fc_tanh(x, 1024)
#         fc2 = fc_tanh(fc1, 1024)
#         mean = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.identity)
#         stdv = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.exp)
#         stdv = tf.maximum(stdv, 0.05)
#         mean = tf.maximum(tf.minimum(mean, 10.0), -10.0)
#         return mean, stdv
#
#
# def decoder(z, reuse=False):
#     with tf.variable_scope('decoder', reuse=reuse):
#         fc1 = fc_tanh(z, 1024)
#         fc2 = fc_tanh(fc1, 1024)
#         logits = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.identity)
#         logits = tf.reshape(logits, [-1, 28, 28, 1])
#         return tf.minimum(tf.maximum(tf.sigmoid(logits), 0.005), 0.995)


# Build the computation graph for training
z_dim = 5
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None] + x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
zstddev_logdet = tf.reduce_mean(tf.reduce_sum(2.0 * tf.log(train_zstddev), axis=1))

train_xmean = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xmean = decoder(gen_z, reuse=True)


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):   # [batch_size, z_dim] [batch_size, z_dim]
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
loss_mmd = compute_mmd(true_samples, train_z) * 10.0

# ELBO loss divided by input dimensions
elbo_per_sample = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) +
                                0.5 * tf.square(train_zmean) - 0.5, axis=1)
loss_elbo = tf.reduce_mean(elbo_per_sample)

# Negative log likelihood per dimension
nll_per_sample = -tf.reduce_sum(tf.log(train_xmean) * train_x + tf.log(1 - train_xmean) * (1 - train_x),
                                axis=(1, 2, 3))
loss_nll = tf.reduce_mean(nll_per_sample)

if args.lagrangian:
    lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(2.0 * args.mi))
    lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(2.0 * args.mi))
else:
    lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(args.lambda1))
    lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(args.lambda2))
# Cannot simply use clipping here because once at max/min there is no more gradient
lambda1_ub = tf.assign(lambda1, 100.0)
lambda1_lb = tf.assign(lambda1, np.maximum(args.mi + 1e-4, 0.0))
lambda2_ub = tf.assign(lambda2, 100.0)
lambda2_lb = tf.assign(lambda2, 0.0)

loss_all = lambda1 * loss_nll + (lambda2 + args.mi - lambda1) * loss_mmd\
           + (lambda1 - args.mi) * loss_elbo - args.epsilon1 * lambda1 - args.epsilon2 * lambda2

lambda_vars = [lambda1, lambda2]
model_vars = [var for var in tf.global_variables() if 'encoder' in var.name or 'decoder' in var.name]
trainer = tf.train.AdamOptimizer(1e-4).minimize(loss_all, var_list=model_vars)
lambda_update = tf.train.GradientDescentOptimizer(1e-5).minimize(-loss_all, var_list=lambda_vars)

limited_mnist = LimitedMnist(args.train_size)

train_summary = tf.summary.merge([
    tf.summary.scalar('elbo', loss_elbo),
    tf.summary.scalar('nll', loss_nll),
    tf.summary.scalar('mmd', loss_mmd),
    tf.summary.scalar('loss', loss_all),
    tf.summary.scalar('lambda1', lambda1),
    tf.summary.scalar('lambda2', lambda2),
])

sample_summary = tf.summary.merge([
    create_display(tf.reshape(train_x, [100] + x_dim), 'dataset'),
    create_display(tf.reshape(gen_xmean, [100] + x_dim), 'samples'),
    create_display(tf.reshape(train_xmean, [100] + x_dim), 'reconstruction'),
])

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path)

# Start training
# plt.ion()
for i in range(500000):
    bx = limited_mnist.next_batch(batch_size)
    bx = np.reshape(bx, [-1] + x_dim)

    if args.lagrangian:
        sess.run([trainer, lambda_update], feed_dict={train_x: bx})
    else:
        sess.run(trainer, feed_dict={train_x: bx})

    if sess.run(lambda1) < 0.0:
        sess.run(lambda1_lb)
    if sess.run(lambda1) > 100.0:
        sess.run(lambda1_ub)
    if sess.run(lambda2) < 0.0:
        sess.run(lambda2_lb)
    if sess.run(lambda2) > 100.0:
        sess.run(lambda2_ub)

    if i % 100 == 0:
        merged = sess.run(train_summary, feed_dict={train_x: bx})
        summary_writer.add_summary(merged, i)
        if i % 1000 == 0:
            print("Iteration %d" % i)
    if i % 500 == 0:
        bx = limited_mnist.next_batch(100)
        bz = np.random.normal(size=(100, z_dim))
        summary_val = sess.run(sample_summary, feed_dict={train_x: bx, gen_z: bz})
        summary_writer.add_summary(summary_val, i)



