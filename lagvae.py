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
parser.add_argument('-z', type=int, default=5)
parser.add_argument('-t', type=str, default='cnn')
parser.add_argument('--lagrangian', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 100


def make_model_path(name):
    if args.lagrangian:
        log_path = os.path.join('log/lagrangian', name)
    else:
        log_path = os.path.join('log/infovae_adam', name)
    print(log_path)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path


if args.lagrangian:
    name_append = '_lag'
else:
    name_append = ''

if args.lagrangian:
    log_path = make_model_path('%s/%d/%.2f_%.2f_%.2f%s' % (args.t, args.z, args.mi, args.epsilon1, args.epsilon2, name_append))
else:
    log_path = make_model_path('%s/%d/%.2f_%.2f_%.2f%s' % (args.t, args.z, args.mi, args.lambda1, args.lambda2, name_append))

# Encoder and decoder use the DC-GAN architecture
# 28 x 28 x 1
# def encoder(x, z_dim):
#     if args.t == 'cnn':
#         with tf.variable_scope('encoder'):
#             conv = conv2d_bn_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
#             conv = conv2d_bn_lrelu(conv, 128, 4, 2)   # None x 7 x 7 x 128
#             conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # None x (7x7x128)
#             fc = fc_bn_lrelu(conv, 1024)
#             mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
#             stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.exp)
#             return mean, stddev
#     elif args.t == 'cnns':
#         with tf.variable_scope('encoder'):
#             conv = conv2d_bn_lrelu(x, 32, 4, 2)   # None x 14 x 14 x 64
#             conv = conv2d_bn_lrelu(conv, 64, 4, 2)   # None x 7 x 7 x 128
#             conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # None x (7x7x128)
#             fc = fc_bn_lrelu(conv, 1024)
#             mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
#             stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.exp)
#             # stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
#             # stddev = tf.maximum(stddev, 0.05)
#             # mean = tf.maximum(tf.minimum(mean, 10.0), -10.0)
#             return mean, stddev
#     else:
#         with tf.variable_scope('encoder'):
#             x = tf.reshape(x, [-1, 784])
#             fc1 = fc_tanh(x, 1024)
#             fc2 = fc_tanh(fc1, 1024)
#             mean = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.identity)
#             stdv = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.exp)
#             return mean, stdv
#
#
# def decoder(z, reuse=False):
#     if args.t == 'cnn':
#         with tf.variable_scope('decoder') as vs:
#             if reuse:
#                 vs.reuse_variables()
#             fc = fc_bn_relu(z, 1024)
#             fc = fc_bn_relu(fc, 7*7*128)
#             conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 7, 7, 128]))
#             conv = conv2d_t_bn_relu(conv, 64, 4, 2)
#             mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.identity)
#             # mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.sigmoid)
#             # mean = tf.maximum(tf.minimum(mean, 0.995), 0.005)
#             return mean
#     elif args.t == 'cnns':
#         with tf.variable_scope('decoder') as vs:
#             if reuse:
#                 vs.reuse_variables()
#             fc = fc_bn_relu(z, 1024)
#             fc = fc_bn_relu(fc, 7*7*64)
#             conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 7, 7, 64]))
#             conv = conv2d_t_bn_relu(conv, 32, 4, 2)
#             mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.identity)
#             # mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.sigmoid)
#             # mean = tf.maximum(tf.minimum(mean, 0.995), 0.005)
#             return mean
#     else:
#         with tf.variable_scope('decoder', reuse=reuse):
#             fc1 = fc_tanh(z, 1024)
#             fc2 = fc_tanh(fc1, 1024)
#             logits = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.identity)
#             # logits = tf.reshape(logits, [-1, 28, 28, 1])
#             return logits # tf.minimum(tf.maximum(tf.sigmoid(logits), 0.005), 0.995)

def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        fc1 = fc_tanh(x, 1024)
        fc2 = fc_tanh(fc1, 1024)
        mean = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.identity)
        stdv = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.exp)
        # stdv = tf.maximum(stdv, 0.1)
        return mean, stdv


def decoder(z, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
        fc1 = fc_tanh(z, 1024)
        fc2 = fc_tanh(fc1, 1024)
        logits = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.identity)
        return logits


# Build the computation graph for training
z_dim = args.z
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None] + [784])
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev, tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
zstddev_logdet = tf.reduce_mean(tf.reduce_sum(2.0 * tf.log(train_zstddev + 1e-8), axis=1))

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


def stein_gradient(x, x_grad, sigma):
    x_size = x.get_shape()[0].value
    x_dim = x.get_shape()[1].value
    xm = tf.reshape(x, [1, x_size, x_dim])
    xm = tf.tile(xm, (x_size, 1, 1))
    ym = tf.reshape(x, [x_size, 1, x_dim])
    ym = tf.tile(ym, (1, x_size, 1))
    km = tf.exp(-tf.reduce_sum(tf.square(xm - ym), axis=2) / (2.0 * sigma * sigma * x_dim))
    tk = tf.tile(tf.expand_dims(km, axis=2), [1, 1, x_dim])
    km_grad = -(ym - xm) / (sigma * sigma * x_dim) * tk
    km_grad = tf.reduce_sum(km_grad, axis=1)
    return (tf.matmul(km, x_grad) + km_grad) / x_size


# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
# ELBO loss divided by input dimensions
elbo_per_sample = tf.reduce_sum(-tf.log(1e-8 + train_zstddev) + 0.5 * tf.square(train_zstddev) +
                                0.5 * tf.square(train_zmean) - 0.5, axis=1)
# Negative log likelihood per dimension
nll_per_sample = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_x, logits=train_xmean), axis=(1,))
                 # -tf.reduce_sum(tf.log(train_xmean) * train_x + tf.log(1 - train_xmean) * (1 - train_x),
                 #                axis=(1, 2, 3))

if args.lagrangian:
    lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
else:
    lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(args.lambda1))
    lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(args.lambda2))

# Cannot simply use clipping here because once at max/min there is no more gradient
# lambda1_ub = tf.assign(lambda1, tf.minimum(lambda1, 100.0))
# lambda1_lb = tf.assign(lambda1, tf.maximum(lambda1, 0.0))
# lambda2_ub = tf.assign(lambda2, tf.minimum(lambda2, 100.0))
# lambda2_lb = tf.assign(lambda2, tf.maximum(lambda2, 0.0))
lambda1_clip = tf.assign(lambda1, tf.minimum(tf.maximum(lambda1, 0.0), 100.0))
lambda2_clip = tf.assign(lambda1, tf.minimum(tf.maximum(lambda1, 0.0), 100.0))
lambda_clip = tf.group(lambda1_clip, lambda2_clip) # [lambda1_ub, lambda1_lb, lambda2_ub, lambda2_lb]

loss_nll = tf.reduce_mean(nll_per_sample)
loss_elbo = tf.reduce_mean(elbo_per_sample)
loss_mmd = compute_mmd(true_samples, train_z) * 10.0
if args.mi < 0:
    loss_all = lambda1 * loss_nll + (lambda1 - args.mi) * loss_elbo + lambda2 * loss_mmd - args.epsilon1 * lambda1 - args.epsilon2 * lambda2
elif args.mi > 0:
    loss_all = (lambda1 + args.mi) * loss_nll + lambda1 * loss_elbo + lambda2 * loss_mmd - args.epsilon1 * lambda1 - args.epsilon2 * lambda2
else:
    loss_all = lambda1 * loss_nll + lambda1 * loss_elbo + lambda2 * loss_mmd - args.epsilon1 * lambda1 - args.epsilon2 * lambda2

lambda_vars = [lambda1, lambda2]
model_vars = [var for var in tf.global_variables() if 'encoder' in var.name or 'decoder' in var.name]
lr = tf.placeholder(tf.float32, shape=[])
trainer = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999).minimize(loss_all, var_list=model_vars)
lambda_update = tf.train.GradientDescentOptimizer(lr).minimize(-loss_all, var_list=lambda_vars)

limited_mnist = LimitedMnist(args.train_size, binary=True)

train_summary = tf.summary.merge([
    tf.summary.scalar('train/elbo', loss_elbo),
    tf.summary.scalar('train/nll', loss_nll),
    tf.summary.scalar('train/mmd', loss_mmd),
    tf.summary.scalar('train/loss', loss_all),
    tf.summary.scalar('train/lambda1', lambda1),
    tf.summary.scalar('train/lambda2', lambda2),
    tf.summary.scalar('train/vae_loss', loss_elbo + loss_nll)
])

test_summary = tf.summary.merge([
    tf.summary.scalar('test/elbo', loss_elbo),
    tf.summary.scalar('test/nll', loss_nll),
    tf.summary.scalar('test/mmd', loss_mmd),
    tf.summary.scalar('test/loss', loss_all),
    tf.summary.scalar('test/lambda1', lambda1),
    tf.summary.scalar('test/lambda2', lambda2),
    tf.summary.scalar('test/vae_loss', loss_elbo + loss_nll)
])


sample_summary = tf.summary.merge([
    create_display(tf.reshape(train_x, [100] + x_dim), 'dataset'),
    create_display(tf.reshape(tf.sigmoid(gen_xmean), [100] + x_dim), 'samples'),
    create_display(tf.reshape(tf.sigmoid(train_xmean), [100] + x_dim), 'reconstruction'),
])

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path)

# Start training
# plt.ion()
lr_ = 1e-4
epoch = 500 * 5
for i in range(500000):
    bx = limited_mnist.next_full_batch(batch_size)
    bx = np.reshape(bx, [-1] + [784])

    if args.lagrangian:
        sess.run([trainer, lambda_update], feed_dict={train_x: bx, lr: lr_})
    else:
        sess.run(trainer, feed_dict={train_x: bx, lr: lr_})
    sess.run(lambda_clip)

    if i % 100 == 0:
        merged = sess.run(train_summary, feed_dict={train_x: bx})
        summary_writer.add_summary(merged, i)
        elbo, nll = sess.run([loss_elbo, loss_nll],
                     feed_dict={train_x: bx})
        print("Iteration %d: all %.4f nll %.4f elbo %.4f" % (i, nll + elbo, nll, elbo))

    if i % 500 == 0:
        bx = limited_mnist.next_batch(100)
        bx = np.reshape(bx, [-1] + [784])
        bz = np.random.normal(size=(100, z_dim))
        summary_val = sess.run(sample_summary, feed_dict={train_x: bx, gen_z: bz})
        summary_writer.add_summary(summary_val, i)

    if i % 1000 == 0:
        bx = limited_mnist.next_test_batch(100)
        bx = np.reshape(bx, [-1] + [784])
        merged = sess.run(test_summary, feed_dict={train_x: bx})
        summary_writer.add_summary(merged, i)

    if i == epoch:
        lr_ = lr_ * 0.1
        epoch = epoch * 5
        print('learning rate {}'.format(lr_))