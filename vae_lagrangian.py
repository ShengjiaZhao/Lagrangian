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
import math
from scipy import stats
from eval_ll import LLEvaluator

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-n', '--train_size', type=int, default=60000, help='Number of samples for training')
parser.add_argument('-m', '--mi', type=float, default=0.0, help='Information Preference')
parser.add_argument('-z', '--z', type=int, default=10)
parser.add_argument('-t', '--t', type=str, default='cnn')
parser.add_argument('-s', '--slack', type=float, default=4.0)
parser.add_argument('--no_eval', action='store_true')
args = parser.parse_args()


def make_model_path(name):
    log_path = os.path.join('log/final', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path


log_path = make_model_path('%s/%d/%.2f_%.2f' % (args.t, args.z, args.mi, args.slack))

# python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 200


def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        if args.t == 'cnn':
            conv = conv2d_bn_lrelu(x, 32, 4, 2)   # None x 14 x 14 x 64
            conv = conv2d_bn_lrelu(conv, 32, 4, 1)
            conv = conv2d_bn_lrelu(conv, 64, 4, 2)   # None x 7 x 7 x 128
            conv = conv2d_bn_lrelu(conv, 64, 4, 1)
            conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # None x (7x7x128)
            fc = fc_bn_lrelu(conv, 512)
            mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
            stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
            stddev = tf.maximum(stddev, 0.01)
            return mean, stddev
        else:
            x = tf.reshape(x, [-1, np.prod(x_dim)])
            fc1 = fc_tanh(x, 1024)
            fc2 = fc_tanh(fc1, 1024)
            mean = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.identity)
            stddev = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.sigmoid)
            return mean, stddev


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        if args.t == 'cnn':
            fc1 = fc_bn_relu(z, 512)
            fc2 = fc_bn_relu(fc1, 7*7*32)
            fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 32]))
            conv = conv2d_t_bn_relu(fc2, 64, 4, 2)
            conv = conv2d_t_bn_relu(conv, 32, 4, 1)
            conv = conv2d_t_relu(conv, 32, 4, 2)
            mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 1, activation_fn=tf.identity)
            return mean
        else:
            fc1 = fc_tanh(z, 1024)
            fc2 = fc_tanh(fc1, 1024)
            logits = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.identity)
            logits = tf.reshape(logits, [-1] + x_dim)
            return logits


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


# Build the computation graph for training
z_dim = args.z
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None] + x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))

train_xlogit = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xmean = tf.sigmoid(decoder(gen_z, reuse=True))

# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
loss_mmd = compute_mmd(true_samples, train_z) * 500.0

# ELBO loss divided by input dimensions
zkl_per_sample = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) +
                                0.5 * tf.square(train_zmean) - 0.5, axis=1)
loss_zkl = tf.reduce_mean(zkl_per_sample)

# Negative log likelihood per dimension
nll_per_sample = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_x, logits=train_xlogit), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(nll_per_sample)

zkl_anneal = tf.placeholder_with_default(1.0, [])
loss_elbo = loss_zkl * zkl_anneal + loss_nll

lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(50.0))
lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(50.0))

# Cannot simply use clipping here because once at max/min there is no more gradient
lambda1_clip = tf.assign(lambda1, tf.maximum(tf.minimum(lambda1, 200.0), 0.0))
lambda2_clip = tf.assign(lambda2, tf.maximum(tf.minimum(lambda2, 200.0), 0.0))

epsilon1_ph, epsilon2_ph = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
loss1 = lambda1 * loss_elbo + lambda2 * loss_mmd
loss2 = loss1 - epsilon1_ph * lambda1 - epsilon2_ph * lambda2
if args.mi > 1e-5:
    loss2 += args.mi * loss_nll
elif args.mi < 1e-5:
    loss2 -= args.mi * loss_zkl

lambda_vars = [lambda1, lambda2]
model_vars = [var for var in tf.global_variables() if 'encoder' in var.name or 'decoder' in var.name]
# optimizer = tf.train.AdamOptimizer(1e-4)
# grads = tf.gradients(loss_all, model_vars)
trainer1 = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999).minimize(loss1, var_list=model_vars)
trainer2 = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999).minimize(loss2, var_list=model_vars)
lambda_update = tf.train.GradientDescentOptimizer(5e-3).minimize(-loss2, var_list=lambda_vars)

limited_mnist = LimitedMnist(args.train_size, binary=True)

train_summary = tf.summary.merge([
    tf.summary.scalar('zkl', loss_zkl),
    tf.summary.scalar('nll', loss_nll),
    tf.summary.scalar('elbo', loss_elbo),
    tf.summary.scalar('mmd', loss_mmd),
    tf.summary.scalar('loss1', loss1),
    tf.summary.scalar('loss2', loss2),
    tf.summary.scalar('lambda1', lambda1),
    tf.summary.scalar('lambda2', lambda2),
])

# train_ll_ph, test_ll_ph = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
train_mi_ph, test_mi_ph = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
test_elbo_ph = tf.placeholder(tf.float32)
test_elbo_summary = tf.summary.scalar('test_elbo', test_elbo_ph)
eval_summary = tf.summary.merge([
    # tf.summary.scalar('train_ll', train_ll_ph),
    # tf.summary.scalar('test_ll', test_ll_ph),
    tf.summary.scalar('train_mi', train_mi_ph),
    tf.summary.scalar('test_mi', test_mi_ph),
    test_elbo_summary,
])

sample_summary = tf.summary.merge([
    create_display(tf.slice(gen_xmean, [0, 0, 0, 0], [100] + x_dim), 'samples'),
    create_display(tf.slice(tf.sigmoid(train_xlogit), [0, 0, 0, 0], [100] + x_dim), 'reconstruction'),
])


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path)


# class ModelWrapper:
#     def __init__(self):
#         self.sess = sess
#         self.dataset = limited_mnist
#         self.data_dims = x_dim
#         self.z_dim = z_dim
#
#     def get_generator(self, z):
#         return decoder(z, reuse=True)
#
#
# ll_evaluator = LLEvaluator(model=ModelWrapper())


def compute_test_elbo():
    elbo_vals = []
    for j in range(50):
        bx = limited_mnist.next_test_batch(batch_size)
        elbo_val = sess.run(loss_elbo, feed_dict={train_x: bx})
        elbo_vals.append(elbo_val)
    return np.mean(elbo_vals)


# Start training
# plt.ion()
if args.t == 'cnn':
    initial_steps = 100000
else:
    initial_steps = 200000
for i in range(1, initial_steps):
    bx = limited_mnist.next_batch(batch_size)
    if i < 200:
        reg_val = 0.01
    else:
        reg_val = 1.0
    sess.run(trainer1, feed_dict={train_x: bx, zkl_anneal: reg_val})
    if i % 100 == 0:
        merged, elbo, nll = sess.run([train_summary, loss_zkl, loss_nll], feed_dict={train_x: bx, epsilon1_ph: 0.0, epsilon2_ph: 0.0})
        summary_writer.add_summary(merged, i)
        if i % 500 == 0:
            print("Iteration %d, nll %.4f, elbo loss %.4f" % (i, nll, elbo))

    if i % 500 == 0:
        bx = limited_mnist.next_batch(batch_size)
        bz = np.random.normal(size=(batch_size, z_dim))
        summary_val = sess.run(sample_summary, feed_dict={train_x: bx, gen_z: bz})
        summary_writer.add_summary(summary_val, i)
        test_elbo = compute_test_elbo()
        summary_val = sess.run(test_elbo_summary, feed_dict={test_elbo_ph: test_elbo})
        summary_writer.add_summary(summary_val, i)


def estimate_mi():
    start_time = time.time()
    print("Estimating MI")
    means, stddevs = [], []
    for j in range(200):
        bx = limited_mnist.next_full_batch(batch_size)
        mean, stddev = sess.run([train_zmean, train_zstddev], feed_dict={train_x: bx})
        means.append(mean)
        stddevs.append(stddev)
    train_mean = np.concatenate(means, axis=0)
    train_stddev = np.concatenate(stddevs, axis=0)

    print("Extracted training samples, time elapsed=%.2f" % (time.time() - start_time))

    values = []
    for k in range(1):
        values.append(train_mean + train_stddev * np.random.normal(size=train_stddev.shape))
    values = np.concatenate(values, axis=0)
    kernel = stats.gaussian_kde(values.transpose())

    print("Trained kernel, time elapsed=%.2f" % (time.time() - start_time))

    # Estimate Mutual Information on training set
    means, stddevs = [], []
    for j in range(20):
        bx = limited_mnist.next_test_batch(batch_size)
        mean, stddev = sess.run([train_zmean, train_zstddev], feed_dict={train_x: bx})
        means.append(mean)
        stddevs.append(stddev)
    test_mean = np.concatenate(means, axis=0)
    test_stddev = np.concatenate(stddevs, axis=0)

    print("Extracted testing samples, time elapsed=%.2f" % (time.time() - start_time))

    # log_q_z_x = np.sum(-0.5 * np.log(2 * math.pi * math.e) - np.log(train_stddev), axis=1)
    # log_r_z = np.zeros(shape=(train_mean.shape[0],))
    # for k in range(1):
    #     samples = train_mean + train_stddev * np.random.normal(size=train_stddev.shape)
    #     log_r_z += kernel.logpdf(samples.transpose())
    #     print("Computed training MI iter %d, time elapsed=%.2f" % (k, time.time() - start_time))
    # log_r_z /= 2.0
    # train_mi = np.mean(log_q_z_x - log_r_z)

    log_q_z_x = np.sum(-0.5 * np.log(2 * math.pi * math.e) - np.log(test_stddev), axis=1)
    log_r_z = np.zeros(shape=(test_mean.shape[0],))
    for k in range(1):
        samples = test_mean + test_stddev * np.random.normal(size=test_stddev.shape)
        log_r_z += kernel.logpdf(samples.transpose())
        print("Computed testing MI iter %d, time elapsed=%.2f" % (k, time.time() - start_time))
    log_r_z /= 2.0
    test_mi = np.mean(log_q_z_x - log_r_z)

    train_mi = test_mi
    return train_mi, test_mi


epsilon1, epsilon2 = 0.0, 0.0
for i in range(100):
    bx = limited_mnist.next_batch(batch_size)
    elbo_val, mmd_val = sess.run([loss_elbo, loss_mmd], feed_dict={train_x: bx})
    epsilon1 += elbo_val
    epsilon2 += mmd_val
epsilon1 /= 100.0
epsilon2 /= 100.0
epsilon1 += args.slack
epsilon2 *= 1.2

# Start training
# plt.ion()
next_eval = initial_steps
for i in range(initial_steps, 1000000):
    if i == next_eval and not args.no_eval:
        # ll_evaluator.train()
        # train_ll, test_ll = ll_evaluator.compute_ll(10)
        train_mi, test_mi = estimate_mi()
        test_elbo = compute_test_elbo()
        # print(train_ll, test_ll, train_mi, test_mi)
        summary_val = sess.run(eval_summary,
                               feed_dict={train_mi_ph: train_mi, test_mi_ph: test_mi,
                                          test_elbo_ph: test_elbo})
        # summary_val = sess.run(eval_summary,
        #                        feed_dict={train_ll_ph: train_ll, test_ll_ph: test_ll,
        #                                   train_mi_ph: train_mi, test_mi_ph: test_mi,
        #                                   test_elbo_ph: test_elbo})
        summary_writer.add_summary(summary_val, i)
        next_eval = int(next_eval * 1.35 + 10000)

    bx = limited_mnist.next_batch(batch_size)

    sess.run([trainer2, lambda_update], feed_dict={train_x: bx, epsilon1_ph: epsilon1, epsilon2_ph: epsilon2})
    sess.run([lambda1_clip, lambda2_clip])

    if i % 100 == 0:
        merged = sess.run(train_summary, feed_dict={train_x: bx, epsilon1_ph: epsilon1, epsilon2_ph: epsilon2})
        summary_writer.add_summary(merged, i)
        if i % 1000 == 0:
            print("Iteration %d" % i)

    if i % 500 == 0:
        bx = limited_mnist.next_batch(batch_size)
        bz = np.random.normal(size=(batch_size, z_dim))
        summary_val = sess.run(sample_summary,
                               feed_dict={train_x: bx, gen_z: bz, epsilon1_ph: epsilon1, epsilon2_ph: epsilon2})
        summary_writer.add_summary(summary_val, i)


    # if i == interval:
    #     is_nll = 0.0
    #     for j in range(40):
    #         test_data = limited_mnist.test_batch(200)
    #         is_nll += compute_nll_by_is(test_data, sess=sess, verbose=True)
    #     is_nll /= 40.0
    #     summary_val = sess.run(nll_summary, feed_dict={nll_ph: is_nll})
    #     summary_writer.add_summary(summary_val, i)
    #     interval = interval * 1.4 + 2000
