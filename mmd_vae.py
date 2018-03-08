import tensorflow as tf
import numpy as np
import os, time
import subprocess
import argparse
from scipy import misc as misc
import math
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# Train on limited data
class LimitedMnist:
    def __init__(self):
        self.full_mnist = input_data.read_data_sets('mnist_data')

    def next_batch(self, batch_size):
        ret = self.full_mnist.train.next_batch(batch_size)[0]
        r = np.random.uniform(size=ret.shape)
        return (ret > r).astype(np.float32)

    def next_test_batch(self, batch_size):
        ret = self.full_mnist.test.next_batch(batch_size)[0]
        r = np.random.uniform(size=ret.shape)
        return (ret > r).astype(np.float32)


class LimitedBernoilli:
    def __init__(self):
        self.dim = 784
        self.size = 500
        self.full_mnist = input_data.read_data_sets('mnist_data')
        self.ret = self.full_mnist.train.next_batch(self.size)[0]
        self.ent = -self.ret * np.log(self.ret + 1e-8) - (1 - self.ret) * np.log(1 - self.ret + 1e-8)
        self.ent = np.mean(np.sum(self.ent, axis=(1,)))
        print('entropy %f' % self.ent)

    def next_batch(self, batch_size):
        r = np.random.uniform(size=[batch_size, self.dim])
        idx = np.random.choice(self.size, batch_size)
        ret = self.ret[idx]
        return (ret > r).astype(np.float32)

    def next_test_batch(self, batch_size):
        return self.next_batch(batch_size)


class RunningAvgLogger:
    def __init__(self, log_path, max_step=20):
        self.log_path = log_path
        self.max_step = max_step
        self.items = {}

    def __exit__(self):
        self.flush()

    def add_item(self, name, value):
        if math.isnan(value):
            return False
        if name not in self.items:
            self.items[name] = {'val': float(value), 'count': 1}
        else:
            self.items[name]['count'] += 1
            self.items[name]['val'] = self.update_running_avg(self.items[name]['val'], float(value), self.items[name]['count'])
        return True

    def update_running_avg(self, avg, new_val, avg_count=None):
        if avg_count is None:
            avg_count = self.max_step
        if avg_count > self.max_step:
            avg_count = self.max_step
        return avg * (avg_count - 1.0) / avg_count + new_val * (1.0 / avg_count)

    def flush(self):
        writer = open(self.log_path, 'w')
        for name in self.items:
            writer.write("%s %f %d\n" % (name, self.items[name]['val'], self.items[name]['count']))
        writer.close()


parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('-z', '--zdim', type=int, default=50, help='z dimension')
parser.add_argument('-m', '--mi', type=float, default=0.0, help='Information Preference')  # [-inf .. 0.0] [0.0 ... 1.0]
parser.add_argument('-l1', '--lambda1', type=float, default=1.0)
parser.add_argument('-l2', '--lambda2', type=float, default=0.0)
parser.add_argument('-e1', '--epsilon1', type=float, default=0.0)
parser.add_argument('-e2', '--epsilon2', type=float, default=0.0)
parser.add_argument('-t', type=str, default='mnist')
parser.add_argument('--lagrangian', action='store_true')
parser.add_argument('--cont', action='store_true')
args = parser.parse_args()


# python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 100


def make_model_path(name):
    if args.lagrangian:
        log_path = os.path.join('log/lagrangian', name)
    else:
        log_path = os.path.join('log/infovae', name)
    if os.path.isdir(log_path):
        subprocess.call(('mv %s/log.txt %s/log.txt.bak' % (log_path, log_path)).split())
    try:
        os.makedirs(log_path)
    except FileExistsError:
        pass
    return log_path


if args.lagrangian:
    log_path = make_model_path('%s/%d/%.2f_%.2f_%.2f' % (args.t, args.zdim, args.mi, args.epsilon1, args.epsilon2))
else:
    log_path = make_model_path('%s/%d/%.2f_%.2f_%.2f' % (args.t, args.zdim, args.mi, args.lambda1, args.lambda2))


def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv


def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv


def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc


def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


def fc_tanh(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.tanh(fc)
    return fc


def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        fc1 = fc_tanh(x, 1024)
        fc2 = fc_tanh(fc1, 1024)
        mean = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.identity)
        stdv = tf.contrib.layers.fully_connected(fc2, z_dim, activation_fn=tf.exp)
        return mean, stdv


def decoder(z, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
        fc1 = fc_tanh(z, 1024)
        fc2 = fc_tanh(fc1, 1024)
        logits = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.identity)
        return logits


z_dim = args.zdim
x_dim = [784]  # [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None]+x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
zstddev_logdet = tf.reduce_mean(tf.reduce_sum(2.0 * tf.log(train_zstddev), axis=1))

train_logits = decoder(train_z)
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_logits = decoder(gen_z, reuse=True)


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


true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
loss_mmd = compute_mmd(true_samples, train_z)

loss_elbo_per_sample = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) + 0.5 * tf.square(train_zmean) - 0.5,
                          axis=1)
loss_elbo = tf.reduce_mean(loss_elbo_per_sample)

# Negative log likelihood per dimension
loss_nll_per_sample = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_x, logits=train_logits), axis=(1))
loss_nll = tf.reduce_mean(loss_nll_per_sample)

# negative log likelihood measured by sampling
sample_nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=train_x, logits=gen_logits)
sample_nll = tf.reduce_sum(sample_nll, axis=(1))

# negative log likelihood measured by is
is_nll = loss_elbo_per_sample + loss_nll_per_sample


if args.lagrangian:
    lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
else:
    lambda1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(args.lambda1))
    lambda2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(args.lambda2))

lambda1_clip = tf.assign(lambda1, tf.minimum(tf.maximum(lambda1, 0.0), 100.0))
lambda2_clip = tf.assign(lambda1, tf.minimum(tf.maximum(lambda1, 0.0), 100.0))
lambda_clip = tf.group(lambda1_clip, lambda2_clip)

if args.lagrangian:
    epsilon1 = args.epsilon1
    epsilon2 = args.epsilon2
else:
    epsilon1 = 1.0
    epsilon2 = 1.0


if args.mi < 0:
    loss_all = lambda1 * loss_nll + (lambda1 - args.mi) * loss_elbo + lambda2 * loss_mmd - epsilon1 * lambda1 - epsilon2 * lambda2
elif args.mi > 0:
    loss_all = (lambda1 + args.mi) * loss_nll + lambda1 * loss_elbo + lambda2 * loss_mmd - epsilon1 * lambda1 - epsilon2 * lambda2
else:
    loss_all = lambda1 * loss_nll + lambda1 * loss_elbo + lambda2 * loss_mmd - epsilon1 * lambda1 - epsilon2 * lambda2

# loss_all = loss_nll + (1.0 - args.mi) * loss_elbo

learning_rate = tf.placeholder(tf.float32, [])

lambda_vars = [lambda1, lambda2]
model_vars = [var for var in tf.global_variables() if 'encoder' in var.name or 'decoder' in var.name]
trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss_all, var_list=model_vars)
if args.lagrangian:
    lambda_update = tf.train.GradientDescentOptimizer(1e-4).minimize(-loss_all, var_list=lambda_vars)
else:
    lambda_update = None

logger = RunningAvgLogger(os.path.join(log_path, 'log.txt'), max_step=50)

if args.t == 'mnist':
    data = LimitedMnist()
else:
    data = LimitedBernoilli()

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


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples, max_samples=100):
    if max_samples > samples.shape[0]:
        max_samples = samples.shape[0]
    cnt, height, width = int(math.floor(math.sqrt(max_samples))), samples.shape[1], samples.shape[2]
    samples = samples[:cnt*cnt]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


def compute_z_logdet(is_train=True):
    z_list = []
    for k in range(50):
        if is_train:
            batch_x = data.next_batch(batch_size)
        else:
            batch_x, _ = data.next_test_batch(batch_size)
        batch_x = np.reshape(batch_x, [-1]+x_dim)
        z = sess.run(train_z, feed_dict={train_x: batch_x})
        z_list.append(z)
    z_list = np.concatenate(z_list, axis=0)
    cov = np.cov(z_list.T)
    sign, logdet = np.linalg.slogdet(cov)
    return logdet


saver = tf.train.Saver()

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path)

if not args.cont:
    for i in range(500000):
        if i < 10000:
            lr = 0.001
        elif i >= 100000:
            lr = 1e-5
        else:
            lr = 1e-4

        bx = data.next_batch(batch_size)
        bx = np.reshape(bx, [-1] + [784])

        if args.lagrangian:
            sess.run([trainer, lambda_update], feed_dict={train_x: bx, learning_rate: lr})
        else:
            sess.run(trainer, feed_dict={train_x: bx, learning_rate: lr})
        sess.run(lambda_clip)

        if i % 100 == 0:
            merged = sess.run(train_summary, feed_dict={train_x: bx})
            summary_writer.add_summary(merged, i)
            elbo, nll = sess.run([loss_elbo, loss_nll],
                                 feed_dict={train_x: bx})
            print("Iteration %d: all %.4f nll %.4f elbo %.4f" % (i, nll + elbo, nll, elbo))

        if i % 1000 == 0:
            bx = data.next_test_batch(100)
            bx = np.reshape(bx, [-1] + [784])
            merged = sess.run(test_summary, feed_dict={train_x: bx})
            summary_writer.add_summary(merged, i)

    saver.save(sess=sess, save_path=log_path + '/model.ckpt')
else:
    saver.restore(sess=sess, save_path=log_path + '/model.ckpt')

