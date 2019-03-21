import os
import sys
import time
import logger
import pickle
import importlib

import numpy as np
import tensorflow as tf
from data_iter import DataIterator

if len(sys.argv) < 3:
    sys.exit("Usage: train_rnn.py <configuration_name> <train data filename>")

# data preparation
config_name = sys.argv[1]
data_path = sys.argv[2]

config = importlib.import_module('configurations.%s' % config_name)
experiment_id = '%s-%s-%s' % (
    config_name.split('.')[-1], os.path.basename(data_path.split('.')[0]),
    time.strftime("%Y%m%d-%H%M%S", time.localtime()))
print(experiment_id)

# metadata
if not os.path.isdir('metadata'):
    os.makedirs('metadata')
metadata_target_path = 'metadata'

# logs
if not os.path.isdir('logs'):
        os.makedirs('logs')
sys.stdout = logger.Logger('logs/%s.log' % experiment_id)
sys.stderr = sys.stdout

# load data
with open(data_path, 'r') as f:
    data = f.read()

# construct symbol set
tokens_set = set(data.split())
start_symbol, end_symbol = '<s>', '</s>'
tokens_set.update({start_symbol, end_symbol})

# construct token to number dictionary
idx2token = list(tokens_set)
vocab_size = len(idx2token)
print('vocabulary size:', vocab_size)
token2idx = dict(zip(idx2token, range(vocab_size)))
tunes = data.split('\n\n')
del data

# transcribe tunes from symbol to index
tunes = [[token2idx[c] for c in [start_symbol] + t.split() + [end_symbol]] for t in tunes]
tunes.sort(key=lambda x: len(x), reverse=True)
ntunes = len(tunes)

tune_lens = np.array([len(t) for t in tunes])
max_len = max(tune_lens)

# tunes for validation
nvalid_tunes = ntunes * config.validation_fraction
nvalid_tunes = int(config.batch_size * max(1, np.rint(
    nvalid_tunes / float(config.batch_size))))  # round to the multiple of batch_size

rng = np.random.RandomState(42)
valid_idxs = rng.choice(np.arange(ntunes), nvalid_tunes, replace=False)

# tunes for training
ntrain_tunes = ntunes - nvalid_tunes
train_idxs = np.delete(np.arange(ntunes), valid_idxs)

print('n tunes:', ntunes)
print('n train tunes:', ntrain_tunes)
print('n validation tunes:', nvalid_tunes)
print('min, max length', min(tune_lens), max(tune_lens))

x = tf.placeholder(tf.int32, [config.batch_size, None])
mask = tf.placeholder(tf.float32, [config.batch_size, None])
seq_length = tf.placeholder(tf.int32, [config.batch_size])
prob_dropout = tf.placeholder_with_default(0.0, shape=())

# model graph
def build_graph(x, mask, seq_length, prob_dropout, batch_size):

    W_emb = tf.Variable(tf.eye(vocab_size, dtype=tf.float32), name='w_emb')
    l_emb = tf.nn.embedding_lookup(W_emb, x)

    # change this in order to take the sequence length
    cells = [
        tf.nn.rnn_cell.LSTMCell(config.rnn_size),
        tf.nn.rnn_cell.LSTMCell(config.rnn_size),
        tf.nn.rnn_cell.LSTMCell(config.rnn_size)
    ]

    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    initial_state = cell.zero_state(batch_size, tf.float32)

    rnn_output, _ = tf.nn.dynamic_rnn(cell, 
                                      l_emb, 
                                      initial_state=initial_state, 
                                      sequence_length=seq_length) 

    if config.dropout > 0:
        rnn_output = tf.keras.layers.Dropout(prob_dropout)(rnn_output)

    l_reshp = tf.reshape(rnn_output, (-1, config.rnn_size))

    l_out = tf.keras.layers.Dense(units=vocab_size, 
                                  kernel_initializer=tf.initializers.orthogonal, 
                                  activation=tf.nn.softmax)(l_reshp)

    # output overall params
    print("total parameters", np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
    # output layer type, num_param, output_shape
    print('Embedding shape', l_emb.get_shape())
    print('RNN output shape', rnn_output.get_shape())
    print('Reshape shape', l_reshp.get_shape())
    print('Dense shape', l_out.get_shape())

    y = tf.keras.backend.flatten(x[:, 1:])

    # training loss
    # predictions, dim_0: sample number, dim_1: the next token in the sample
    index = tf.stack([tf.range(tf.shape(y)[0]), y], axis=1)
    p1 = tf.reshape(tf.log(tf.gather_nd(l_out, index)), tf.shape(mask))
    # mask the short sequence value , 
    # sum on the timestamp axis, and then sum on the sample axis
    loss = -1. * tf.reduce_mean(tf.reduce_sum(mask * p1, axis=1), axis=0)

    # optimizer
    var_lr = tf.Variable(config.learning_rate ,trainable=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=var_lr)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, config.grad_clipping)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    return train_op, loss, var_lr

# create batch
def create_batch(idxs):
    max_seq_len = max([len(tunes[i]) for i in idxs])
    x = np.zeros((config.batch_size, max_seq_len), dtype='float32')
    sequence_length = np.zeros((config.batch_size,), dtype='int32')
    mask = np.zeros((config.batch_size, max_seq_len - 1), dtype='float32')
    for i, j in enumerate(idxs):
        x[i, :tune_lens[j]] = tunes[j]
        sequence_length[i] = tune_lens[j]
        mask[i, : tune_lens[j] - 1] = 1
    return x, sequence_length, mask

train_data_iterator = DataIterator(tune_lens[train_idxs], train_idxs, config.batch_size, random_lens=False)
valid_data_iterator = DataIterator(tune_lens[valid_idxs], valid_idxs, config.batch_size, random_lens=False)

print('Train model')
train_batches_per_epoch = ntrain_tunes / config.batch_size
max_niter = config.max_epoch * train_batches_per_epoch
losses_train = []

nvalid_batches = nvalid_tunes / config.batch_size
losses_eval_valid = []
niter = 0
start_epoch = 0
prev_time = time.clock()

train_op, loss_op, var_lr = build_graph(x, mask, seq_length, prob_dropout, config.batch_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:

    sess.run(init)

    # restore the checkpoint
    if not os.path.exists(metadata_target_path):
        print('[!] Checkpoints path does not exist...')
        quit()

    ckpt = tf.train.get_checkpoint_state(metadata_target_path)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(metadata_target_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))

    for epoch in range(start_epoch, config.max_epoch):
        for train_batch_idxs in train_data_iterator:
            x_batch, sequence_length, mask_batch = create_batch(train_batch_idxs)
            _, train_loss = sess.run([train_op, loss_op], feed_dict={x: x_batch, 
                                                                     mask: mask_batch, 
                                                                     seq_length: sequence_length,
                                                                     prob_dropout: config.dropout})
            current_time = time.clock()

            print('%d/%d (epoch %.3f) train_loss=%6.8f time/batch=%.2fs' % (
                niter, max_niter, niter / float(train_batches_per_epoch), train_loss, current_time - prev_time))

            prev_time = current_time
            losses_train.append(train_loss)
            niter += 1

            if niter % config.validate_every == 0:
                print('Validating')
                avg_valid_loss = 0
                for valid_batch_idx in valid_data_iterator:
                    x_batch, sequence_length, mask_batch = create_batch(valid_batch_idx)
                    avg_valid_loss += sess.run(loss_op, feed_dict = {x: x_batch,
                                                                     mask: mask_batch,
                                                                     seq_length: sequence_length,
                                                                     prob_dropout: 0.0})
                    # avg_valid_loss += validate(x_batch, x_batch)
                avg_valid_loss /= nvalid_batches
                losses_eval_valid.append(avg_valid_loss)
                print("    loss:\t%.6f" % avg_valid_loss)
                print

        if epoch > config.learning_rate_decay_after:
            new_learning_rate = np.float32(sess.run(var_lr) * config.learning_rate_decay)
            sess.run(tf.assign(var_lr, new_learning_rate))
            print('setting learning rate to %.7f' % new_learning_rate)

        if (epoch + 1) % config.save_every == 0:
            saver.save(sess, os.path.join(metadata_target_path, experiment_id), global_step=niter)