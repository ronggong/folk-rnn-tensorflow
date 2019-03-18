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
metadata_target_path = 'metadata/%s.pkl' % experiment_id

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

# model graph
def build_graph(x, mask):
    W_emb = tf.Variable(tf.eye(vocab_size, dtype=tf.float32), name='w_emb')
    l_emb = tf.nn.embedding_lookup(W_emb, x)

    l_mask = mask

    cells = [
        tf.keras.layers.LSTMCell(units=config.rnn_size),
        tf.keras.layers.LSTMCell(units=config.rnn_size),
        tf.keras.layers.LSTMCell(units=config.rnn_size)
    ]
    rnn_output = tf.keras.layers.RNN(cells)(l_emb)

    if config.dropout > 0:
        rnn_output = tf.keras.layers.Dropout(config.dropout)(rnn_output)

    l_reshp = tf.reshape(rnn_output, (-1, config.rnn_size))
    l_out = tf.keras.layers.Dense(units=vocab_size, kernel_initializer=tf.initializers.orthogonal, activation=tf.nn.softmax)
    predictions = l_out

    y = tf.keras.layers.Flatten()(x[:, 1:])

# create batch
def create_batch(idxs):
    max_seq_len = max([len(tunes[i]) for i in idxs])
    x = np.zeros((config.batch_size, max_seq_len), dtype='float32')
    mask = np.zeros((config.batch_size, max_seq_len - 1), dtype='float32')
    for i, j in enumerate(idxs):
        x[i, :tune_lens[j]] = tunes[j]
        mask[i, : tune_lens[j] - 1] = 1
    return x, mask

train_data_iterator = DataIterator(tune_lens[train_idxs], train_idxs, config.batch_size, random_lens=False)
valid_data_iterator = DataIterator(tune_lens[valid_idxs], valid_idxs, config.batch_size, random_lens=False)

print('Train model')
train_batches_per_epoch = ntrain_tunes / config.batch_size
max_niter = config.max_epoch * train_batches_per_epoch
losses_train = []

nvalid_batches = nvalid_tunes / config.batch_size
losses_eval_valid = []
niter = 1
start_epoch = 0
prev_time = time.clock()

# resume the training
if hasattr(config, 'resume_path'):
    print('Load metadata for resuming')
    with open(config.resume_path) as f:
        resume_metadata = pickle.load(f)

    # nn.layers.set_all_param_values(l_out, resume_metadata['param_values'])
    start_epoch = resume_metadata['epoch_since_start'] + 1
    niter = resume_metadata['iters_since_start']
    # learning_rate.set_value(resume_metadata['learning_rate'])
    print('setting learning rate to %.7f' % resume_metadata['learning_rate'])

for epoch in range(start_epoch, config.max_epoch):
    for train_batch_idxs in train_data_iterator:
        x_batch, mask_batch = create_batch(train_batch_idxs)
        train_loss = 0.0
        # train_loss = train(x_batch, mask_batch)
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
                x_batch, mask_batch = create_batch(valid_batch_idx)
                avg_valid_loss + 0.0
                # avg_valid_loss += validate(x_batch, x_batch)
            avg_valid_loss /= nvalid_batches
            losses_eval_valid.append(avg_valid_loss)
            print("    loss:\t%.6f" % avg_valid_loss)
            print

    # if epoch > config.learning_rate_decay_after:
        # new_learning_rate = np.float32(learning_rate.get_value() * config.learning_rate_decay)
        # learning_rate.set_value(new_learning_rate)
        # print('setting learning rate to %.7f' % new_learning_rate)

    if (epoch + 1) % config.save_every == 0:
        with open(metadata_target_path, 'w') as f:
            pickle.dump({
                'configuration': config_name,
                'experiment_id': experiment_id,
                'epoch_since_start': epoch,
                'iters_since_start': niter,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                # 'learning_rate': learning_rate.get_value(),
                'token2idx': token2idx,
                # 'param_values': nn.layers.get_all_param_values(l_out),
            }, f, pickle.HIGHEST_PROTOCOL)

        print("  saved to %s" % metadata_target_path)
