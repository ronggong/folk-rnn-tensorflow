from __future__ import print_function

import os
import sys
import time
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle
import numpy as np
import tensorflow as tf
import argparse
from graph import build_graph

parser = argparse.ArgumentParser()
parser.add_argument('metadata_path')
parser.add_argument('--rng_seed', type=int, default=42)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--ntunes', type=int, default=1)
parser.add_argument('--seed')
parser.add_argument('--terminal', action="store_true")

args = parser.parse_args()

metadata_path = args.metadata_path
rng_seed = args.rng_seed
temperature = args.temperature
ntunes = args.ntunes
seed = args.seed

with open(os.path.join(metadata_path, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f, encoding='latin1')

config = importlib.import_module('configurations.%s' % metadata['configuration'])

# samples dir
if not os.path.isdir('samples'):
        os.makedirs('samples')
target_path = "samples/%s-s%d-%.2f-%s.txt" % (
    metadata['experiment_id'], rng_seed, temperature, time.strftime("%Y%m%d-%H%M%S", time.localtime()))

token2idx = metadata['token2idx']
idx2token = dict((v, k) for k, v in token2idx.items())
vocab_size = len(token2idx)

x = tf.placeholder(tf.int32, [1, None])

config.dropout = 0
config.batch_size = 1
predictions, _, _, _ = build_graph(x, None, None, None, config, vocab_size, is_training=False)

saver = tf.train.Saver()

conf = tf.ConfigProto()
conf.gpu_options.allow_growth=True

with tf.Session(config=conf) as sess:
    # restore the checkpoint
    if not os.path.exists(metadata_path):
        print('[!] Checkpoints path does not exist...')
        quit()

    ckpt = tf.train.get_checkpoint_state(metadata_path)

    # read the check point if it is there
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        experiment_id = ckpt_name
        saver.restore(sess, os.path.join(metadata_path, ckpt_name))
        print('[*] Read checkpoints {}'.format(ckpt_name))

    start_idx, end_idx = token2idx['<s>'], token2idx['</s>']

    rng = np.random.RandomState(rng_seed)
    vocab_idxs = np.arange(vocab_size)

    # Converting the seed passed as an argument into a list of idx
    seed_sequence = [start_idx]
    if seed is not None:
        for token in seed.split(' '):
            seed_sequence.append(token2idx[token])

    for i in range(ntunes):
        sequence = seed_sequence[:]
        while sequence[-1] != end_idx:
            # output probablities
            proba = sess.run(predictions, feed_dict={x: np.array(np.expand_dims(sequence, axis=0), dtype='int32')})
            # sample the probablities of the last timestamp
            next_itoken = rng.choice(vocab_idxs, p=proba[-1, :]/temperature)
            sequence.append(next_itoken)

        abc_tune = [idx2token[j] for j in sequence[1:-1]]
        if not args.terminal:
            f = open(target_path, 'a+')
            f.write('X:' + repr(i) + '\n')
            f.write(abc_tune[0] + '\n')
            f.write(abc_tune[1] + '\n')
            f.write(' '.join(abc_tune[2:]) + '\n\n')
            f.close()
        else:
            print('X:' + repr(i))
            print(abc_tune[0])
            print(abc_tune[1])
            print(' '.join(abc_tune[2:]) + '\n')

    if not args.terminal:
        print('Saved to '+target_path)
