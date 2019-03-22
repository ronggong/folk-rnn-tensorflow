# folk-rnn-tensorflow

Tensorflow implementation of folk-rnn (https://github.com/IraKorshunova/folk-rnn). The original folk-rnn is implemented using Theano and Lasagne.

Written in python3.

Run `python train_rnn.py config5 data/allabcworepeats_parsed (or other training datas)` for model training.

Checkpoints are stored in `metadata` folder. Relaunch training will automatically restore the checkpoints.

Support `CudnnLSTM` for CUDA GPU training, and normal LSTM for CPU inference as well.

The model results are not compared with the original implementation since I don't have a GPU. Comparison test is welcome!

Rong Gong