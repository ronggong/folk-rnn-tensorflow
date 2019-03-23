# folk-rnn-tensorflow

Tensorflow implementation of folk-rnn (https://github.com/IraKorshunova/folk-rnn). The original folk-rnn is implemented using Theano and Lasagne.

Written in python==3.6.8 and tensorflow==1.12

Run `python train_rnn.py config5 data/allabcworepeats_parsed (or other training datas)` for model training.

Checkpoints and metadata such as `token2idx` dict are stored in `metadata` folder. Relaunch training will automatically restore the checkpoints. 

Run `python sample_rnn.py metadata --seed "<your seed note sequence>"` for model inference. The first two token of the seed are the measure and the key. A seed example `M:3/4 K:Cmaj`. It will load the latest checkpoints in `metadata` folder.

Support `CudnnLSTM` for CUDA GPU training, and normal LSTM for CPU inference as well.

The model results are not compared with the original implementation since I don't have a GPU. Comparison test is welcome!

Rong Gong, 2019
MIT License