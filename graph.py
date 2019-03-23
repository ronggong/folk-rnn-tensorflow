import tensorflow as tf


# model graph
def build_graph(x, mask, seq_length, prob_dropout, config, vocab_size, is_training=True):
    
    W_emb = tf.Variable(tf.eye(vocab_size, dtype=tf.float32), name='w_emb')
    l_emb = tf.nn.embedding_lookup(W_emb, x)

    # cudnn
    if config.cudnn_lstm:
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            3,
            config.rnn_size,
            direction='unidirectional',
            dropout=0,
            bias_initializer=tf.zeros_initializer()
        )

        rnn_output, _ = lstm(l_emb, initial_state=None, training=is_training)
    else:
        # cudnn compatible lstm for training or inference
        with tf.variable_scope('cudnn_lstm'):
            cells = [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(config.rnn_size) for _ in range(3)]
            
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            initial_state = cell.zero_state(config.batch_size, tf.float32)

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
    #print("total parameters", np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
    # output layer type, num_param, output_shape
    print('Embedding shape', l_emb.get_shape())
    print('RNN output shape', rnn_output.get_shape())
    print('Reshape shape', l_reshp.get_shape())
    print('Dense shape', l_out.get_shape())

    if is_training:
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
    else:
        train_op = None
        loss = None
        var_lr = None

    return l_out, train_op, loss, var_lr