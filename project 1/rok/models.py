import os
import warnings
import pickle
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from load_embedding import load_embedding
import tf_utils


class ModelA:
    """
    EXPERIMENT A MODEL.
    Graph construction in __init__.
    """
    def __init__(self, experiment, mode, vocabulary_size, embedding_dim, state_dim, down_state_dim, sent_dim, cont_dim,
                 initializer, pad_idx, eos_idx, non_idx=-1, learning_rate=0.001, batch_size=64, max_grad_norm=5.0,
                 num_epochs=1):

        # Experiment (A, B, C) and mode (E - experiment, G - generator)
        self.experiment = experiment
        self.mode = mode

        # Model parameters
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.down_state_dim = down_state_dim
        self.sent_dim = sent_dim
        self.cont_dim = cont_dim
        self.pad_idx = pad_idx
        self.non_idx = non_idx
        self.eos_idx = eos_idx

        # Default train parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.initializer = initializer
        self.global_step = None

        # Input file
        self.sentences_file = tf.placeholder(tf.string)

        # Dataset
        dataset = tf.data.TextLineDataset(self.sentences_file).map(tf_utils.parse_ids_file).repeat(self.num_epochs)\
                    .batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        self.X_batch, self.y_batch = iterator.get_next()
        self.iterator_op = iterator.make_initializer(dataset)

        # LSTM Cell
        self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim)

        # Weights
        self.output_weight = tf.get_variable("output_weight", shape=[self.state_dim, self.vocabulary_size],
                                             initializer=self.initializer, trainable=True)  # 512x20000
        self.embedding_weight = tf.get_variable("embedding_weight", shape=[self.vocabulary_size, self.embedding_dim],
                                                initializer=self.initializer, trainable=True)  # 20000x100

        # Forward Pass
        self.loss, self.perplexity, self.perplexities = self.forward_pass(self.X_batch, self.y_batch)

        # Train
        self.optimize_op = self.train()

    def forward_pass(self, x, y):
        # Initialize LSTM cell
        batch_size = tf.shape(x)[0]  # Adjust for last batch
        state_c, state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)  # 64x512

        # EMBEDDING LAYER
        x_embedded = tf.nn.embedding_lookup(self.embedding_weight, x)  # 64x29x100

        # RNN PASS
        losses = []
        for t in range(self.sent_dim - 1):
            # Inputs for step t
            x_t = x_embedded[:, t, :]  # 64x100
            y_t = y[:, t]  # 64x1

            # LSTM pass
            lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t, state=(state_c, state_h))  # 64x512
            logits = tf.matmul(lstm_output, self.output_weight)  # 64x20000

            # Loss
            loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_t, logits=logits)  # 64x1
            losses.append(loss_t)

        # LOSS
        losses = tf.stack(losses)  # 29x64
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=0))  # 64x1 -> 1x1

        # PERPLEXITY
        sentence_mask = 1.0 - tf.cast(tf.equal(x, self.pad_idx), dtype=tf.float32)

        neg_log_prob = tf.multiply(tf.transpose(losses), sentence_mask)  # 64x29, without <pad>
        sent_length = tf.reduce_sum(sentence_mask, axis=1)  # 64x1, sentence length including <bos> and <eos>
        sum_neg_log_prob = tf.reduce_sum(neg_log_prob, axis=1)  # 64x1, sum of negative log probabilities

        perplexities = tf.exp(tf.divide(sum_neg_log_prob, sent_length))  # 64x1
        mean_perplexity = tf.reduce_mean(perplexities)  # 1x1

        return loss, mean_perplexity, perplexities

    def train(self):
        self.global_step = tf.Variable(1, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
        optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return optimize_op


class ModelB:
    """
    EXPERIMENT B MODEL.
    Graph construction in __init__.
    """
    def __init__(self, experiment, mode, vocabulary_size, embedding_dim, state_dim, down_state_dim, sent_dim, cont_dim,
                 initializer, pad_idx, eos_idx, non_idx=-1, learning_rate=0.001, batch_size=64, max_grad_norm=5.0,
                 num_epochs=1):

        # Experiment (A, B, C) and mode (E - experiment, G - generator)
        self.experiment = experiment
        self.mode = mode

        # Model parameters
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.down_state_dim = down_state_dim
        self.sent_dim = sent_dim
        self.cont_dim = cont_dim
        self.pad_idx = pad_idx
        self.non_idx = non_idx
        self.eos_idx = eos_idx

        # Default train parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.initializer = initializer
        self.global_step = None

        # Input file
        self.sentences_file = tf.placeholder(tf.string)
        self.embedding_file = tf.placeholder(tf.string)

        # Dataset
        dataset = tf.data.TextLineDataset(self.sentences_file).map(tf_utils.parse_ids_file).repeat(self.num_epochs)\
                    .batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        self.X_batch, self.y_batch = iterator.get_next()
        self.iterator_op = iterator.make_initializer(dataset)

        # LSTM Cell
        self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim)

        # Weights
        self.output_weight = tf.get_variable("output_weight", shape=[self.state_dim, self.vocabulary_size],
                                             initializer=self.initializer, trainable=True)  # 512x20000

        self.embedding_weight = tf.Variable(np.empty((VOCABULARY_SIZE, EMBEDDING_DIM), dtype=np.float32),
                                            collections=[], trainable=False)  # 20000x100

        # Forward Pass
        self.loss, self.perplexity, self.perplexities = self.forward_pass(self.X_batch, self.y_batch)

        # Train
        self.optimize_op = self.train()

    def forward_pass(self, x, y):
        # Initialize LSTM cell
        batch_size = tf.shape(x)[0]  # Adjust for last batch
        state_c, state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)  # 64x512

        # EMBEDDING LAYER
        x_embedded = tf.nn.embedding_lookup(self.embedding_weight, x)  # 64x29x100

        # RNN PASS
        losses = []
        for t in range(self.sent_dim - 1):
            # Inputs for step t
            x_t = x_embedded[:, t, :]  # 64x100
            y_t = y[:, t]  # 64x1

            # LSTM pass
            lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t, state=(state_c, state_h))  # 64x512
            logits = tf.matmul(lstm_output, self.output_weight)  # 64x20000

            # Loss
            loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_t, logits=logits)  # 64x1
            losses.append(loss_t)

        # LOSS
        losses = tf.stack(losses)  # 29x64
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=0))  # 64x1 -> 1x1

        # PERPLEXITY
        sentence_mask = 1.0 - tf.cast(tf.equal(x, self.pad_idx), dtype=tf.float32)

        neg_log_prob = tf.multiply(tf.transpose(losses), sentence_mask)  # 64x29, without <pad>
        sent_length = tf.reduce_sum(sentence_mask, axis=1)  # 64x1, sentence length including <bos> and <eos>
        sum_neg_log_prob = tf.reduce_sum(neg_log_prob, axis=1)  # 64x1, sum of negative log probabilities

        perplexities = tf.exp(tf.divide(sum_neg_log_prob, sent_length))  # 64x1
        mean_perplexity = tf.reduce_mean(perplexities)  # 1x1

        return loss, mean_perplexity, perplexities

    def train(self):
        self.global_step = tf.Variable(1, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
        optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return optimize_op


class ModelC:
    """
    EXPERIMENT C MODEL.
    Graph construction in __init__.
    """
    def __init__(self, experiment, mode, vocabulary_size, embedding_dim, state_dim, down_state_dim, sent_dim, cont_dim,
                 initializer, pad_idx, eos_idx, non_idx=-1, learning_rate=0.001, batch_size=64, max_grad_norm=5.0,
                 num_epochs=1):
        # Experiment (A, B, C) and mode (E - experiment, G - generator)
        self.experiment = experiment
        self.mode = mode

        # Model parameters
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.down_state_dim = down_state_dim
        self.sent_dim = sent_dim
        self.cont_dim = cont_dim
        self.pad_idx = pad_idx
        self.non_idx = non_idx
        self.eos_idx = eos_idx

        # Default train parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.initializer = initializer
        self.global_step = None

        # Input file
        self.sentences_file = tf.placeholder(tf.string)

        # Dataset
        dataset = tf.data.TextLineDataset(self.sentences_file).map(tf_utils.parse_ids_file).repeat(self.num_epochs)\
                    .batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        self.X_batch, self.y_batch = iterator.get_next()
        self.iterator_op = iterator.make_initializer(dataset)

        # LSTM Cell
        self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim)

        # Weights
        self.output_weight = tf.get_variable("output_weight", shape=[self.down_state_dim, self.vocabulary_size],
                                             initializer=self.initializer, trainable=True)  # 512x20000
        self.embedding_weight = tf.get_variable("embedding_weight", shape=[self.vocabulary_size, self.embedding_dim],
                                                initializer=self.initializer, trainable=True)  # 20000x100
        self.down_weight = tf.get_variable("down_weight", shape = [self.state_dim, self.down_state_dim],
                                           initializer=initializer, trainable=True)  # 1024x512

        # Forward Pass
        self.loss, self.perplexity, self.perplexities = self.forward_pass(self.X_batch, self.y_batch)

        # Train
        self.optimize_op = self.train()

    def forward_pass(self, x, y):
        # Initialize LSTM cell
        batch_size = tf.shape(x)[0]  # Adjust for last batch
        state_c, state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)  # 64x1024

        # EMBEDDING LAYER
        x_embedded = tf.nn.embedding_lookup(self.embedding_weight, x)  # 64x29x100

        # RNN PASS
        losses = []
        for t in range(self.sent_dim - 1):
            # Inputs for step t
            x_t = x_embedded[:, t, :]  # 64x100
            y_t = y[:, t]  # 64x1

            # LSTM pass
            lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t, state=(state_c, state_h))  # 64x1024
            down_logits = tf.matmul(lstm_output, self.down_weight)  # 64x512
            logits = tf.matmul(down_logits, self.output_weight)  # 64x20000

            # Loss
            loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_t, logits=logits)  # 64x1
            losses.append(loss_t)

        # LOSS
        losses = tf.stack(losses)  # 29x64
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=0))  # 64x1 -> 1x1

        # PERPLEXITY
        sentence_mask = 1.0 - tf.cast(tf.equal(x, self.pad_idx), dtype=tf.float32)

        neg_log_prob = tf.multiply(tf.transpose(losses), sentence_mask)  # 64x29, without <pad>
        sent_length = tf.reduce_sum(sentence_mask, axis=1)  # 64x1, sentence length including <bos> and <eos>
        sum_neg_log_prob = tf.reduce_sum(neg_log_prob, axis=1)  # 64x1, sum of negative log probabilities

        perplexities = tf.exp(tf.divide(sum_neg_log_prob, sent_length))  # 64x1
        mean_perplexity = tf.reduce_mean(perplexities)  # 1x1

        return loss, mean_perplexity, perplexities

    def train(self):
        self.global_step = tf.Variable(1, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
        optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return optimize_op


class Generator:
    """
    EXPERIMENT CONDITIONAL GENERATION MODEL.
    Graph construction in __init__.
    """
    def __init__(self, experiment, mode, vocabulary_size, embedding_dim, state_dim, down_state_dim, sent_dim, cont_dim,
                 initializer, pad_idx, eos_idx, non_idx=-1, learning_rate=0.001, batch_size=64, max_grad_norm=5.0,
                 num_epochs=1):
        # Experiment (A, B, C) and mode (E - experiment, G - generator)
        self.experiment = experiment
        self.mode = mode

        # Model parameters
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.down_state_dim = down_state_dim
        self.sent_dim = sent_dim
        self.cont_dim = cont_dim
        self.pad_idx = pad_idx
        self.non_idx = non_idx
        self.eos_idx = eos_idx

        # Default train parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.initializer = initializer
        self.global_step = None

        # Input file
        self.sentences_file = tf.placeholder(tf.string)

        # Dataset
        dataset = tf.data.TextLineDataset(self.sentences_file).map(tf_utils.parse_cont_ids_file)\
                    .repeat(self.num_epochs).batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        self.X_batch = iterator.get_next()
        self.iterator_op = iterator.make_initializer(dataset)

        # LSTM Cell
        self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim)

        # Weights
        self.output_weight = tf.get_variable("output_weight", shape=[self.down_state_dim, self.vocabulary_size],
                                             initializer=self.initializer, trainable=True)  # 512x20000
        self.embedding_weight = tf.get_variable("embedding_weight", shape=[self.vocabulary_size, self.embedding_dim],
                                                initializer=self.initializer, trainable=True)  # 20000x100
        self.down_weight = tf.get_variable("down_weight", shape=[self.state_dim, self.down_state_dim],
                                           initializer=initializer, trainable=True)  # 1024x512

        self.predictions = self.generator(self.X_batch)

    def generator(self, x):
        # Initialize LSTM cell
        batch_size = tf.shape(x)[0]  # Adjust for last batch
        state_c, state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)  # 64x1024

        # EMBEDDING LAYER
        # x_embedded = tf.nn.embedding_lookup(self.embedding_weight, x)  # 64x29x100

        sentence_mask = 1 - tf.cast(tf.equal(x, self.non_idx), dtype=tf.int32)  # 64x20
        sentence_length = tf.reduce_sum(sentence_mask, axis=1)  # 64x1

        x = tf.multiply(x, sentence_mask) + self.pad_idx * (1 - sentence_mask)

        # RNN PASS
        predictions = []
        for t in range(self.cont_dim + 1):
            x_t = x[:, t]

            if t > 0:
                cond = tf.less_equal(t + 1, sentence_length)
                x_t_embedded = tf.keras.backend.switch(cond,
                                              tf.nn.embedding_lookup(self.embedding_weight, x_t),
                                              tf.nn.embedding_lookup(self.embedding_weight, prediction))  # 64x100
            else:
                x_t_embedded = tf.nn.embedding_lookup(self.embedding_weight, x[:, t])

            # LSTM pass
            lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t_embedded, state=(state_c, state_h))  # 64x1024
            down_logits = tf.matmul(lstm_output, self.down_weight)  # 64x512
            logits = tf.matmul(down_logits, self.output_weight)  # 64x20000

            prediction = tf.argmax(logits, axis=1)  # 64x1

            cond = tf.less_equal(t + 1, sentence_length)
            output = tf.keras.backend.switch(cond, x_t, tf.cast(prediction, dtype=tf.int32))

            predictions.append(output)

        predictions = tf.transpose(tf.stack(predictions))  # 20x64 -> 64x20

        return predictions
