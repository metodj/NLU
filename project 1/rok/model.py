import os
import tensorflow as tf
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from load_embedding import load_embedding
import tf_utils


class Model:
    """
    EXPERIMENT A, B, C.
    CONDITIONAL GENERATION.
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

        # Mode
        if self.mode == "E":
            dataset = tf.data.TextLineDataset(self.sentences_file).map(tf_utils.parse_ids_file).repeat(self.num_epochs)\
                .batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            self.X_batch, self.y_batch = iterator.get_next()

        elif self.mode == "G":
            dataset = tf.data.TextLineDataset(self.sentences_file).map(tf_utils.parse_cont_ids_file)\
                .batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            self.X_batch = iterator.get_next()

        self.iterator_op = iterator.make_initializer(dataset)

        # Weights
        if self.experiment == "A":
            self.output_weight = tf.get_variable("output_weight", shape=[self.state_dim, self.vocabulary_size],
                                                 initializer=self.initializer, trainable=True)  # 512x20000
            self.embedding_weight = tf.get_variable("embedding_weight",
                                                    shape=[self.vocabulary_size, self.embedding_dim],
                                                    initializer=self.initializer, trainable=True)  # 20000x100
        elif self.experiment == "B":
            self.output_weight = tf.get_variable("output_weight", shape=[self.state_dim, self.vocabulary_size],
                                                 initializer=self.initializer, trainable=True)  # 512x20000

            self.embedding_weight = tf.Variable(np.empty((self.vocabulary_size, self.embedding_dim), dtype=np.float32),
                                                trainable=False)  # 20000x100
        elif self.experiment == "C":
            self.output_weight = tf.get_variable("output_weight", shape=[self.down_state_dim, self.vocabulary_size],
                                                 initializer=self.initializer, trainable=True)  # 512x20000
            self.embedding_weight = tf.get_variable("embedding_weight",
                                                    shape=[self.vocabulary_size, self.embedding_dim],
                                                    initializer=self.initializer, trainable=True)  # 20000x100
            self.down_weight = tf.get_variable("down_weight", shape=[self.state_dim, self.down_state_dim],
                                               initializer=self.initializer, trainable=True)  # 1024x512

        # LSTM Cell
        self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim)

        if self.mode == "E":
            # Forward pass
            self.loss, self.perplexity, self.perplexities = self.forward_pass(self.X_batch, self.y_batch)

            # Train
            self.optimize_op = self.train()
        elif self.mode == "G" and self.experiment in ["A", "B", "C"]:
            # Generator
            self.predictions = self.generator(self.X_batch)

    def forward_pass(self, x, y):
        if self.mode != "E":
            assert "Not in experiment mode - E."

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
            if self.experiment in ["A", "B"]:
                lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t, state=(state_c, state_h))  # 64x512
                logits = tf.matmul(lstm_output, self.output_weight)  # 64x20000
            elif self.experiment == "C":
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

    def generator(self, x):
        if self.mode != "G":
            assert "Not in generator mode - G."

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
            if self.experiment in ["A", "B"]:
                lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t_embedded, state=(state_c, state_h))  # 64x512
                logits = tf.matmul(lstm_output, self.output_weight)  # 64x20000
            elif self.experiment == "C":
                lstm_output, (state_c, state_h) = self.LSTM(inputs=x_t_embedded, state=(state_c, state_h))  # 64x1024
                down_logits = tf.matmul(lstm_output, self.down_weight)  # 64x512
                logits = tf.matmul(down_logits, self.output_weight)  # 64x20000

            prediction = tf.argmax(logits, axis=1)  # 64x1

            cond = tf.less_equal(t + 1, sentence_length)
            output = tf.keras.backend.switch(cond, x_t, tf.cast(prediction, dtype=tf.int32))

            predictions.append(output)

        predictions = tf.transpose(tf.stack(predictions))  # 20x64 -> 64x20

        return predictions


# # -------------------------------------------------------------------------------------------------------------------- #
# # DIRECTORIES
# DATA_DIR = "./data/"
# RESULTS_DIR = "./results/"
# MODEL_DIR = "./model/"
# WORD_EMBEDDINGS_FILE = "wordembeddings-dim100.word2vec"
# SENTENCES_TRAIN_FILE = "sentences.train"
# SENTENCES_TEST_FILE = "sentences_test.txt"
# SENTENCES_EVAL_FILE = "sentences.eval"
# SENTENCES_CONTINUATION_FILE = "sentences.continuation"
#
# # LANGUAGE MODEL PARAMETERS
# EMBEDDING_DIM = 100
# STATE_DIM = 512
# DOWN_STATE_DIM = 512
# VOCABULARY_SIZE = 20000
# SENT_DIM = 30
# CONT_DIM = 20
#
# # RNN PARAMETERS
# BATCH_SIZE = 64
# LEARNING_RATE = 0.001
# MAX_GRAD_NORM = 5.0
# NUM_EPOCHS = 1
#
# with open(RESULTS_DIR + "vocabulary.pkl", "rb") as f:
#     vocabulary, word_to_idx, idx_to_word = pickle.load(f)
#
# # -------------------------------------------------------------------------------------------------------------------- #
# # RUN
# tf.reset_default_graph()
# tf.set_random_seed(12345)
# np.random.seed(12345)
#
# # MODEL
# EXPERIMENT = "C"
# MODE = "E"
#
# if EXPERIMENT == "C":
#     STATE_DIM = 1024
#
# model = Model(experiment=EXPERIMENT,
#               mode=MODE,
#               vocabulary_size=VOCABULARY_SIZE,
#               embedding_dim=EMBEDDING_DIM,
#               state_dim=STATE_DIM,
#               down_state_dim=DOWN_STATE_DIM,
#               sent_dim=SENT_DIM,
#               cont_dim=CONT_DIM,
#               initializer=tf.contrib.layers.xavier_initializer(),
#               pad_idx=word_to_idx["<pad>"],
#               eos_idx=word_to_idx["<eos>"],
#               )
#
# saver = tf.train.Saver()
#
# with tf.Session() as session:
#     if MODE == "E":
#         session.run(tf.global_variables_initializer())
#
#         # LOAD EMBEDDING
#         if EXPERIMENT == "B":
#             load_embedding(session, word_to_idx, model.embedding_weight,
#                            DATA_DIR + WORD_EMBEDDINGS_FILE, EMBEDDING_DIM,
#                            VOCABULARY_SIZE)
#
#         # TRAINING
#         session.run(model.iterator_op,
#                     {model.sentences_file: RESULTS_DIR + "X_train.ids"})
#
#         batch_count = 0
#         total_batch = 500
#         while True:
#             try:
#                 batch_loss, batch_perplexity, _ = session.run([model.loss, model.perplexity, model.optimize_op])
#                 epoch = 1
#                 if batch_count % 100 == 0:
#                     print("epoch: {}/{:<6}batch: {:>5}/{:<10}loss = {:<13.2f}perp = {:<13.2f}".format(epoch, NUM_EPOCHS,
#                                                                                                       batch_count + 1,
#                                                                                                       total_batch,
#                                                                                                       batch_loss,
#                                                                                                       batch_perplexity))
#
#                 batch_count += 1
#                 if batch_count > total_batch:
#                     break
#             except tf.errors.OutOfRangeError:
#                 break
#
#         save_path = saver.save(session, MODEL_DIR + "/experiment" + EXPERIMENT +
#                                "/experiment" + EXPERIMENT + ".ckpt")
#         print("Model saved in path: %s" % save_path)
#
#         # EVALUATION
#         session.run(model.iterator_op, {model.sentences_file: RESULTS_DIR + "X_eval.ids"})
#         eval_perplexities = np.array([], dtype=np.float32)
#         batch_count = 0
#         while True:
#             try:
#                 batch_perplexities = session.run(model.perplexities)
#                 eval_perplexities = np.append(eval_perplexities, batch_perplexities)
#                 batch_count += 1
#             except tf.errors.OutOfRangeError:
#                 break
#         print("Evaluation finished.")
#
#         with open(RESULTS_DIR + "groupXX.perplexity" + EXPERIMENT, "w") as f:
#             for i in range(eval_perplexities.shape[0]):
#                 f.write("%0.3f" % eval_perplexities[i] + "\n")
#
#     elif MODE == "G":
#         saver.restore(session, MODEL_DIR + "/experiment" + EXPERIMENT +
#                       "/experiment" + EXPERIMENT + ".ckpt")
#         print("Model restored.")
#
#         session.run(model.iterator_op, {model.sentences_file: RESULTS_DIR + "X_cont.ids"})
#
#
#         continuation_ids = []
#         batch_count = 0
#         while True:
#             try:
#                 batch_predictions = session.run(model.predictions)
#                 continuation_ids.append(batch_predictions)
#                 batch_count = batch_count + 1
#
#                 print(batch_count, end="\r")
#             except tf.errors.OutOfRangeError:
#                 break
#
#         continuation_ids = np.concatenate(continuation_ids, axis=0)
#         print(continuation_ids.shape)
#
#         with open(RESULTS_DIR + "groupXX.continuation", "w") as f:
#             for i in range(continuation_ids.shape[0]):
#                 try:
#                     eos_pos = continuation_ids[i, 1:].tolist().index(int(word_to_idx["<eos>"]))
#                 except:
#                     eos_pos = 20
#
#                 gen_sent = " ".join([idx_to_word[token_id] if idx < eos_pos else "" for idx, token_id in
#                                      enumerate(continuation_ids[i, 1:].tolist())])
#                 f.write(gen_sent + "\n")

