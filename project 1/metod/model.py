import tensorflow as tf


class Model:

    def __init__(self,
                 batch_size,
                 vocab_size,
                 embed_size,
                 hidden_units,
                 num_epochs,
                 experiment):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_units = hidden_units
        self.num_epochs = num_epochs

        self.initializer = tf.contrib.layers.xavier_initializer()
        self.max_gradient_norm = 5
        self.global_step = tf.Variable(0, trainable=False)

        self.experiment = experiment

        self.pad_index = 2
        self.eos_index = 1

        def parse(line):
            line_split = tf.string_split([line])
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
            return input_seq, output_seq

        with tf.name_scope('inputs'):
            self.file_name_train = tf.placeholder(tf.string)
            self.file_name_validation = tf.placeholder(tf.string)
            self.file_name_test = tf.placeholder(tf.string)

            training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).batch(
                self.batch_size)
            validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).batch(self.batch_size)
            test_dataset = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)

            iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
            self.input_batch, self.output_batch = iterator.get_next()

            self.training_init_op = iterator.make_initializer(training_dataset)
            self.validation_init_op = iterator.make_initializer(validation_dataset)
            self.test_init_op = iterator.make_initializer(test_dataset)

            self.nr_words = tf.reduce_prod(tf.shape(self.input_batch))

        
        self.LSTM = tf.nn.rnn_cell.LSTMCell(self.hidden_units, initializer=self.initializer, trainable=True)

        with tf.name_scope("embeddings"):

            if self.experiment == "A" or self.experiment == "C":
                self.input_embedding_mat = tf.get_variable('input_embedding_mat', shape=(vocab_size, embed_size),
                                                           dtype=tf.float32, initializer=self.initializer,
                                                           trainable=True)
            else:
                self.input_embedding_mat = tf.get_variable('input_embedding_mat', shape=(vocab_size, embed_size),
                                                           dtype=tf.float32, initializer=self.initializer,
                                                           trainable=False)

            if self.experiment == "C":
                self.projection_mat = tf.get_variable("projection_mat", shape=(int(hidden_units / 2), hidden_units),
                                                      dtype=tf.float32, initializer=self.initializer, trainable=True)

                self.output_embedding_mat = tf.get_variable('output_embedding_mat',
                                                            shape=(vocab_size, int(hidden_units / 2)),
                                                            dtype=tf.float32, initializer=self.initializer,
                                                            trainable=True)
            else:
                self.output_embedding_mat = tf.get_variable('output_embedding_mat', shape=(vocab_size, hidden_units),
                                                            dtype=tf.float32, initializer=self.initializer,
                                                            trainable=True)

            def output_embedding(current_output):
                return tf.matmul(current_output, tf.transpose(self.output_embedding_mat))

            def projection(current_output):
                return tf.matmul(current_output, tf.transpose(self.projection_mat))

        # TODO: mask logits and output_batch before passing them to loss (to diminish the influence of <pad> tags on training)
        def forward_pass(self, input_batch, output_batch, perplexity='train'):
            input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, input_batch)  # (64,29,100)

            batch_size = tf.shape(input_batch)[0]
            with tf.name_scope('rnn'):
                state_c, state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)
                preds = []
                for i in range(29):
                    output, (state_c, state_h) = self.LSTM(
                        tf.reshape(input_embedded[:, i, :], [batch_size, self.embed_size]), state=(state_c, state_h))
                    preds.append(output)

                preds = tf.stack(preds,
                                 axis=1)  # concatenate preds over axis=1 (2nd dimension), to obtain tensor of size (batch_size,29,512)
                if self.experiment == "C":
                    preds = tf.map_fn(projection, preds)  # (64,29,512)

                logits = tf.map_fn(output_embedding, preds)  # (64,29,20000)
                logits = tf.reshape(logits, [-1, self.vocab_size])
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(output_batch, [-1]),
                                                                      logits=logits)

                tf.summary.scalar("loss",tf.reduce_sum(loss))
                # no need to calculate perplexity during training...
                if perplexity == 'train':
                    return loss, 0

                else:
                    mask = 1.0 - tf.cast(tf.equal(output_batch, self.pad_index), dtype=tf.float32)
                    negative_loglikelihood = tf.reduce_sum(tf.multiply(loss, tf.reshape(mask, [-1])))
                    nr_words = tf.reduce_sum(mask)
                    perplexity = tf.math.exp(negative_loglikelihood / nr_words)
                    return loss, perplexity

        def generate_sentence(self, input_batch):

            input_batch = tf.reshape(input_batch, [-1])
            sentence_length = tf.cast(tf.where(tf.equal(input_batch, self.eos_index))[0, 0], dtype=tf.int32)
            input_batch = tf.reshape(input_batch[:sentence_length], [1, -1])
            input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, input_batch)

            with tf.name_scope('rnn'):

                state_c, state_h = self.LSTM.zero_state(batch_size=1, dtype=tf.float32)
                preds, last_predicted = [], tf.convert_to_tensor(
                    [[0]])  # to avoid problems with tf.cond in the for loop below

                for i in range(20):

                    input_ = tf.cond(tf.less(i, sentence_length),
                                     true_fn=lambda: tf.reshape(input_embedded[:, i, :], [1, self.embed_size]),
                                     false_fn=lambda: tf.reshape(tf.nn.embedding_lookup(self.input_embedding_mat,
                                                                                        last_predicted),
                                                                 [1, self.embed_size]))

                    output, (state_c, state_h) = self.LSTM(input_, state=(state_c, state_h))
                    if self.experiment == "C":
                        output = projection(output)
                    output = output_embedding(output)

                    word_id = tf.argmax(output, axis=1)[0]
                    last_predicted = tf.reshape(tf.convert_to_tensor(word_id), [1, 1])

                    word = tf.cond(tf.less(i, sentence_length), true_fn=lambda: tf.reshape(input_batch, [-1])[i], \
                                   false_fn=lambda: tf.cast(word_id, dtype=tf.int32))
                    preds.append(word)

                    # fuck you tensorflow :)
                #                     if word == self.eos_index:
                #                         break

                preds = tf.convert_to_tensor(preds)
                eos_index = tf.cast(tf.equal(preds, self.eos_index), dtype=tf.int32)
                eos_index = tf.cond(tf.less_equal(tf.reduce_sum(eos_index), 0), true_fn=lambda: tf.constant(20), \
                                    false_fn=lambda: tf.cast(tf.where(tf.equal(preds, self.eos_index))[0, 0],
                                                             dtype=tf.int32))

                preds = preds[:eos_index]

            return preds

        with tf.name_scope('loss'):
            self.loss, self.perplexity = forward_pass(self, self.input_batch, self.output_batch)
            self.loss_, self.perplexity_ = forward_pass(self, self.input_batch, self.output_batch, perplexity='test')

        with tf.name_scope('optimization'):
            params = tf.trainable_variables()
            opt = tf.train.AdamOptimizer()
            # gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        #         with tf.name_scope("optimization"):
        #             optimizer = tf.train.AdamOptimizer()
        #             optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_gradient_norm)
        #             self.updates  = optimizer.minimize(self.loss, global_step=self.global_step)

        with tf.name_scope('sentence_generation'):
            self.preds = generate_sentence(self, self.input_batch)