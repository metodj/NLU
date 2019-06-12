import tensorflow as tf
import utils_1 as utils_1


class Model:

    def __init__(self,initializer,stories_file, validation_file, test_file, embedding_dim=3, state_dim = 64, stories_dim = 5,  learning_rate=0.001, batch_size=8, max_grad_norm=5.0,
                 num_epochs=1, num_epochs_sc=1, sent_perc_train=0.80, sc_perc_train=0.80):


        #Model parameters

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.stories_dim = stories_dim


        #Train parameters

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_epochs_sc = num_epochs_sc
        self.initializer = initializer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)


        # Input files
        self.stories_file = stories_file
        self.validation_file = validation_file
        self.test_file = test_file

        # Train percentages
        self.sent_perc_train=sent_perc_train
        self.sc_perc_train=sc_perc_train


        with tf.name_scope("dataset_initialization_training"):

            dataset = tf.data.Dataset.from_tensor_slices(utils_1.read_embed_createtensor_from_file_stories(self.stories_file)).map(utils_1.embedding_io_map)\
                .batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            self.x_batch, self.y_batch = iterator.get_next()

            self.iterator_op = iterator.make_initializer(dataset)

        with tf.name_scope('dataset_initialization_validation'):

            valid_dataset = tf.data.Dataset.from_tensor_slices(utils_1.read_embed_createtensor_from_stories_val(self.validation_file)).map(utils_1.embedding_io_map_val)\
                .batch(self.batch_size)
            iterator_val = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)

            self.s_batch, self.e1_e_batch, self.e2_e_batch, self.ans_batch = iterator_val.get_next()

            self.iterator_val_op = iterator_val.make_initializer(valid_dataset)

        with tf.name_scope('dataset_initialization_test'):

            test_dataset = tf.data.Dataset.from_tensor_slices(utils_1.read_embed_createtensor_from_stories_val(self.test_file)).map(utils_1.embedding_io_map_val)\
                .batch(self.batch_size)
            iterator_test = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)

            self.s_batch_test, self.e1_e_batch_test, self.e2_e_batch_test, self.ans_batch_test = iterator_test.get_next()

            self.iterator_test_op = iterator_test.make_initializer(test_dataset)

        # Weights
        with tf.name_scope("weights_initialization_s"):

            self.output_weights_s = tf.get_variable("output_weights_s",shape=[self.state_dim,self.embedding_dim], initializer=self.initializer, trainable=True)
            self.output_bias_s = tf.get_variable("output_bias_s",shape=self.embedding_dim, initializer=self.initializer, trainable=True)

        with tf.name_scope('similarity_matrix_s'):
            self.similarity_matrix = tf.get_variable("similarity_matrix",shape=[self.embedding_dim, self.embedding_dim],initializer=self.initializer, trainable=True)

        with tf.name_scope('lstm_s'):

            #to adjust dimension for last batch
            batch_size = tf.shape(self.x_batch)[0]

            self.LSTM_s = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim, name="lstm_cell_s")

            state_c,state_h = self.LSTM_s.zero_state(batch_size=batch_size, dtype=tf.float32)

            lstm_output, (state_c,state_h) = tf.nn.dynamic_rnn(cell=self.LSTM_s, inputs=self.x_batch, dtype=tf.float32)

            probabilities_sent = tf.nn.softmax(tf.matmul(state_h,self.output_weights_s, name="output_multiplication")+self.output_bias_s)

        with tf.name_scope('cosine_sim_and_loss'):

            self.cosine_sim = utils_1.cosine_similarity(probabilities_sent, self.y_batch)

            self.loss = -(tf.reduce_mean(self.cosine_sim))

        with tf.name_scope('fine_tuning_story_cloze'):

            lstm_output_sc, (state_c_sc, state_h_sc) = tf.nn.dynamic_rnn(cell=self.LSTM_s, inputs=self.s_batch, dtype=tf.float32)

            self.e_p_batch = tf.nn.softmax(tf.matmul(state_h_sc, self.output_weights_s) + self.output_bias_s)

            logits_1 = tf.linalg.diag_part(tf.matmul(tf.matmul(a=self.e_p_batch, b=self.similarity_matrix),self.e1_e_batch, transpose_b=True))

            logits_2 = tf.linalg.diag_part(tf.matmul(tf.matmul(a=self.e_p_batch, b=self.similarity_matrix),self.e2_e_batch, transpose_b=True))

            self.logits_sc = tf.stack([logits_1,logits_2], axis = 1)

            self.probs_sc = tf.nn.softmax(self.logits_sc, axis=-1)

        with tf.name_scope('sc_loss'):

            one_hot_labels_sent = tf.one_hot(self.ans_batch, depth=2, dtype=tf.float32)

            predicted_labels_sent = tf.argmax(self.probs_sc, axis=-1, output_type=tf.int32)

            per_example_loss = -tf.reduce_sum(one_hot_labels_sent * self.probs_sc, axis=-1)

            self.loss_sc = tf.reduce_mean(per_example_loss)

        with tf.name_scope('accuracy_sc'):
            self.accuracy_sc = tf.metrics.accuracy(labels=self.ans_batch,predictions=predicted_labels_sent)

        with tf.name_scope('test_sc'):
            lstm_output_sc_test, (state_c_sc_test, state_h_sc_test) = tf.nn.dynamic_rnn(cell=self.LSTM_s, inputs=self.s_batch_test,
                                                                         dtype=tf.float32)

            e_p_batch_test = tf.nn.softmax(tf.matmul(state_h_sc_test, self.output_weights_s) + self.output_bias_s)

            logits_1_test = tf.linalg.diag_part(
                tf.matmul(tf.matmul(a=e_p_batch_test, b=self.similarity_matrix), self.e1_e_batch_test, transpose_b=True))

            logits_2_test = tf.linalg.diag_part(
                tf.matmul(tf.matmul(a=e_p_batch_test, b=self.similarity_matrix), self.e2_e_batch_test, transpose_b=True))

            logits_sc_test = tf.stack([logits_1_test, logits_2_test], axis=1)

            probs_sc_test = tf.nn.softmax(logits_sc_test, axis=-1)

            predicted_labels_sent_test = tf.argmax(probs_sc_test, axis=-1, output_type=tf.int32)

        with tf.name_scope('accuracy_sc'):

            self.accuracy_sc_test = tf.metrics.accuracy(labels=self.ans_batch_test,predictions=predicted_labels_sent_test)

        with tf.name_scope('accuracy_intializer'):

            # Isolate the variables stored behind the scenes by the metric operation
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_sc")

            # Define initializer to initialize/reset running variables
            self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        with tf.name_scope('optimization'):

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
            self.optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.optimize_sc_op = optimizer.minimize(self.loss_sc, global_step=self.global_step)

        with tf.name_scope('saver'):

            self.saver = tf.train.Saver()

