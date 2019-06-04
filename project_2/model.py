import os
import tensorflow as tf
import numpy as np
import pickle
import warnings
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

import utils as utils


class Model:

    def __init__(self,initializer,stories_file, validation_file, embedding_dim=3, state_dim = 64, stories_dim = 5,  learning_rate=0.001, batch_size=8, max_grad_norm=5.0,
                 num_epochs=1):


        #Model parameters

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.stories_dim = stories_dim


        #Train parameters

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.initializer = initializer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)



        # Input files
        self.stories_file = stories_file
        self.validation_file = validation_file


        with tf.name_scope("dataset_initialization_training"):

            dataset = tf.data.Dataset.from_tensor_slices(utils.read_embed_createtensor_from_file_stories(self.stories_file)).map(utils.embedding_io_map)\
                .batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            #print(dataset.output_types, dataset.output_shapes)

            self.x_batch, self.y_batch = iterator.get_next()

            self.iterator_op = iterator.make_initializer(dataset)


        with tf.name_scope('dataset_initialization_validation'):

            valid_dataset = tf.data.Dataset.from_tensor_slices(utils.read_embed_createtensor_from_stories_val(self.validation_file)).map(utils.embedding_io_map_val)\
                .batch(self.batch_size)
            iterator_val = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)

            #print(valid_dataset.output_types, valid_dataset.output_shapes)

            self.s_batch, self.e1_e_batch, self.e2_e_batch, self.ans_batch = iterator_val.get_next()

            self.iterator_val_op = iterator_val.make_initializer(valid_dataset)

        # Weights
        with tf.name_scope("weights_initialization"):

            self.output_weights = tf.get_variable("output_weights",shape=[self.state_dim,self.embedding_dim], initializer=self.initializer, trainable=True)
            self.output_bias = tf.get_variable("output_bias",shape=self.embedding_dim, initializer=self.initializer, trainable=True)

            self.similarity_matrix = tf.get_variable("similarity_matrix",shape=[self.embedding_dim, self.embedding_dim],initializer=self.initializer, trainable=True)


        with tf.name_scope('lstm'):

            #to adjust dimension for last batch

            batch_size = tf.shape(self.x_batch)[0]


            # LSTM Cell  : is it correct the number of units?
            self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim, name="lstm_cell")

            state_c,state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)

            lstm_output, (state_c,state_h) = tf.nn.dynamic_rnn(cell=self.LSTM, inputs=self.x_batch, dtype=tf.float32)

            logits_sent = tf.nn.softmax(tf.matmul(state_h,self.output_weights, name="output_multiplication")+self.output_bias)



        with tf.name_scope('cosine_sim_and_loss'):

            self.cosine_sim = utils.cosine_similarity(logits_sent, self.y_batch)

            self.loss = -(tf.reduce_mean(self.cosine_sim))


        with tf.name_scope('fine_tuning_story_cloze'):

            lstm_output_sc, (state_c_sc, state_h_sc) = tf.nn.dynamic_rnn(cell=self.LSTM, inputs=self.s_batch, dtype=tf.float32)

            self.e_p_batch = tf.nn.softmax(tf.matmul(state_h_sc, self.output_weights) + self.output_bias)

            logits_1 = tf.linalg.diag_part(tf.matmul(tf.matmul(a=self.e_p_batch, b=self.similarity_matrix),self.e1_e_batch, transpose_b=True))

            logits_2 = tf.linalg.diag_part(tf.matmul(tf.matmul(a=self.e_p_batch, b=self.similarity_matrix),self.e2_e_batch, transpose_b=True))

            self.logits_sc = tf.stack([logits_1,logits_2], axis = 1)

            '''
            print(logits_1.shape, logits_2.shape, self.logits_sc.shape)
            print(self.ans_batch.shape)
            print(tf.one_hot(indices=self.ans_batch, depth=2))
            '''
            # self.one_hot_ans_batch = tf.one_hot(indices=self.ans_batch, depth=2)

            loss_sc_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ans_batch, logits=self.logits_sc, name = 'sparse_cross_entropy_softmax_sc')

            self.loss_sc = tf.reduce_mean(loss_sc_batch)

        '''

        with tf.name_scope('accuracy'):

            self.predictions = tf.argmax(self.logits_sc, axis=1)

            #print(self.predictions.shape, self.ans_batch.shape)

            self.accuracy = tf.metrics.accuracy(labels=self.ans_batch, predictions=self.predictions)

        '''


        with tf.name_scope('optimization'):

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
            self.optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)

            self.optimize_sc_op = optimizer.minimize(self.loss_sc, global_step=self.global_step)



'''

    def train_sentiment_model(self):
        with tf.name_scope("optimization_sentiment_model"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
            optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return optimize_op

    def train_story_cloze(self):
        with tf.name_scope("optimization_story_cloze"):
            optimizer_sc = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer_sc = tf.contrib.estimator.clip_gradients_by_norm(optimizer_sc, clip_norm=self.max_grad_norm)
            optimize_sc_op = optimizer_sc.minimize(self.loss_sc, global_step=self.global_step)
        return optimize_sc_op


'''





