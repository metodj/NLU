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

    def __init__(self,initializer,stories_file, embedding_dim=3, state_dim = 64, stories_dim = 5,  learning_rate=0.001, batch_size=8, max_grad_norm=5.0,
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



        # Input file
        self.stories_file = stories_file



        with tf.name_scope("dataset_initialization"):

            dataset = tf.data.Dataset.from_tensor_slices(utils.read_embed_createtensor_from_file_stories(self.stories_file)).map(utils.embedding_io_map)\
                .repeat(self.num_epochs)\
                .batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            self.x_batch, self.y_batch = iterator.get_next()

            self.iterator_op = iterator.make_initializer(dataset)


        # Weights
        with tf.name_scope("weights_initialization"):

            self.output_weights = tf.get_variable("output_weights",shape=[self.state_dim, self.embedding_dim], initializer=self.initializer, trainable=True)
            self.output_bias = tf.get_variable("output_bias",shape=self.embedding_dim, initializer=self.initializer, trainable=True)


        with tf.name_scope('lstm'):

            #to adjust dimension for last batch

            batch_size = tf.shape(self.x_batch)[0]


            # LSTM Cell  : is it correct the number of units?
            self.LSTM = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_dim, name="lstm_cell")

            state_c,state_h = self.LSTM.zero_state(batch_size=batch_size, dtype=tf.float32)

            lstm_output, (state_c,state_h) = tf.nn.dynamic_rnn(cell=self.LSTM, inputs=self.x_batch, dtype=tf.float32)

            logits = tf.nn.softmax(tf.matmul(state_h, self.output_weights, name="output_multiplication")+self.output_bias)



        with tf.name_scope('cosine_sim_and_loss'):

            self.cosine_sim, self.cosine_sim_shape  = utils.cosine_similarity(logits, self.y_batch)

            self.loss = -(tf.reduce_mean(self.cosine_sim))




        with tf.name_scope('train'):

            self.optimize_op = self.train()




    def train(self):
        with tf.name_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self.max_grad_norm)
            optimize_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return optimize_op







