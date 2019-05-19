import os
import tensorflow as tf
import numpy as np
import pickle
import warnings
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

import utils as utils
from model import Model

#GRAPH

tf.reset_default_graph()


model = Model(initializer=tf.contrib.layers.xavier_initializer(),
              stories_file='data/train_stories.csv',
              embedding_dim=3,
              state_dim=64,
              stories_dim=5,
              learning_rate=0.001,
              batch_size=8,
              max_grad_norm=5.0,
              num_epochs=2)


#SESSION

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    session.run(model.iterator_op)

    count_batches = 0


    with open('experiments/experiment4.txt', 'w') as result_file:


        while True:

            try:

                c_s_shape, b_loss, _, g_step = session.run([ model.cosine_sim_shape, model.loss, model.optimize_op, model.global_step])

                count_batches+=1

                result_file.write('Processed batches: '+ str(count_batches)+'\n'+
                                  'Cosine_sim_shape: '+ str(c_s_shape)+'\n'+
                                  'Batch loss: '+str(b_loss)+'\n'+
                                  'Global step: '+ str(g_step)+'\n')
                if g_step % 200 == 0:
                    print(b_loss)

            except tf.errors.OutOfRangeError:
                break
