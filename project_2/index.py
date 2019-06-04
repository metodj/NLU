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


# train stories are 88161
# total train batches are 11021
# validation stories are 1871
# total validation batches are 234



#GRAPH

tf.reset_default_graph()


model = Model(initializer=tf.contrib.layers.xavier_initializer(),
              stories_file='data/train_stories.csv',
              validation_file='data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv',
              embedding_dim=3,
              state_dim=64,
              stories_dim=5,
              learning_rate=0.001,
              batch_size=8,
              max_grad_norm=5.0,
              num_epochs=1)


#SESSION

with tf.Session() as session:

    session.run(tf.global_variables_initializer())


    with open('experiments/exp_final_1.txt', 'w') as result_file:

        train_sent_batches = 8817

        for i in range(model.num_epochs):

            session.run(model.iterator_op)

            count_batches = 0


            result_file.write('\n\n'+'START TRAINING SENTIMENT PART EPOCH '+str(i+1)+'\n\n')

            # TRAINING THE SENTIMENT MODEL ON PART OF TRAINING DATASET

            while True:

                try:

                    b_loss, _, g_step = session.run([ model.loss, model.optimize_op, model.global_step])

                    count_batches += 1

                    if count_batches % 300 == 0:

                        result_file.write('Processed batches: '+ str(count_batches)+'\n'+
                                          'Batch loss: '+str(b_loss)+'\n'+
                                          'Global step: '+ str(g_step)+'\n')

                    if count_batches == train_sent_batches:
                        break

                except tf.errors.OutOfRangeError:
                    break


            # TESTING THE SENTIMENT MODEL ON PART OF TRAINING DATASET

            result_file.write('\n\n'+'START TESTING SENTIMENT PART EPOCH ' + str(i+1) + '\n\n')

            while True:

                try:
                    b_loss = session.run([model.loss])

                    count_batches += 1

                    if count_batches % 300 == 0:

                        result_file.write('Processed batches: ' + str(count_batches-train_sent_batches) + '\n' +
                                          'Batch loss: ' + str(b_loss) + '\n' +
                                          'Global step: ' + str(g_step) + '\n')

                except tf.errors.OutOfRangeError:
                    break
        
        result_file.write('\n\n'+'STARTING TRAINING STORY CLOZE'+'\n\n')

        session.run(model.iterator_val_op)

        count_sc_batches = 0

        train_sc_batches = 187

        while True:

            try:

                b_loss_sc, _ , g_step = session.run([model.loss_sc, model.optimize_sc_op, model.global_step])

                count_sc_batches += 1

                if count_sc_batches % 10:

                    result_file.write('Processed batches: ' + str(count_sc_batches) + '\n' +\
                                      'Batch loss: ' + str(b_loss_sc) + '\n' +\
                                      'Global step: ' + str(g_step) + '\n')

                if count_sc_batches == train_sc_batches:
                    break

            except tf.errors.OutOfRangeError:
                break
        
        result_file.write('\n\n' + 'STARTING TESTING STORY CLOZE' + '\n\n')

        '''
        while True:

            try:

                acc = session.run([model.accuracy])

                count_sc_batches += 1

                result_file.write(str(acc)+'\n')

            except tf.errors.OutOfRangeError:
                break
        '''





#   THIS PART IS TO HAVE THE TXT FILE WITH THE PREDICTED AND THE VADER SENTIMENT EMBEDDINGS


    with open('experiments/s_files/s_file_final_1.txt', 'w') as p_file:
        
        session.run(model.iterator_val_op)
        
        
        while True:

            try:

                ep, ee1, ee2 = session.run([model.e_p_batch, model.e1_e_batch, model. e2_e_batch])

                for i in range(model.batch_size):

                    try:

                        p_file.write(str(ep[i]) +' '+ str(ee1[i]) +'\n')
                        p_file.write(str(ep[i]) +' '+ str(ee2[i]) +'\n')


                    except IndexError:
                        break

            except tf.errors.OutOfRangeError:
                break









