import tensorflow as tf
import math
from model_1 import Model


#import numpy as np
#import os
#import pickle
#import warnings
#import pandas as pd
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#import csv
#import utils as utils


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
              num_epochs=3,
              num_epochs_sc=1,
              #sc_train_loss='AC'
              sent_perc_train=1,
              sc_perc_train=0.80,
              )

# variable_names = [v.name for v in tf.trainable_variables()]

#SESSION

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    session.run(model.running_vars_initializer)

    final_accuracy = 0.0


    with open('experiments_new/experiment_n_5.txt', 'w') as result_file:

        result_file.write('\n\nDETAILS: ' + '\n\n' + \
                          'Batch size : ' + str(model.batch_size) + '\n' + \
                          'Num epochs sentiment model training : ' + str(model.num_epochs) + '\n' + \
                          'Num epochs story cloze fine tuning training : ' + str(model.num_epochs_sc) + '\n' +\
                          'Percentage training sentiment model: '+str(model.sent_perc_train) +'\n' +\
                          'Percentage training story cloze: '+str(model.sc_perc_train) +'\n')

        train_sent_batches = math.floor(11021 * model.sent_perc_train)

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


        for j in range(model.num_epochs_sc):

            # TRAINING STORY CLOZE TASK

            result_file.write('\n\n'+'STARTING TRAINING PART STORY CLOZE EPOCH '+ str(j+1) +'\n\n')

            session.run(model.iterator_val_op)

            count_sc_batches = 0

            train_sc_batches = math.floor(234 * model.sc_perc_train)

            while True:

                try:

                    if count_sc_batches == train_sc_batches:
                        break

                    b_loss_sc, _ , g_step = session.run([model.loss_sc, model.optimize_sc_op, model.global_step])

                    count_sc_batches += 1

                    if count_sc_batches % 30:

                        result_file.write('Processed batches: ' + str(count_sc_batches) + '\n' +\
                                          'Batch loss: ' + str(b_loss_sc) + '\n' +\
                                          'Global step: ' + str(g_step) + '\n')

                except tf.errors.OutOfRangeError:
                    break

            # TESTING STORY CLOZE TASK

            result_file.write('\n\n' + 'STARTING TESTING PART STORY CLOZE EPOCH ' + str(j+1) +'\n\n')

            while True:

                try:

                    acc, g_step = session.run([model.accuracy_sc, model.global_step])

                    count_sc_batches += 1

                    if count_sc_batches % 30:
                        result_file.write('Processed batches :' + str(count_sc_batches-train_sc_batches) + '\n' +\
                                          'Accuracy: ' + str(acc) +'\n'
                                          'Global step (must remain constant): ' + str(g_step) + '\n'\
                                         )
                    final_accuracy = acc

                except tf.errors.OutOfRangeError:
                    break

            result_file.write('\n\n' + 'ACCURACY TEST STORY CLOZE TASK EPOCH ' + str(j + 1) + '\n'+str(final_accuracy)+'\n\n')




        # save_path = model.saver.save(session, "sentiment_pp/sentiment_2/sentiment_2.ckpt")
        # result_file.write("Model saved in path " + str(save_path))

    '''
    variable_names_values = session.run(variable_names)

    for v in variable_names_values:
        print(v)
    '''

'''

#   THIS PART IS TO HAVE THE TXT FILE WITH THE PREDICTED AND THE VADER SENTIMENT EMBEDDINGS


    with open('experiments_final/sentiment_txt_files/split/sentiment_tr3ep+50%val.txt', 'w') as p_file:
        
        session.run(model.iterator_val_op)
        
        
        while True:

            try:

                ep, ee1, ee2 = session.run([model.e_p_batch, model.e1_e_batch, model. e2_e_batch])

                for i in range(model.batch_size):

                    try:

                        p_file.write(str(ep[i]) +' '+ str(ee1[i]) +'\n')
                        p_file.write(str(ep[i]) +' '+ str(ee2[i]) +'\n')

                        #p_file.write(str(ep[i]) + ' ' + str(ee1[i])+' '+ str(ee2[i]) + '\n')


                    except IndexError:
                        break

            except tf.errors.OutOfRangeError:
                break




'''