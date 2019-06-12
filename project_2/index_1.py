import os
import tensorflow as tf
import math
from model_1 import Model


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .csv files.")
flags.DEFINE_string("sentiment_results_dir", None, "The output directory where the model txt result files will be written.")
flags.DEFINE_string("results_file", None, "The name of the results file.")


def main(_):

    #GRAPH

    tf.reset_default_graph()

    # Create results directory
    tf.gfile.MakeDirs(FLAGS.sentiment_results_dir)

    model = Model(initializer=tf.contrib.layers.xavier_initializer(),
                  stories_file=os.path.join(FLAGS.data_dir, 'train_stories.csv'),
                  validation_file=os.path.join(FLAGS.data_dir,'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'),
                  test_file=os.path.join(FLAGS.data_dir,'test_for_report-stories_labels.csv'),
                  embedding_dim=3,
                  state_dim=64,
                  stories_dim=5,
                  learning_rate=0.001,
                  batch_size=8,
                  max_grad_norm=5.0,
                  num_epochs=3,
                  num_epochs_sc=1,
                  sent_perc_train=1,
                  sc_perc_train=1,
                  )

    #SESSION

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        session.run(model.running_vars_initializer)

        final_accuracy = 0.0

        path_results_file = os.path.join(FLAGS.sentiment_results_dir,FLAGS.results_file + '.txt')

        with open(path_results_file, 'w') as result_file:       #HERE PUT PATH FOR THE TXT FILE WITH THE RESULTS

            result_file.write('\n\nDETAILS: ' + '\n\n' + \
                              'Batch size : ' + str(model.batch_size) + '\n' + \
                              'Num epochs sentiment model training : ' + str(model.num_epochs) + '\n' + \
                              'Num epochs story cloze fine tuning training : ' + str(model.num_epochs_sc) + '\n' +\
                              'Percentage of dataset used for training sentiment model: '+str(model.sent_perc_train)+'/1'+'\n' +\
                              'Percentage of dataset used for training parameters for story cloze: '+str(model.sc_perc_train)+'/1'+'\n')

            train_sent_batches = math.floor(11021 * model.sent_perc_train)

            for i in range(model.num_epochs):

                session.run(model.iterator_op)

                count_batches = 0

                result_file.write('\n\n'+'TRAINING SENTIMENT MODEL EPOCH '+str(i+1)+'\n\n')

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
                if 1-model.sent_perc_train > 0 :
                    result_file.write('\n\n'+'VALIDATING SENTIMENT MODEL EPOCH ' + str(i+1) + 'ON' +str(1-model.sent_perc_train)+ 'OF THE DATASET' + '\n\n')

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

                result_file.write('\n\n'+'FINE TUNING PARAMETERS FOR STORY CLOZE TASK EPOCH '+ str(j+1) + '\n\n')

                session.run(model.iterator_val_op)

                count_sc_batches = 0

                train_sc_batches = math.floor(234 * model.sc_perc_train)

                while True:

                    try:

                        if count_sc_batches == train_sc_batches:
                            break

                        b_loss_sc, _ , g_step = session.run([model.loss_sc, model.optimize_sc_op, model.global_step])

                        count_sc_batches += 1

                        if count_sc_batches % 30 == 0:

                            result_file.write('Processed batches: ' + str(count_sc_batches) + '\n' +\
                                              'Batch loss: ' + str(b_loss_sc) + '\n' +\
                                              'Global step: ' + str(g_step) + '\n')

                    except tf.errors.OutOfRangeError:
                        break

                # TESTING STORY CLOZE TASK ON PART OF VAL SET

                if 1-model.sc_perc_train > 0:
                    result_file.write('\n\n' + 'VALIDATING FINE TUNING FOR STORY CLOZE TASK EPOCH ' + str(j+1) + 'ON' +str(1-model.sc_perc_train)+ 'OF THE DATASET' +'\n\n')

                while True:

                    try:

                        acc, g_step = session.run([model.accuracy_sc, model.global_step])

                        count_sc_batches += 1

                        if count_sc_batches % 30 == 0:
                            result_file.write('Processed batches :' + str(count_sc_batches-train_sc_batches) + '\n' +\
                                              'Accuracy: ' + str(acc[1]) +'\n' \
                                              # + 'Global step (must remain constant): ' + str(g_step) + '\n'\
                                             )
                        final_accuracy = acc

                    except tf.errors.OutOfRangeError:
                        break



            # TESTING STORY CLOZE TASK ON TEST SET

            result_file.write('\n\n' + 'TESTING FOR STORY CLOZE TASK ' + '\n\n')

            session.run(model.iterator_test_op)

            count_sc_batches_test = 0

            final_accuracy_test = 0

            while True:

                try:

                    acc_test, g_step = session.run([model.accuracy_sc_test, model.global_step])

                    count_sc_batches_test += 1

                    if count_sc_batches_test % 50 == 0:
                        result_file.write('Processed batches :' + str(count_sc_batches_test) + '\n' +\
                                          'Accuracy: ' + str(acc_test[1]) + '\n' \
                                          # + 'Global step (must remain constant): ' + str(g_step) + '\n' \
                                          )
                    final_accuracy_test = acc_test[1]

                except tf.errors.OutOfRangeError:
                    break

            result_file.write('\n\n' + 'ACCURACY STORY CLOZE TASK ON TEST SET' + '\n' + str(final_accuracy_test) + '\n\n')

            # save_path = model.saver.save(session, "sentiment_pp/sentiment_2/sentiment_2.ckpt")
            # result_file.write("Model saved in path " + str(save_path))

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("sentiment_results_dir")
    flags.mark_flag_as_required("results_file")
    tf.app.run()
