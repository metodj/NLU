import os
import platform
import sys

import tensorflow as tf
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)

from model import Model
from load_embedding import load_embedding
import utils
import tf_utils

logger = utils.Logger("./logs/")
timer = utils.Timer()

# !pip install tensorboardcolab
# from tensorboardcolab import *
# tbc = TensorBoardColab()

logger.append("SYSTEM", platform.system())
logger.append("MACHINE", platform.machine())
logger.append("PLATFORM", platform.platform())
logger.append("UNAME", platform.uname(), "\n")

logger.append("PYTHON", sys.version.split('\n'))
logger.append("TF VERSION", tf.__version__, "\n")

# -------------------------------------------------------------------------------------------------------------------- #
# MODEL
tf.app.flags.DEFINE_string("EXPERIMENT", "C", "model selection (A, B, C)")
tf.app.flags.DEFINE_string("MODE", "G", "mode (Experiment - E, Generation - G)")
tf.app.flags.DEFINE_boolean("RESTORE", True, "Restore existing model")

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# DIRECTORIES AND FILES
tf.app.flags.DEFINE_string("DATA_DIR", "./data/", "data directory")
tf.app.flags.DEFINE_string("RESULTS_DIR", "./results/", "results directory")
tf.app.flags.DEFINE_string("MODEL_DIR", "./model/", "saved model directory")
tf.app.flags.DEFINE_string("WORD_EMBEDDINGS_FILE", "wordembeddings-dim100.word2vec", "word embedding file")
tf.app.flags.DEFINE_string("SENTENCES_TRAIN_FILE", "sentences.train", "train file")
tf.app.flags.DEFINE_string("SENTENCES_TEST_FILE", "sentences_test.txt", "test file")
tf.app.flags.DEFINE_string("SENTENCES_EVAL_FILE", "sentences.eval", "evaluation file")
tf.app.flags.DEFINE_string("SENTENCES_CONTINUATION_FILE", "sentences.continuation", "continuation file")

# LANGUAGE MODEL PARAMETERS
tf.app.flags.DEFINE_integer("EMBEDDING_DIM", 100, "word embedding dimension")
tf.app.flags.DEFINE_integer("DOWN_STATE_DIM", 512, "down projection dimension")
tf.app.flags.DEFINE_integer("STATE_DIM", 512, "rnn cell hidden state dimension")
tf.app.flags.DEFINE_integer("VOCABULARY_SIZE", 20000, "vocabulary size")
tf.app.flags.DEFINE_integer("SENT_DIM", 30, "train sentence length")
tf.app.flags.DEFINE_integer("CONT_DIM", 20, "continuation max. sentence length")

# RNN PARAMETERS
tf.app.flags.DEFINE_integer("BATCH_SIZE", 64, "batch size")
tf.app.flags.DEFINE_integer("NUM_EPOCHS", 1, "number of epochs for training")
tf.app.flags.DEFINE_float("LEARNING_RATE", 0.001, "learning rate for rnn")
tf.app.flags.DEFINE_float("MAX_GRAD_NORM", 5.0, "max. norm for gradient clipping")
tf.app.flags.DEFINE_string('f', '', 'tensorflow bug')

FLAGS = tf.app.flags.FLAGS
if FLAGS.EXPERIMENT == "C":
    FLAGS.STATE_DIM = 1024
tf_utils.print_flags(FLAGS, logger)

# -------------------------------------------------------------------------------------------------------------------- #
# PREPROCESSING
logger.append("PREPROCESSING STARTING.")
vocabulary, word_to_idx, idx_to_word = utils.create_vocabulary(FLAGS.DATA_DIR + FLAGS.SENTENCES_TRAIN_FILE,
                                                               FLAGS.VOCABULARY_SIZE)
X_train = utils.create_dataset(FLAGS.DATA_DIR + FLAGS.SENTENCES_TRAIN_FILE, word_to_idx)
logger.append("X_train CREATED.")
X_test = utils.create_dataset(FLAGS.DATA_DIR + FLAGS.SENTENCES_TEST_FILE, word_to_idx)
logger.append("X_test CREATED.")
X_eval = utils.create_dataset(FLAGS.DATA_DIR + FLAGS.SENTENCES_EVAL_FILE, word_to_idx)
logger.append("X_eval CREATED.")
X_cont = utils.load_continuation(FLAGS.DATA_DIR + FLAGS.SENTENCES_CONTINUATION_FILE, word_to_idx)
logger.append("X_cont CREATED.")

with open(FLAGS.RESULTS_DIR + "vocabulary.pkl", "wb") as f:
    pickle.dump((vocabulary, word_to_idx, idx_to_word), f)

with open(FLAGS.RESULTS_DIR + "X_train.ids", "w") as f:
    for i in range(X_train.shape[0]):
        f.write(" ".join([str(x) for x in X_train[i, :]]) + "\n")

with open(FLAGS.RESULTS_DIR + "X_test.ids", "w") as f:
    for i in range(X_test.shape[0]):
        f.write(" ".join([str(x) for x in X_test[i, :]]) + "\n")

with open(FLAGS.RESULTS_DIR + "X_eval.ids", "w") as f:
    for i in range(X_eval.shape[0]):
        f.write(" ".join([str(x) for x in X_eval[i, :]]) + "\n")

with open(FLAGS.RESULTS_DIR + "X_cont.ids", "w") as f:
    for i in range(X_cont.shape[0]):
        f.write(" ".join([str(x) for x in X_cont[i, :]]) + "\n")

num_train = X_train.shape[0]
num_test = X_test.shape[0]
num_eval = X_eval.shape[0]
num_cont = X_cont.shape[0]

logger.append("vocabulary:", len(vocabulary))
logger.append("X_train:", X_train.shape)
logger.append("X_test:", X_test.shape)
logger.append("X_eval:", X_eval.shape)
logger.append("<bos> idx", word_to_idx["<bos>"])
logger.append("<eos> idx", word_to_idx["<eos>"])
logger.append("<pad> idx", word_to_idx["<pad>"])
logger.append("<unk> idx", word_to_idx["<unk>"])
logger.append("PREPROCESSING FINISHED.\n")

# -------------------------------------------------------------------------------------------------------------------- #
# LOAD VOCABULARY
with open(FLAGS.RESULTS_DIR + "vocabulary.pkl", "rb") as f:
    vocabulary, word_to_idx, idx_to_word = pickle.load(f)

logger.append("VOCABULARY LOADED.\n")
# -------------------------------------------------------------------------------------------------------------------- #
# RUN
tf.reset_default_graph()
tf.set_random_seed(12345)
np.random.seed(12345)

model = Model(experiment=FLAGS.EXPERIMENT,
              mode=FLAGS.MODE,
              vocabulary_size=FLAGS.VOCABULARY_SIZE,
              embedding_dim=FLAGS.EMBEDDING_DIM,
              state_dim=FLAGS.STATE_DIM,
              down_state_dim=FLAGS.DOWN_STATE_DIM,
              sent_dim=FLAGS.SENT_DIM,
              cont_dim=FLAGS.CONT_DIM,
              initializer=tf.contrib.layers.xavier_initializer(),
              pad_idx=word_to_idx["<pad>"],
              eos_idx=word_to_idx["<eos>"],
              num_epochs=FLAGS.NUM_EPOCHS
              )
logger.append("TRAINABLE VARIABLES.")
tf_utils.trainable_parameters(logger)

saver = tf.train.Saver()
timer.__enter__()

logger.append("TF SESSION STARTING.\n")
with tf.Session() as session:
    #     writer = tbc.get_deep_writers("./")
    #     writer.add_graph(session.graph)

    if FLAGS.MODE == "E":
        logger.append("EXPERIMENT STARTING.")
        with tf.name_scope("experiment"):
            if not FLAGS.RESTORE:
                session.run(tf.global_variables_initializer())

                # LOAD EMBEDDING
                if FLAGS.EXPERIMENT == "B":
                    load_embedding(session, word_to_idx, model.embedding_weight,
                                   FLAGS.DATA_DIR + FLAGS.WORD_EMBEDDINGS_FILE,
                                   FLAGS.EMBEDDING_DIM, FLAGS.VOCABULARY_SIZE)
            else:
                saver.restore(session, FLAGS.MODEL_DIR + "/experiment" +
                              FLAGS.EXPERIMENT + "/experiment" +
                              FLAGS.EXPERIMENT + ".ckpt")
                logger.append("MDOEL RESTORED.")

            # TRAINING
            #             summary_op = tf.summary.merge_all()

            session.run(model.iterator_op,
                        {model.sentences_file: FLAGS.RESULTS_DIR + "X_train.ids"})

            logger.append("TRAINING STARTING.")
            batch_count = 0
            batch_perplexity = 100
            total_batch = num_train // FLAGS.BATCH_SIZE + 1
            while True:
                try:
                    #                     batch_loss, batch_perplexity, _, global_step, summary = session.run([model.loss, model.perplexity,
                    #                                                                    model.optimize_op, model.global_step, summary_op])

                    batch_loss, _, global_step = session.run([model.loss,
                                                              model.optimize_op, model.global_step])

                    #                     writer.add_summary(summary, global_step)
                    epoch = 1
                    if batch_count % 100 == 0:
                        logger.append("batch: {:>5}/{:<6}".format(batch_count + 1, total_batch),
                                      "loss = {:<8.2f}".format(batch_loss), "perp = {:<8.2f}".format(batch_perplexity))

                    batch_count += 1
                except tf.errors.OutOfRangeError:
                    break

            logger.append("TRAINING FINISHED.")
            #             writer.flush()
            save_path = saver.save(session, FLAGS.MODEL_DIR + "/experiment" +
                                   FLAGS.EXPERIMENT + "/experiment" +
                                   FLAGS.EXPERIMENT + ".ckpt")
            logger.append("MODEL SAVED", save_path)

            # EVALUATION
            logger.append("EVALUATION STARTING.")
            session.run(model.iterator_op, {model.sentences_file: FLAGS.RESULTS_DIR + "X_test.ids"})
            eval_perplexities = np.array([], dtype=np.float32)
            batch_count = 0
            while True:
                try:
                    batch_perplexities = session.run(model.perplexities)
                    eval_perplexities = np.append(eval_perplexities, batch_perplexities)
                    batch_count += 1
                except tf.errors.OutOfRangeError:
                    break
            logger.append("EVALUATION FINISHED.")

            with open(FLAGS.RESULTS_DIR + "group23.perplexity" + FLAGS.EXPERIMENT, "w") as f:
                for i in range(eval_perplexities.shape[0]):
                    f.write("%0.3f" % eval_perplexities[i] + "\n")

            logger.append("EXPERIMENT FINISHED.\n")
    elif FLAGS.MODE == "G":
        logger.append("GENERATION STARTING.")
        with tf.name_scope("generation"):
            saver.restore(session, FLAGS.MODEL_DIR + "/experiment" +
                          FLAGS.EXPERIMENT + "/experiment" +
                          FLAGS.EXPERIMENT + ".ckpt")
            logger.append("MODEL RESTORED.")

            session.run(model.iterator_op, {model.sentences_file: FLAGS.RESULTS_DIR + "X_cont.ids"})

            continuation_ids = []
            batch_count = 0
            while True:
                try:
                    batch_predictions = session.run(model.predictions)
                    continuation_ids.append(batch_predictions)
                    batch_count = batch_count + 1

                    print(batch_count, end="\r")
                except tf.errors.OutOfRangeError:
                    break

            continuation_ids = np.concatenate(continuation_ids, axis=0)
            print(continuation_ids.shape)

            with open(FLAGS.RESULTS_DIR + "group23.continuation", "w") as f:
                for i in range(continuation_ids.shape[0]):
                    try:
                        eos_pos = continuation_ids[i, 1:].tolist().index(int(word_to_idx["<eos>"]))
                    except:
                        eos_pos = 20

                    gen_sent = " ".join([idx_to_word[token_id] if idx <= eos_pos else "" for idx, token_id in
                                         enumerate(continuation_ids[i, 1:].tolist())])
                    f.write(gen_sent + "\n")
        logger.append("GENERATION FINISHED.\n")
    logger.append("SESSION FINISHING.\n")
timer.__exit__()
tf_utils.delete_flags(FLAGS)
logger.create_log()
