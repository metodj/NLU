from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import tensorflow as tf
from bert import tokenization

from bert_sct_utils_v2 import create_tokenizer_from_hub_module, SctProcessor, get_config
from bert_sct_utils_v2 import model_fn_builder, file_based_input_fn_builder, file_based_convert_examples_to_features

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files).")
flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("results_dir", "./results/", "The output directory for result files.")

# BERT
flags.DEFINE_string("bert_model_hub", None, "Pretrained BERT. (default)")
flags.DEFINE_string("bert_dir", None, "Local directory of pretrained BERT.")
flags.DEFINE_bool("bert_trainable", False, "Whether BERT weights are trainable.")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint of BERT.")
flags.DEFINE_string("init_checkpoint_sent", None, "Initial checkpoint of pre-trained sentiment model.")

# Training or evaluation
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

# Model selection
flags.DEFINE_bool("sentiment_only", False, "sentiment only")
flags.DEFINE_bool("commonsense_only", False, "commonsense only")
flags.DEFINE_bool("combination", False, "combination")

flags.DEFINE_integer("max_seq_length", 400, "Max. length after tokenization. If shorter padded, else truncated.")
flags.DEFINE_integer("batch_size", 4, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 1.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for.")


timestamp = datetime.datetime.now().strftime("%d_%H-%M")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not(FLAGS.bert_model_hub or (FLAGS.bert_dir and FLAGS.init_checkpoint)):
        raise ValueError("bert_model_hub or (bert_config_file and init_checkpoint)")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Create output directory
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Processor
    processor = SctProcessor()
    label_list = processor.get_labels()

    bert_config = None
    if not FLAGS.init_checkpoint:
        tokenizer = create_tokenizer_from_hub_module(FLAGS.bert_model_hub)

    else:
        tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(FLAGS.bert_dir, "vocab.txt"),
                                               do_lower_case=False)
        bert_config = get_config(os.path.join(FLAGS.bert_dir, "bert_config.json"))

    # ---------------------------------------------------------------------------------------------------------------- #
    # Model and Estimator
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir,
                                        save_summary_steps=10,
                                        save_checkpoints_steps=1000)

    model_fn = model_fn_builder(bert_model_hub=FLAGS.bert_model_hub,
                                bert_trainable=FLAGS.bert_trainable,
                                bert_config=bert_config,
                                init_checkpoint=FLAGS.init_checkpoint,
                                init_checkpoint_sent=FLAGS.init_checkpoint_sent,
                                num_labels=len(label_list),
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                sentiment_only=FLAGS.sentiment_only,
                                commonsense_only=FLAGS.commonsense_only,
                                combination=FLAGS.combination,
                                )

    params = {
        "batch_size": FLAGS.batch_size,
    }

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params=params)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Training
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.data_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples,
                                                label_list,
                                                FLAGS.max_seq_length,
                                                tokenizer,
                                                train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=False)

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Evaluation
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        eval_file = os.path.join(FLAGS.data_dir, "eval.tf_record")
        file_based_convert_examples_to_features(eval_examples,
                                                label_list,
                                                FLAGS.max_seq_length,
                                                tokenizer,
                                                eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))

        eval_steps = int(len(eval_examples) // FLAGS.batch_size)

        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                    seq_length=FLAGS.max_seq_length,
                                                    is_training=False,
                                                    drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.results_dir, timestamp + "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    # ---------------------------------------------------------------------------------------------------------------- #
    # Test
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.data_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples,
                                                label_list,
                                                FLAGS.max_seq_length,
                                                tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                                       seq_length=FLAGS.max_seq_length,
                                                       is_training=False,
                                                       drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.results_dir, timestamp + "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                predict_label = str(prediction["predict_labels"])

                probs = ["{0:.4f}".format(class_probability) for class_probability in probabilities]

                # output_line = "\t".join(probs) + "\t" + predict_label + "\n"
                output_line = predict_label + "\n"
                writer.write(output_line)
                num_written_lines += 1


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
