import tensorflow as tf
import csv
import os
import collections
import numpy as np
from bert import modeling, tokenization, optimization


# -------------------------------------------------------------------------------------------------------------------- #
# Data wrappers
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 vs_sent1=None, vs_sent2=None, vs_sent3=None, vs_sent4=None, vs_sent5=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.vs_sent1 = vs_sent1
        self.vs_sent2 = vs_sent2
        self.vs_sent3 = vs_sent3
        self.vs_sent4 = vs_sent4
        self.vs_sent5 = vs_sent5


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentiment, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sentiment = sentiment
        self.is_real_example = is_real_example


# -------------------------------------------------------------------------------------------------------------------- #
# Data loaders
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _string_to_array(cls, string):
        raise NotImplementedError()


class SctProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "sct.train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%s" % (line[0])
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])

            vs_sent1 = self._string_to_array(line[4][1:-1])
            vs_sent2 = self._string_to_array(line[5][1:-1])
            vs_sent3 = self._string_to_array(line[6][1:-1])
            vs_sent4 = self._string_to_array(line[7][1:-1])
            vs_sent5 = self._string_to_array(line[8][1:-1])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, vs_sent1=vs_sent1,
                                         vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4, vs_sent5=vs_sent5))

        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "sct.validation.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%s" % (line[0])
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])

            vs_sent1 = self._string_to_array(line[4][1:-1])
            vs_sent2 = self._string_to_array(line[5][1:-1])
            vs_sent3 = self._string_to_array(line[6][1:-1])
            vs_sent4 = self._string_to_array(line[7][1:-1])
            vs_sent5 = self._string_to_array(line[8][1:-1])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, vs_sent1=vs_sent1,
                                         vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4, vs_sent5=vs_sent5))
        return examples

    def get_labels(self):
        """See base class."""
        return ["negative", "positive"]

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "sct.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "test-%s" % (line[0])
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])

            vs_sent1 = self._string_to_array(line[4][1:-1])
            vs_sent2 = self._string_to_array(line[5][1:-1])
            vs_sent3 = self._string_to_array(line[6][1:-1])
            vs_sent4 = self._string_to_array(line[7][1:-1])
            vs_sent5 = self._string_to_array(line[8][1:-1])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, vs_sent1=vs_sent1,
                                         vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4, vs_sent5=vs_sent5))
        return examples

    @classmethod
    def _string_to_array(cls, string):
        return np.array(list(map(float, string.strip().split())), dtype=np.float)


# -------------------------------------------------------------------------------------------------------------------- #
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(input_ids=[0] * max_seq_length,
                             input_mask=[0] * max_seq_length,
                             segment_ids=[0] * max_seq_length,
                             label_id=0,
                             sentiment=np.zeros(shape=(5, 4), dtype=np.float32),
                             is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    # Sentiment
    sentiment = np.zeros(shape=(5, 4))
    sentiment[0, :] = example.vs_sent1
    sentiment[1, :] = example.vs_sent2
    sentiment[2, :] = example.vs_sent3
    sentiment[3, :] = example.vs_sent4
    sentiment[4, :] = example.vs_sent5

    sentiment = np.reshape(sentiment, newshape=(-1,))

    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("sentiment: %s, %s" % (str(sentiment.shape), str(sentiment)))

    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id,
                            sentiment=sentiment, is_real_example=True)

    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_tensor_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        features["sentiment_array"] = create_tensor_feature(feature.sentiment)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        "sentiment_array": tf.FixedLenFeature([5, 4], tf.float32),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# -------------------------------------------------------------------------------------------------------------------- #
# Model loaders
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, sentiment_array, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""

    # Narrative
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    batch_size = 4
    # batch_size = tf.shape(output_layer)[0]

    # Sentiment
    # rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64, name="sentiment_rnn_cell",
    #                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    sentiment_weights = tf.get_variable(
        "sentiment_weights", [4, 4],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    sentiment_bias = tf.get_variable(
        "sentiment_bias", [4], initializer=tf.zeros_initializer())

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)


        sentiment_context = sentiment_array[:, 0, :]
        sentiment_answer = sentiment_array[:, -1, :]

        # initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        # _, (_, sentiment_hidden) = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=sentiment_context,
        #                                              initial_state=initial_state, dtype=tf.float32)

        logits = tf.matmul(sentiment_context, sentiment_weights)
        logits = tf.matmul(logits, sentiment_answer)
        logits = tf.nn.bias_add(logits, sentiment_bias)
        sentiment_probability = tf.nn.softmax(logits)

        print("sentiment_probabiltiy", sentiment_probability.get_shape())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        sentiment_array = features["sentiment_array"]

        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, sentiment_array,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        tvars = [tvar for tvar in tvars if tvar.name not in ["loss/rnn/sentiment_rnn_cell/bias:0",
                                                             "loss/rnn/sentiment_rnn_cell/kernel:0",
                                                             "sentiment_weights:0",
                                                             "sentiment_bias:0"
                                                             ]]

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


