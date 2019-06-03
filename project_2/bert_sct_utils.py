import tensorflow as tf
import csv
import os
import collections
import numpy as np
from bert import tokenization, optimization, modeling
import tensorflow_hub as hub


# -------------------------------------------------------------------------------------------------------------------- #
# BERT Hub Module
def create_tokenizer_from_hub_module(bert_model_hub):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return tokenization.FullTokenizer(vocab_file=vocab_file,
                                      do_lower_case=do_lower_case)


def get_config(bert_config_file):
    config = modeling.BertConfig.from_json_file(bert_config_file)
    return config


# -------------------------------------------------------------------------------------------------------------------- #
# Data wrappers
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 vs_sent1=None, vs_sent2=None, vs_sent3=None, vs_sent4=None, vs_sent5=None, cs_dist=None):
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
        self.cs_dist = cs_dist


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentiment, cs_dist,is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sentiment = sentiment
        self.cs_dist = cs_dist
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
            label = int(tokenization.convert_to_unicode(line[1]))

            vs_sent1 = self._string_to_array(line[4][1:-1])
            vs_sent2 = self._string_to_array(line[5][1:-1])
            vs_sent3 = self._string_to_array(line[6][1:-1])
            vs_sent4 = self._string_to_array(line[7][1:-1])
            vs_sent5 = self._string_to_array(line[8][1:-1])

            cs_dist = self._string_to_array(line[9][1:-1])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, vs_sent1=vs_sent1,
                                         vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4, vs_sent5=vs_sent5,
                                         cs_dist=cs_dist))

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
            label = int(tokenization.convert_to_unicode(line[1]))

            vs_sent1 = self._string_to_array(line[4][1:-1])
            vs_sent2 = self._string_to_array(line[5][1:-1])
            vs_sent3 = self._string_to_array(line[6][1:-1])
            vs_sent4 = self._string_to_array(line[7][1:-1])
            vs_sent5 = self._string_to_array(line[8][1:-1])

            cs_dist = self._string_to_array(line[9][1:-1])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, vs_sent1=vs_sent1,
                                         vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4, vs_sent5=vs_sent5,
                                         cs_dist=cs_dist))
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1]
        # return ["negative", "positive"]

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
            label = int(tokenization.convert_to_unicode(line[1]))

            vs_sent1 = self._string_to_array(line[4][1:-1])
            vs_sent2 = self._string_to_array(line[5][1:-1])
            vs_sent3 = self._string_to_array(line[6][1:-1])
            vs_sent4 = self._string_to_array(line[7][1:-1])
            vs_sent5 = self._string_to_array(line[8][1:-1])

            cs_dist = self._string_to_array(line[9][1:-1])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, vs_sent1=vs_sent1,
                                         vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4, vs_sent5=vs_sent5,
                                         cs_dist=cs_dist))
        return examples

    @classmethod
    def _string_to_array(cls, string):
        return np.array(list(map(float, string.strip().split())), dtype=np.float)


# -------------------------------------------------------------------------------------------------------------------- #
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

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

    # Common Sense Distances
    cs_dist = example.cs_dist

    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("sentiment: %s, %s" % (str(sentiment.shape), str(sentiment)))
        tf.logging.info("cs_dist: %s, %s" % (str(cs_dist.shape), str(cs_dist)))

    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id,
                            sentiment=sentiment, cs_dist=cs_dist, is_real_example=True)

    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 100 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

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
        features["sentiment"] = create_tensor_feature(feature.sentiment)
        features["cs_dist"] = create_tensor_feature(feature.cs_dist)

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
        "sentiment": tf.FixedLenFeature([5, 4], tf.float32),
        "cs_dist": tf.FixedLenFeature([4], tf.float32),
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
def create_model(bert_model_hub, bert_trainable, bert_config, is_training, input_ids,
                 input_mask, segment_ids, labels, sentiment, cs_dist, num_labels):
    """Creates a classification model."""

    # ---------------------------------------------------------------------------------------------------------------- #
    # BERT
    if bert_model_hub:
        bert_module = hub.Module(bert_model_hub, trainable=bert_trainable)
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        output_layer = bert_outputs["pooled_output"]  # (batch_size, 768)
    else:
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()  # (batch_size, 768)

    hidden_size = output_layer.shape[-1].value  # 768
    batch_size = tf.shape(output_layer)[0]

    # ---------------------------------------------------------------------------------------------------------------- #
    # Sentiment
    sentiment_context = sentiment[:, 0:-1, :]  # (batch_size, 4, 4)
    sentiment_answer = sentiment[:, -1, :]  # (batch_size, 4)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Common Sense
    # cs_dist = cs_dist  # (batch_size, 4)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Weight initialization
    output_weights = tf.get_variable("rs_output_weights", [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("rs_output_bias", [num_labels], initializer=tf.zeros_initializer())

    # ---------------------------------------------------------------------------------------------------------------- #
    # Loss
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, rate=0.1)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        tf.logging.info("cs_dist, shape = %s" % cs_dist.shape)
        tf.logging.info("logits, shape = %s" % logits.shape)  # (batch_size, num_labels)
        tf.logging.info("probabilities, shape = %s" % probabilities.shape)  # (batch_size, num_labels)
        tf.logging.info("log_probs, shape = %s" % log_probs.shape)  # (batch_size, num_labels)
        tf.logging.info("one_hot_labels, shape = %s" % one_hot_labels.shape)  # (batch_size, num_labels)
        tf.logging.info("predicted_labels, shape = %s" % predicted_labels.shape)  # (batch_size, )
        tf.logging.info("per_example_loss, shape = %s" % per_example_loss.shape)  # (batch_size, )
        tf.logging.info("loss, shape = %s" % loss.shape)

        return loss, per_example_loss, logits, probabilities, predicted_labels


def model_fn_builder(bert_model_hub, bert_trainable, bert_config, init_checkpoint, num_labels, learning_rate,
                     num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for Estimator."""

        # ------------------------------------------------------------------------------------------------------------ #
        # Input
        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]  # (batch_size, max_seq_length)
        input_mask = features["input_mask"]  # (batch_size, max_seq_length)
        segment_ids = features["segment_ids"]  # (batch_size, max_seq_length)
        label_ids = features["label_ids"]  # (batch_size, )
        sentiment = features["sentiment"]  # (batch_size, 5, 4)
        cs_dist = features["cs_dist"]  # (batch_size, 4)

        # ------------------------------------------------------------------------------------------------------------ #
        # Model
        loss, per_example_loss, logits, probabilities, predicted_labels = create_model(bert_model_hub,
                                                                                       bert_trainable,
                                                                                       bert_config,
                                                                                       mode == tf.estimator.ModeKeys.TRAIN,
                                                                                       input_ids,
                                                                                       input_mask,
                                                                                       segment_ids,
                                                                                       label_ids,
                                                                                       sentiment,
                                                                                       cs_dist,
                                                                                       num_labels)

        # ------------------------------------------------------------------------------------------------------------ #
        # Initialize
        tvars = tf.trainable_variables()

        if bert_model_hub:
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

        else:
            initialized_variable_names = {}
            if init_checkpoint:
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        # ------------------------------------------------------------------------------------------------------------ #
        # Mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps,
                                                     num_warmup_steps, use_tpu=False)

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=loss,
                                                     train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(_per_example_loss, _label_ids, _predictions):
                accuracy = tf.metrics.accuracy(labels=_label_ids, predictions=_predictions)
                loss = tf.metrics.mean(values=_per_example_loss)
                f1_score = tf.contrib.metrics.f1_score(_label_ids, _predictions)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "f1": f1_score
                }

            eval_metrics = metric_fn(per_example_loss, label_ids, predicted_labels)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=loss,
                                                     eval_metric_ops=eval_metrics)

        else:
            predictions = {
                'probabilities': probabilities,
                "predict_labels": predicted_labels,
                "target_labels": label_ids
            }

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions=predictions)
        return output_spec

    return model_fn
