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

    def __init__(self, guid, text_a, text_b_pos=None, text_b_neg=None, label=None,
                 vs_sent1=None, vs_sent2=None, vs_sent3=None, vs_sent4=None, vs_sent5_pos=None, vs_sent5_neg=None,
                 cs_dist_pos=None, cs_dist_neg=None):
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
        self.text_b_pos = text_b_pos
        self.text_b_neg = text_b_neg

        self.label = label
        self.vs_sent1 = vs_sent1
        self.vs_sent2 = vs_sent2
        self.vs_sent3 = vs_sent3
        self.vs_sent4 = vs_sent4

        self.vs_sent5_pos = vs_sent5_pos
        self.vs_sent5_neg = vs_sent5_neg

        self.cs_dist_pos = cs_dist_pos
        self.cs_dist_neg = cs_dist_neg

        self.text_b = None
        self.vs_sent5 = None
        self.cs_dist = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentiment, cs_dist, is_real_example=True):
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
        lines = self._read_tsv(os.path.join(data_dir, "sct_v2.train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%s" % (line[0])
            label = int(tokenization.convert_to_unicode(line[1]))

            text_a = tokenization.convert_to_unicode(line[2])

            text_b_pos = tokenization.convert_to_unicode(line[3])
            text_b_neg = tokenization.convert_to_unicode(line[4])

            vs_sent1 = self._string_to_array(line[5][1:-1])
            vs_sent2 = self._string_to_array(line[6][1:-1])
            vs_sent3 = self._string_to_array(line[7][1:-1])
            vs_sent4 = self._string_to_array(line[8][1:-1])

            vs_sent5_pos = self._string_to_array(line[9][1:-1])
            vs_sent5_neg = self._string_to_array(line[10][1:-1])

            cs_dist_pos = self._string_to_array(line[11][1:-1])
            cs_dist_neg = self._string_to_array(line[12][1:-1])

            examples.append(InputExample(guid=guid, label=label, text_a=text_a,
                                         text_b_pos=text_b_pos, text_b_neg=text_b_neg,
                                         vs_sent1=vs_sent1, vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4,
                                         vs_sent5_pos=vs_sent5_pos, vs_sent5_neg=vs_sent5_neg,
                                         cs_dist_pos=cs_dist_pos, cs_dist_neg=cs_dist_neg))

        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "sct_v2.validation.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "eval-%s" % (line[0])
            label = int(tokenization.convert_to_unicode(line[1]))

            text_a = tokenization.convert_to_unicode(line[2])

            text_b_pos = tokenization.convert_to_unicode(line[3])
            text_b_neg = tokenization.convert_to_unicode(line[4])

            vs_sent1 = self._string_to_array(line[5][1:-1])
            vs_sent2 = self._string_to_array(line[6][1:-1])
            vs_sent3 = self._string_to_array(line[7][1:-1])
            vs_sent4 = self._string_to_array(line[8][1:-1])

            vs_sent5_pos = self._string_to_array(line[9][1:-1])
            vs_sent5_neg = self._string_to_array(line[10][1:-1])

            cs_dist_pos = self._string_to_array(line[11][1:-1])
            cs_dist_neg = self._string_to_array(line[12][1:-1])

            examples.append(InputExample(guid=guid, label=label, text_a=text_a,
                                         text_b_pos=text_b_pos, text_b_neg=text_b_neg,
                                         vs_sent1=vs_sent1, vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4,
                                         vs_sent5_pos=vs_sent5_pos, vs_sent5_neg=vs_sent5_neg,
                                         cs_dist_pos=cs_dist_pos, cs_dist_neg=cs_dist_neg))
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1]
        # return ["negative", "positive"]

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "sct_v2.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "test-%s" % (line[0])
            label = int(tokenization.convert_to_unicode(line[1]))

            text_a = tokenization.convert_to_unicode(line[2])

            text_b_pos = tokenization.convert_to_unicode(line[3])
            text_b_neg = tokenization.convert_to_unicode(line[4])

            vs_sent1 = self._string_to_array(line[5][1:-1])
            vs_sent2 = self._string_to_array(line[6][1:-1])
            vs_sent3 = self._string_to_array(line[7][1:-1])
            vs_sent4 = self._string_to_array(line[8][1:-1])

            vs_sent5_pos = self._string_to_array(line[9][1:-1])
            vs_sent5_neg = self._string_to_array(line[10][1:-1])

            cs_dist_pos = self._string_to_array(line[11][1:-1])
            cs_dist_neg = self._string_to_array(line[12][1:-1])

            examples.append(InputExample(guid=guid, label=label, text_a=text_a,
                                         text_b_pos=text_b_pos, text_b_neg=text_b_neg,
                                         vs_sent1=vs_sent1, vs_sent2=vs_sent2, vs_sent3=vs_sent3, vs_sent4=vs_sent4,
                                         vs_sent5_pos=vs_sent5_pos, vs_sent5_neg=vs_sent5_neg,
                                         cs_dist_pos=cs_dist_pos, cs_dist_neg=cs_dist_neg))
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

    if ex_index < 2:
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
        if ex_index % 1000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        example.text_b = example.text_b_pos
        example.vs_sent5 = example.vs_sent5_pos
        example.cs_dist = example.cs_dist_pos
        example.label = 1
        feature_pos = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        example.text_b = example.text_b_neg
        example.vs_sent5 = example.vs_sent5_neg
        example.cs_dist = example.cs_dist_neg
        example.label = 0
        feature_neg = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_tensor_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
            return f

        features = collections.OrderedDict()
        features["input_ids_pos"] = create_int_feature(feature_pos.input_ids)
        features["input_mask_pos"] = create_int_feature(feature_pos.input_mask)
        features["segment_ids_pos"] = create_int_feature(feature_pos.segment_ids)
        features["label_ids_pos"] = create_int_feature([feature_pos.label_id])
        features["is_real_example_pos"] = create_int_feature([int(feature_pos.is_real_example)])
        features["sentiment_pos"] = create_tensor_feature(feature_pos.sentiment)
        features["cs_dist_pos"] = create_tensor_feature(feature_pos.cs_dist)

        features["input_ids_neg"] = create_int_feature(feature_neg.input_ids)
        features["input_mask_neg"] = create_int_feature(feature_neg.input_mask)
        features["segment_ids_neg"] = create_int_feature(feature_neg.segment_ids)
        features["label_ids_neg"] = create_int_feature([feature_neg.label_id])
        features["is_real_example_neg"] = create_int_feature([int(feature_neg.is_real_example)])
        features["sentiment_neg"] = create_tensor_feature(feature_neg.sentiment)
        features["cs_dist_neg"] = create_tensor_feature(feature_neg.cs_dist)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids_pos": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask_pos": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids_pos": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_pos": tf.FixedLenFeature([], tf.int64),
        "is_real_example_pos": tf.FixedLenFeature([], tf.int64),
        "sentiment_pos": tf.FixedLenFeature([5, 4], tf.float32),
        "cs_dist_pos": tf.FixedLenFeature([4], tf.float32),
        "input_ids_neg": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask_neg": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids_neg": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_neg": tf.FixedLenFeature([], tf.int64),
        "is_real_example_neg": tf.FixedLenFeature([], tf.int64),
        "sentiment_neg": tf.FixedLenFeature([5, 4], tf.float32),
        "cs_dist_neg": tf.FixedLenFeature([4], tf.float32),
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
def create_model(bert_model_hub, bert_trainable, bert_config, is_training,
                 input_ids_pos, input_ids_neg, input_mask_pos, input_mask_neg, segment_ids_pos, segment_ids_neg,
                 labels_pos, labels_neg, sentiment_pos, sentiment_neg, cs_dist_pos, cs_dist_neg, num_labels,
                    sentiment_only, commonsense_only, combo):
    """Creates a classification model."""

    # ---------------------------------------------------------------------------------------------------------------- #
    # BERT
    if bert_model_hub:
        bert_module = hub.Module(bert_model_hub, trainable=bert_trainable)

        bert_inputs = dict(input_ids=input_ids_pos, input_mask=input_mask_pos, segment_ids=segment_ids_pos)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        output_layer_pos = bert_outputs["pooled_output"]  # (batch_size, 768)

        bert_inputs = dict(input_ids=input_ids_neg, input_mask=input_mask_neg, segment_ids=segment_ids_neg)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        output_layer_neg = bert_outputs["pooled_output"]  # (batch_size, 768)

    else:
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids_pos,
            input_mask=input_mask_pos,
            token_type_ids=segment_ids_pos,
            use_one_hot_embeddings=False)

        output_layer_pos = model.get_pooled_output()  # (batch_size, 768)

        model = model.__init__(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids_neg,
            input_mask=input_mask_neg,
            token_type_ids=segment_ids_neg,
            use_one_hot_embeddings=False)

        output_layer_neg = model.get_pooled_output()

    hidden_size = output_layer_pos.shape[-1].value  # 768
    batch_size = tf.shape(output_layer_pos)[0]

    # ---------------------------------------------------------------------------------------------------------------- #
    # Sentiment
    sentiment_context = sentiment_pos[:, 0:-1, 1:]  # (batch_size, 4, 3)
    sentiment_answer_pos = sentiment_pos[:, -1, 1:]  # (batch_size, 1, 3)
    sentiment_answer_neg = sentiment_neg[:, -1, 1:]  # (batch_size, 1, 3) or (batch_size, 3)

    hidden_state_dim_sent = 64
    # emb_dim_sent = tf.shape(sentiment_context)[2]  # should be 3
    emb_dim_sent = 3

    # ---------------------------------------------------------------------------------------------------------------- #
    # Weights initialization Sentiment
    with tf.name_scope("weights_initialization_s"):

        output_weights_s = tf.get_variable("output_weights_s", shape=[hidden_state_dim_sent, emb_dim_sent],
                                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        output_bias_s = tf.get_variable("output_bias_s", shape=emb_dim_sent,
                                        initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    with tf.name_scope("similarity_matrix_s"):
        similarity_matrix = tf.get_variable("similarity_matrix", shape=[emb_dim_sent, emb_dim_sent],
                                            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # LSTM Sentiment (story cloze fine tuning)

    with tf.name_scope('lstm_s'):

        LSTM_s = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_state_dim_sent, name="lstm_cell_s")

        initial_state_s = LSTM_s.zero_state(batch_size=batch_size, dtype=tf.float32)

    with tf.name_scope('fine_tuning_story_cloze'):
        lstm_output_sent, (state_c_s, state_h_s) = tf.nn.dynamic_rnn(cell=LSTM_s, inputs=sentiment_context,
                                                                     initial_state=initial_state_s,
                                                                     dtype=tf.float32)

        e_p = tf.nn.softmax(tf.matmul(state_h_s, output_weights_s) + output_bias_s)  # (batch_size, 3)

        logits_pos = tf.linalg.diag_part(
            tf.matmul(tf.matmul(a=e_p, b=similarity_matrix), sentiment_answer_pos, transpose_b=True))

        logits_neg = tf.linalg.diag_part(
            tf.matmul(tf.matmul(a=e_p, b=similarity_matrix), sentiment_answer_neg, transpose_b=True))


        # logits_s = tf.concat([logits_neg, logits_pos], axis=1)
        logits_s = tf.stack([logits_neg, logits_pos], axis=1)
        # logits_s = tf.stack([logits_pos, logits_neg], axis=1)

        probabilities_s = tf.nn.softmax(logits_s, axis=-1)  # (batch_size, 2)

        if sentiment_only:
            log_probs_s = tf.nn.log_softmax(logits_s, axis=-1)

            # Prediction
            one_hot_labels = tf.one_hot(labels_pos, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.argmax(log_probs_s, axis=-1, output_type=tf.int32)

            # Loss
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs_s, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            #
            # tf.logging.info("logits_pos, shape = %s" % logits_pos.shape)  # (batch_size, 1)
            # tf.logging.info("logits_neg, shape = %s" % logits_neg.shape)  # (batch_size, 1)
            # tf.logging.info("logits, shape = %s" % logits_s.shape)  # (batch_size, num_labels)
            #
            # tf.logging.info("probabilities, shape = %s" % probabilities_s.shape)  # (batch_size, num_labels)
            # tf.logging.info("log_probs, shape = %s" % log_probs_s.shape)  # (batch_size, num_labels)
            # tf.logging.info("one_hot_labels, shape = %s" % one_hot_labels.shape)  # (batch_size, num_labels)
            # tf.logging.info("predicted_labels, shape = %s" % predicted_labels.shape)  # (batch_size, )
            # tf.logging.info("per_example_loss, shape = %s" % per_example_loss.shape)  # (batch_size, )
            # tf.logging.info("loss, shape = %s" % loss.shape)

            return loss, per_example_loss, logits_s, probabilities_s, predicted_labels


    # ---------------------------------------------------------------------------------------------------------------- #
    # Common Sense
    cs_dist_neg = cs_dist_neg  # (batch_size, 4)
    cs_dist_pos = cs_dist_pos  # (batch_size, 4)


    with tf.name_scope('commonsense'):

        output_weights_c = tf.get_variable("output_weights_c", [1,4],
                                           initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=True)
        output_bias_c = tf.get_variable("output_bias_c", [1], initializer=tf.zeros_initializer(), trainable=True)

        logits_pos_c = tf.nn.bias_add(tf.matmul(cs_dist_pos,
                                              output_weights_c, transpose_b=True), output_bias_c)  # (batch_size, 1)
        logits_neg_c = tf.nn.bias_add(tf.matmul(cs_dist_neg,
                                              output_weights_c, transpose_b=True), output_bias_c)  # (batch_size, 1)

        logits_c = tf.concat([logits_neg_c, logits_pos_c], axis=1)

        probabilities_c = tf.nn.softmax(logits_c, axis=-1)

        if commonsense_only:

            log_probs_c = tf.nn.log_softmax(logits_c, axis=-1)

            # Prediction
            one_hot_labels = tf.one_hot(labels_pos, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.argmax(log_probs_c, axis=-1, output_type=tf.int32)

            # Loss
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs_c, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            #
            # tf.logging.info("logits_pos, shape = %s" % logits_pos_c.shape)  # (batch_size, 1)
            # tf.logging.info("logits_neg, shape = %s" % logits_neg_c.shape)  # (batch_size, 1)
            # tf.logging.info("logits, shape = %s" % logits_c.shape)  # (batch_size, num_labels)
            #
            # tf.logging.info("probabilities, shape = %s" % probabilities_c.shape)  # (batch_size, num_labels)
            # tf.logging.info("log_probs, shape = %s" % log_probs_c.shape)  # (batch_size, num_labels)
            # tf.logging.info("one_hot_labels, shape = %s" % one_hot_labels.shape)  # (batch_size, num_labels)
            # tf.logging.info("predicted_labels, shape = %s" % predicted_labels.shape)  # (batch_size, )
            # tf.logging.info("per_example_loss, shape = %s" % per_example_loss.shape)  # (batch_size, )
            # tf.logging.info("loss, shape = %s" % loss.shape)

            return loss, per_example_loss, logits_c, probabilities_c, predicted_labels


    # ---------------------------------------------------------------------------------------------------------------- #
    # Weight initialization
    output_weights_n = tf.get_variable("output_weights", [1, hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias_n = tf.get_variable("output_bias", [1], initializer=tf.zeros_initializer())

    # ---------------------------------------------------------------------------------------------------------------- #
    # Loss
    with tf.variable_scope("loss"):

        # Narrative
        if is_training:
            output_layer_pos = tf.nn.dropout(output_layer_pos, rate=0.1)
            output_layer_neg = tf.nn.dropout(output_layer_neg, rate=0.1)

        logits_pos = tf.nn.bias_add(tf.matmul(output_layer_pos,
                                              output_weights_n, transpose_b=True), output_bias_n)  # (batch_size, 1)
        logits_neg = tf.nn.bias_add(tf.matmul(output_layer_neg,
                                              output_weights_n, transpose_b=True), output_bias_n)  # (batch_size, 1)

        logits = tf.concat([logits_neg, logits_pos], axis=1)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # version of combining probabilities: PRODUCT
        if combo:
            probabilities = tf.math.multiply(tf.math.multiply(probabilities, probabilities_c), probabilities_s)
            log_probs = tf.math.log(probabilities)

        # Prediction
        one_hot_labels = tf.one_hot(labels_pos, depth=num_labels, dtype=tf.float32)
        predicted_labels = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        # Loss
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        # tf.logging.info("logits_pos, shape = %s" % logits_pos.shape)  # (batch_size, 1)
        # tf.logging.info("logits_neg, shape = %s" % logits_neg.shape)  # (batch_size, 1)
        # tf.logging.info("logits, shape = %s" % logits.shape)  # (batch_size, num_labels)
        #
        # tf.logging.info("probabilities, shape = %s" % probabilities.shape)  # (batch_size, num_labels)
        # tf.logging.info("log_probs, shape = %s" % log_probs.shape)  # (batch_size, num_labels)
        # tf.logging.info("one_hot_labels, shape = %s" % one_hot_labels.shape)  # (batch_size, num_labels)
        # tf.logging.info("predicted_labels, shape = %s" % predicted_labels.shape)  # (batch_size, )
        # tf.logging.info("per_example_loss, shape = %s" % per_example_loss.shape)  # (batch_size, )
        # tf.logging.info("loss, shape = %s" % loss.shape)

        return loss, per_example_loss, logits, probabilities, predicted_labels


def model_fn_builder(bert_model_hub, bert_trainable, bert_config, init_checkpoint, num_labels, learning_rate,
                     num_train_steps, num_warmup_steps, sentiment_only, commonsense_only, combo, init_checkpoint_sent):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for Estimator."""

        # ------------------------------------------------------------------------------------------------------------ #
        # Input
        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids_pos = features["input_ids_pos"]  # (batch_size, max_seq_length)
        input_mask_pos = features["input_mask_pos"]  # (batch_size, max_seq_length)
        segment_ids_pos = features["segment_ids_pos"]  # (batch_size, max_seq_length)
        label_ids_pos = features["label_ids_pos"]  # (batch_size, )
        sentiment_pos = features["sentiment_pos"]  # (batch_size, 5, 4)
        cs_dist_pos = features["cs_dist_pos"]  # (batch_size, 4)

        input_ids_neg = features["input_ids_neg"]  # (batch_size, max_seq_length)
        input_mask_neg = features["input_mask_neg"]  # (batch_size, max_seq_length)
        segment_ids_neg = features["segment_ids_neg"]  # (batch_size, max_seq_length)
        label_ids_neg = features["label_ids_neg"]  # (batch_size, )
        sentiment_neg = features["sentiment_neg"]  # (batch_size, 5, 4)
        cs_dist_neg = features["cs_dist_neg"]  # (batch_size, 4)

        # ------------------------------------------------------------------------------------------------------------ #
        # Model
        loss, per_example_loss, logits, probabilities, predicted_labels = create_model(bert_model_hub,
                                                                                       bert_trainable,
                                                                                       bert_config,
                                                                                       mode == tf.estimator.ModeKeys.TRAIN,
                                                                                       input_ids_pos, input_ids_neg,
                                                                                       input_mask_pos, input_mask_neg,
                                                                                       segment_ids_pos, segment_ids_neg,
                                                                                       label_ids_pos, label_ids_neg,
                                                                                       sentiment_pos, sentiment_neg,
                                                                                       cs_dist_pos, cs_dist_neg,
                                                                                       num_labels,
                                                                                       sentiment_only,
                                                                                       commonsense_only,
                                                                                       combo)

        # ------------------------------------------------------------------------------------------------------------ #
        # Initialize
        tvars = tf.trainable_variables()

        tvars_sent = [var for var in tvars if 'module' not in var.name]
        tvars = [var for var in tvars if 'module' in var.name]

        if bert_model_hub:
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if "bert" in var.name and mode == tf.estimator.ModeKeys.TRAIN:
                    init_string = ", *INIT_FROM_HUB*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

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
        # Initialize Sentiment

        initialized_variable_names_sent = {}

        if init_checkpoint_sent:
            (assignment_map_sent, initialized_variable_names_sent) = modeling.get_assignment_map_from_checkpoint(
                tvars_sent, init_checkpoint_sent)
            tf.train.init_from_checkpoint(init_checkpoint_sent, assignment_map_sent)

        tf.logging.info("**** Trainable Variables Sentiment ****")
        for var in tvars_sent:
            init_string = ""
            if var.name in initialized_variable_names_sent:
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

                recall = tf.metrics.recall(_label_ids, _predictions)
                precision = tf.metrics.precision(_label_ids, _predictions)

                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "recall": recall,
                    "precision": precision,
                }

            eval_metrics = metric_fn(per_example_loss, label_ids_pos, predicted_labels)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=loss,
                                                     eval_metric_ops=eval_metrics)

        else:
            predictions = {
                'probabilities': probabilities,
                "predict_labels": predicted_labels + 1,
            }

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions=predictions)
        return output_spec

    return model_fn
