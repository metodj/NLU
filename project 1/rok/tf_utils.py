import tensorflow as tf


def parse_ids_file(line):
    line_split = tf.string_split([line])
    input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
    output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
    return input_seq, output_seq
