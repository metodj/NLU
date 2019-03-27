import tensorflow as tf


def parse_ids_file(line):
    line_split = tf.string_split([line])
    input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
    output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
    return input_seq, output_seq


def variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def trainable_parameters():
    tot_count = 0

    for var in tf.trainable_variables():
        v_count = 1
        for d in var.get_shape():
            v_count *= d.value
        tot_count += v_count

        print("{:<30}{:<10}".format(var.name, str(var.get_shape())))

    print("{:<30}{:<10}".format("num_parameters", tot_count))
