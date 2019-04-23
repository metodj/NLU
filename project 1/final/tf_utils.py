import tensorflow as tf


def parse_ids_file(line):
    line_split = tf.string_split([line])
    input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
    output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
    return input_seq, output_seq


def parse_cont_ids_file(line):
    line_split = tf.string_split([line])
    input_seq = tf.string_to_number(line_split.values, out_type=tf.int32)
    return input_seq


def trainable_parameters(*args):
    tot_count = 0

    if len(args) == 1:
        logger = args[0]
        for var in tf.trainable_variables():
            v_count = 1
            for d in var.get_shape():
                v_count *= d.value
            tot_count += v_count
            logger.append(var.name, str(var.get_shape()))

        logger.append("num_parameters", tot_count)
    else:
        for var in tf.trainable_variables():
            v_count = 1
            for d in var.get_shape():
                v_count *= d.value
            tot_count += v_count
            print("{:<35}:{:<10}".format(var.name, str(var.get_shape())))

        print("{:<35}:{:<10}\n".format("num_parameters", tot_count))


def print_flags(flags, *args):
    if len(args) == 1:
        logger = args[0]
        for key in flags.flag_values_dict():
            if key.upper() != "F":
                logger.append(key.upper(), flags[key].value)
    else:
        print("Command-line Arguments")
        for key in flags.flag_values_dict():
            if key.upper() != "F":
                print("{:<35}: {}".format(key.upper(), flags[key].value))
        print("\n")


def delete_flags(flags):
    flags_dict = flags._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        flags.__delattr__(keys)
