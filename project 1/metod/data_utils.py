from collections import Counter
import numpy as np
import json
import os
import tensorflow as tf

def count_trainable_parameters():
    """Counts the number of trainable parameters in the current default graph."""
    tot_count = 0
    for v in tf.trainable_variables():
        v_count = 1
        for d in v.get_shape():
            v_count *= d.value
        tot_count += v_count
    return tot_count
	
def _sentence_preprocessing(sentence):
    sentence = ['<bos>'] + sentence.split() + ['<eos>']
    if len(sentence) <= 30:
        sentence = sentence + ['<pad>']*(30-len(sentence))
        return sentence
    # we ignore sentences with more than 30 words/tokens
    return []


def build_vocab():
    '''
    builds word to id dictionary based on training set corpus
    '''
    tokens = []

    with open("sentences.train", "r") as file:
        for line in file:
            tokens.extend(_sentence_preprocessing(line))

    file.close()

    # build vocab of 20k most frequent words
    vocab = Counter(tokens)
    vocab_20k = list(
        map(lambda x: x[0], vocab.most_common(19999)))  # 19999 cause last, 20000th token, is reserved for <unk>

    word_list = list(set(tokens))
    word_to_id_dict = dict({'<bos>': 1, '<eos>': 2, '<pad>': 3, '<unk>': 4})
    id_counter = 5

    for word in word_list:
        if word_to_id_dict.get(word, 0) == 0 and word in vocab_20k:
            word_to_id_dict[word] = id_counter
            id_counter += 1

    return word_to_id_dict


	
def word_to_ids(input_file, output_file, word_id_dict):
	tokens = []

	with open(input_file, "r") as file:
		for line in file:
			tokens.extend(_sentence_preprocessing(line))
			
	file.close()
	
	words_ids = np.array([word_id_dict.get(word,4)-1 for word in tokens]).reshape(int(len(tokens)/30), 30)
	
	with open(output_file, 'w') as file:
		for i in range(words_ids.shape[0]):
			file.write(' '.join(str(x) for x in list(words_ids[i,:])) + '\n')
    
	file.close()
	print(output_file + " written")
	
def word_to_ids_old(input_file, output_file, word_id_dict):

    words_ids = np.empty((0, 30), dtype=np.int8)
    with open(input_file, "r") as file:
        for line in file:
            tokens = _sentence_preprocessing(line)
            if tokens:
                words_ids = np.vstack((words_ids, [word_id_dict.get(word, 4) - 1 for word in tokens]))


    file.close()

    with open(output_file, 'w') as file2:
        for i in range(words_ids.shape[0]):
            file2.write(' '.join(str(int(x)) for x in list(words_ids[i, :])) + '\n')

    file2.close()






def variable_summaries(var):
    """Attach a lot of summaries to a Tensor for TensorBoard visualizations."""
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


#word_to_id_dict = build_vocab()
#word_to_ids(FILE_TRAIN, INPUT_TRAIN, word_to_id_dict)
#word_to_ids(FILE_EVAL, INPUT_EVAL, word_to_id_dict)
#word_to_ids(FILE_TEST, INPUT_TEST, word_to_id_dict)




