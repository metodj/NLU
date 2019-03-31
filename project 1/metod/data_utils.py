from collections import Counter
import numpy as np
import os


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
    vocab_20k[:10]

    word_list = list(set(tokens))
    word_to_id_dict = dict({'<bos>': 1, '<eos>': 2, '<pad>': 3, '<unk>': 4})
    id_counter = 5

    for word in word_list:
        if word_to_id_dict.get(word, 0) == 0 and word in vocab_20k:
            word_to_id_dict[word] = id_counter
            id_counter += 1

    return word_to_id_dict


def word_to_ids(input_file, output_file, word_id_dict):

    words_ids = np.empty((0, 30))
    with open(input_file, "r") as file:
        for line in file:
            tokens = _sentence_preprocessing(line)
            if tokens:
                words_ids = np.vstack((words_ids, np.array([word_id_dict.get(word, 4) - 1 for word in tokens])))

    file.close()

    with open(output_file, 'w') as file2:
        for i in range(words_ids.shape[0]):
            file2.write(' '.join(str(x) for x in list(words_ids[i, :])) + '\n')

    file2.close()


# names = ['train.ids', 'eval.ids', 'test.ids']


# if __name__ == "__main__":
#     word_to_id_dict = build_vocab()
# 
#     i = 0
#     for file in os.listdir():
#         filename = os.fsdecode(file)
#         if 'sentences' in filename:
#            word_to_ids(filename, names[i], word_to_id_dict)
#            i += 1





