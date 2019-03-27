from collections import Counter
import numpy as np
import time
from datetime import datetime
import logging


def load_sentences(file):
    sentences = []

    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            if len(line) <= 28:
                line = ["<bos>"] + line + ["<eos>"] + ["<pad>"] * (28 - len(line))
                sentences.append(line)

    return sentences


def create_vocabulary(file, vocab_size):
    sentences = load_sentences(file)

    words = [word for sent in sentences for word in sent]
    vocab = [pair[0] for pair in Counter(words).most_common(vocab_size)]

    if "<unk>" not in vocab:
        vocab.pop()
        vocab.append("<unk>")
    if "<bos>" not in vocab:
        vocab.pop()
        vocab.append("<bos>")
    if "<eos>" not in vocab:
        vocab.pop()
        vocab.append("<eos>")
    if "<pad>" not in vocab:
        vocab.pop()
        vocab.append("<pad>")

    vocab.sort()

    word_to_idx = dict(map(reversed, enumerate(vocab)))
    idx_to_word = dict(enumerate(vocab))

    return vocab, word_to_idx, idx_to_word


def create_dataset(file, word_to_idx):
    sentences = load_sentences(file)

    sent_len = len(sentences[0])
    num_sents = len(sentences)

    x = np.zeros(shape=(num_sents, sent_len), dtype=np.int32)

    idx_unk = word_to_idx["<unk>"]
    idx_pad = word_to_idx["<pad>"]
    idx_bos = word_to_idx["<bos>"]
    idx_eos = word_to_idx["<eos>"]

    for sent_id, sent in enumerate(sentences):
        for token_pos, token in enumerate(sent):

            if token == "<pad>":
                x[sent_id, token_pos] = idx_pad
            elif token == "<unk>" or token not in word_to_idx:
                x[sent_id, token_pos] = idx_unk
            elif token == "<bos>":
                x[sent_id, token_pos] = idx_bos
            elif token == "<eos>":
                x[sent_id, token_pos] = idx_eos
            elif token in word_to_idx:
                x[sent_id, token_pos] = word_to_idx[token]

    return x


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self):
        print('Elapsed: %ss' % (time.time() - self.tstart))


class Logger(object):
    def __init__(self, log_dir):
        self.name = log_dir + "log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # logging.basicConfig(filename=self.name, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    @staticmethod
    def append(x, *args):
        # logging.info(str(x))
        if len(args) == 1:
            print("{:<40}{:<15}".format(x, str(args[0])))
        elif len(args) == 2:
            print("{:<40}{:<15}{:<15}".format(x, str(args[0]), str(args[1])))
        else:
            print("{:<40}".format(x))

    @staticmethod
    def create_log():
        logging.shutdown()