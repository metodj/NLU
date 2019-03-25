from collections import Counter
import numpy as np


def load_sentences(file):
    sentences = []

    with open(file, "r") as f:
        for line in f.readlines():

            line = line.strip().split(" ")

            if len(line) <= 28:
                padding = ["<pad>" for _ in range(28 - len(line))]

                tokens = ["<bos>"]
                tokens.extend(line)
                tokens.append("<eos>")
                tokens.extend(padding)

                sentences.append(tokens)

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

