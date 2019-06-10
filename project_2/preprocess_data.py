import os
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def pp_data_train(df, method="random", sentiment=True, common_sense=True):
    df = df.copy()
    cols = list(df.columns)

    def pp_text_a(row):
        text_a = [row[cols.index("InputSentence1")], \
                  row[cols.index("InputSentence2")], \
                  row[cols.index("InputSentence3")], \
                  row[cols.index("InputSentence4")]]
        return text_a

    def pp_text_b_pos(row):
        text_b = row[cols.index("sentence5")]
        return text_b

    def pp_sentiment(row):
        sentiment = analyser.polarity_scores(row)
        np_vec = np.array([sentiment["compound"], sentiment["neg"], sentiment["neu"], sentiment["pos"]],
                          dtype=np.float32)
        return np_vec

    def pp_common_sense(row):
        distance = compute_distance(row["text_a"], row["text_b"])
        return np.array(distance, dtype=np.float32)

    # Positive samples (true)
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df["storyid"]
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)
    df_pos["text_b"] = df.apply(pp_text_b_pos, axis=1)
    df_pos["label"] = 1  # "positive"

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["sentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["sentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["sentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["sentence4"].apply(pp_sentiment)
        df_pos["vs_sent5"] = df["sentence5"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_pos["cs_dist"] = df_pos.apply(pp_common_sense)

    # Negative samples
    if method == "random":
        df_neg = pd.DataFrame()
        df_neg["unique_id"] = df_pos["unique_id"]
        df_neg["text_a"] = df_pos["text_a"]
        df_neg["text_b"] = df["sentence3"].sample(frac=1.0, random_state=0).reset_index(drop=True)  # Random sentence3
        df_neg["label"] = 0  # "negative"

        # Sentiment
        if sentiment:
            df_neg["vs_sent1"] = df_pos["vs_sent1"]
            df_neg["vs_sent2"] = df_pos["vs_sent2"]
            df_neg["vs_sent3"] = df_pos["vs_sent3"]
            df_neg["vs_sent4"] = df_pos["vs_sent4"]
            df_neg["vs_sent5"] = df_neg["text_b"].apply(pp_sentiment)

        # Common Sense
        if common_sense:
            df_neg["cs_dist"] = df_neg.apply(pp_common_sense)

    # Sort by index: positive - first, negative - second
    df_res = pd.concat([df_pos, df_neg], axis=0).set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


def pp_data_val(df, sentiment=True, common_sense=True):
    df = df.copy()
    cols = list(df.columns)

    def pp_text_a(row):
        text_a = [row[cols.index("InputSentence1")], \
                  row[cols.index("InputSentence2")], \
                  row[cols.index("InputSentence3")], \
                  row[cols.index("InputSentence4")]]
        return text_a

    def pp_text_b_pos(row):
        if row[cols.index("AnswerRightEnding")] == 1:
            return row[cols.index("RandomFifthSentenceQuiz1")]
        elif row[cols.index("AnswerRightEnding")] == 2:
            return row[cols.index("RandomFifthSentenceQuiz2")]

    def pp_text_b_neg(row):
        if row[cols.index("AnswerRightEnding")] == 2:
            return row[cols.index("RandomFifthSentenceQuiz1")]
        elif row[cols.index("AnswerRightEnding")] == 1:
            return row[cols.index("RandomFifthSentenceQuiz2")]

    def pp_sentiment(row):
        sentiment = analyser.polarity_scores(row)
        np_vec = np.array([sentiment["compound"], sentiment["neg"], sentiment["neu"], sentiment["pos"]],
                          dtype=np.float32)
        return np_vec

    def pp_common_sense(row):
        distance = compute_distance(row["text_a"], row["text_b"])
        return np.array(distance, dtype=np.float32)

    # Positive samples
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df["InputStoryid"]
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)
    df_pos["text_b"] = df.apply(pp_text_b_pos, axis=1)
    df_pos["label"] = 1  # "positive"

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["InputSentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["InputSentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["InputSentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["InputSentence4"].apply(pp_sentiment)
        df_pos["vs_sent5"] = df_pos["text_b"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_pos["cs_dist"] = df_pos.apply(pp_common_sense)

    df_neg = pd.DataFrame()
    df_neg["unique_id"] = df["InputStoryid"]
    df_neg["text_a"] = df.apply(pp_text_a, axis=1)
    df_neg["text_b"] = df.apply(pp_text_b_neg, axis=1)
    df_neg["label"] = 0  # "negative"

    # Sentiment
    if sentiment:
        df_neg["vs_sent1"] = df_pos["vs_sent1"]
        df_neg["vs_sent2"] = df_pos["vs_sent2"]
        df_neg["vs_sent3"] = df_pos["vs_sent3"]
        df_neg["vs_sent4"] = df_pos["vs_sent4"]
        df_neg["vs_sent5"] = df_neg["text_b"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_neg["cs_dist"] = df_neg.apply(pp_common_sense)

    df_res = pd.concat([df_pos, df_neg], axis=0).set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


def pp_data_test(df, sentiment=True, common_sense=True):
    df = df.copy()
    cols = list(df.columns)

    def pp_text_a(row):
        text_a = [row[cols.index("InputSentence1")], \
                  row[cols.index("InputSentence2")], \
                  row[cols.index("InputSentence3")], \
                  row[cols.index("InputSentence4")]]
        return text_a

    def pp_sentiment(row):
        sentiment = analyser.polarity_scores(row)
        np_vec = np.array([sentiment["compound"], sentiment["neg"], sentiment["neu"], sentiment["pos"]],
                          dtype=np.float32)
        return np_vec

    def pp_common_sense(row):
        distance = compute_distance(row["text_a"], row["text_b"])
        return np.array(distance, dtype=np.float32)

    # Positive samples
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df.index
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)
    df_pos["text_b"] = df["RandomFifthSentenceQuiz2"]
    df_pos["label"] = 1  # "positive"

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["InputSentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["InputSentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["InputSentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["InputSentence4"].apply(pp_sentiment)
        df_pos["vs_sent5"] = df_pos["text_b"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_pos["cs_dist"] = df_pos.apply(pp_common_sense)

    df_neg = pd.DataFrame()
    df_neg["unique_id"] = df.index
    df_neg["text_a"] = df.apply(pp_text_a, axis=1)
    df_neg["text_b"] = df["RandomFifthSentenceQuiz1"]
    df_neg["label"] = 0  # "negative"

    # Sentiment
    if sentiment:
        df_neg["vs_sent1"] = df_pos["vs_sent1"]
        df_neg["vs_sent2"] = df_pos["vs_sent2"]
        df_neg["vs_sent3"] = df_pos["vs_sent3"]
        df_neg["vs_sent4"] = df_pos["vs_sent4"]
        df_neg["vs_sent5"] = df_neg["text_b"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_neg["cs_dist"] = df_neg.apply(pp_common_sense)

    df_res = pd.concat([df_pos, df_neg], axis=0).set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


# Tokenization
# ------------

def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    words = [w for w in words if w not in stoplist]

    return words

def stem(word):
    stem = stemmer.stem(word)
    return stem


# Vectors
# -------

def compute_distance(S, e):
    distance = []
    for s_j in S:
        distance_j = 0 
        num = 0
        for w in tokenize(e):
            max_d = max(cosine_similarity(w, u) for u in tokenize(s_j) if stem(w) != stem(u))
            num += 1

            distance_j += max_d 
        distance_j /= num
        distance.append(distance_j)
    return distance

def get_vector(word):
    if word in numberbatch_vectors:
        return numberbatch_vectors[word]

    return []

def cosine_similarity(word1, word2):
    vector1, vector2 = get_vector(word1), get_vector(word2)

    # Abort if words not in list
    if not vector1 or not vector2:
        return 0

    similarity = np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity

# Files
# -----

def read_numberbatch(input_file):
    vectors = {}
    with open(input_file, encoding='latin-1') as file:
        for (i, line) in enumerate(file):
            if i == 0:
                continue
            if i == 10:
                break

            line = line.split(" ")
            word = line[0]
            vector = [float(v) for v in line[1:]]
            vectors[word] = vector
    return vectors


if __name__ == "__main__":
    # Download NLTK content.
    nltk.download('stopwords')
    nltk.download('punkt')

    stoplist = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    analyser = SentimentIntensityAnalyzer()

    # Load numberbatch vectors.
    numberbatch_vectors = read_numberbatch(os.path.join("..\\..\\numberbatch", "numberbatch-en.txt"))

    # Load data
    data_train_val = pd.read_csv(os.path.join("data", "cloze_test_val__spring2016 - cloze_test_ALL_val.csv"))
    data_train = pd.read_csv(os.path.join("data", "train_stories.csv"))

    data_val = pd.read_csv(os.path.join("data", "test_for_report-stories_labels.csv"))
    data_test = pd.read_csv(os.path.join("data", "test-stories.csv"))

    # Training set: train + validation (fine-tuning)
    df_train = pp_data_train(data_train, sentiment=True, common_sense=True)
    df_train_val = pp_data_val(data_train_val, sentiment=True, common_sense=True)

    # df_train = pd.concat([df_train, df_train_val, df_train_val, df_train_val], axis=0)

    df_train_tmp = df_train.copy()
    df_train = df_train_val

    # Validation set
    df_val = pp_data_val(data_val, sentiment=True, common_sense=True)

    # Test set (no label)
    df_test = pp_data_test(data_test, sentiment=True, common_sense=True)

    df_train.to_csv(os.path.join("data_pp", "sct.train.tsv"), sep="\t", header=True, index=False)
    df_val.to_csv(os.path.join("data_pp", "sct.validation.tsv"), sep="\t", header=True, index=False)
    df_test.to_csv(os.path.join("data_pp", "sct.test.tsv"), sep="\t", header=True, index=False)

    if not os.path.exists("data_pp_test"):
        os.makedirs("data_pp_test")

    df_train_test = pd.concat([df_train_tmp.iloc[0:data_train_val.shape[0]],
                               df_train.iloc[-data_train_val.shape[0]:]], axis=0)

    df_train_test = df_train_test.sample(frac=1.0, random_state=0).reset_index(drop=True)

    df_train_test.to_csv(os.path.join("data_pp_test", "sct.train.tsv"), sep="\t", header=True, index=False)
    df_val.to_csv(os.path.join("data_pp_test", "sct.validation.tsv"), sep="\t", header=True, index=False)
    df_test.to_csv(os.path.join("data_pp_test", "sct.test.tsv"), sep="\t", header=True, index=False)

    # max_seq_len
    idx_a = list(df_train.columns).index("text_a")
    idx_b = list(df_train.columns).index("text_b")
    print("train_len", df_train.apply(lambda row: len(row[idx_a]), axis=1).max() +
          df_train.apply(lambda row: len(row[idx_b]), axis=1).max())
    print("val_len", df_val.apply(lambda row: len(row[idx_a]), axis=1).max() +
          df_val.apply(lambda row: len(row[idx_b]), axis=1).max())
    print("test_len", df_test.apply(lambda row: len(row[idx_a]), axis=1).max() +
          df_test.apply(lambda row: len(row[idx_b]), axis=1).max())
