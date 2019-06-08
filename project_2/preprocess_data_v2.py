import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def pp_data_train(df, method="random", sentiment=True, common_sense=True):
    df = df.copy()
    cols = list(df.columns)

    def pp_text_a(row):
        text_a = ""
        text_a = text_a + " " + row[cols.index("sentence1")]  # sentence1
        text_a = text_a + " " + row[cols.index("sentence2")]  # sentence2
        text_a = text_a + " " + row[cols.index("sentence3")]  # sentence3
        text_a = text_a + " " + row[cols.index("sentence4")]  # sentence4
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
        return np.ones(shape=(4,), dtype=np.float32)

    # Positive samples (true)
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df["storyid"]
    df_pos["label"] = 1  # "positive"

    df_pos["text_a"] = df.apply(pp_text_a, axis=1)
    df_pos["text_b_pos"] = df.apply(pp_text_b_pos, axis=1)

    if method == "random":
        df_pos["text_b_neg"] = df["sentence3"].sample(frac=1.0, random_state=0).reset_index(drop=True)

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["sentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["sentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["sentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["sentence4"].apply(pp_sentiment)
        df_pos["vs_sent5_pos"] = df_pos["text_b_pos"].apply(pp_sentiment)
        df_pos["vs_sent5_neg"] = df_pos["text_b_neg"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_pos["cs_dist_pos"] = df_pos["text_a"].apply(pp_common_sense)
        df_pos["cs_dist_neg"] = -df_pos["text_a"].apply(pp_common_sense)

    # Sort by index: positive - first, negative - second
    df_res = df_pos.set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


def pp_data_val(df, sentiment=True, common_sense=True):
    df = df.copy()
    cols = list(df.columns)

    def pp_text_a(row):
        text_a = ""
        text_a = text_a + " " + row[cols.index("InputSentence1")]  # sentence1
        text_a = text_a + " " + row[cols.index("InputSentence2")]  # sentence2
        text_a = text_a + " " + row[cols.index("InputSentence3")]  # sentence3
        text_a = text_a + " " + row[cols.index("InputSentence4")]  # sentence4
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
        return np.ones(shape=(4,), dtype=np.float32)

    # Positive samples
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df["InputStoryid"]
    df_pos["label"] = 1  # "positive"
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)

    df_pos["text_b_pos"] = df.apply(pp_text_b_pos, axis=1)
    df_pos["text_b_neg"] = df.apply(pp_text_b_neg, axis=1)

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["InputSentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["InputSentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["InputSentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["InputSentence4"].apply(pp_sentiment)
        df_pos["vs_sent5_pos"] = df_pos["text_b_pos"].apply(pp_sentiment)
        df_pos["vs_sent5_neg"] = df_pos["text_b_neg"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_pos["cs_dist_pos"] = df_pos["text_a"].apply(pp_common_sense)
        df_pos["cs_dist_neg"] = -df_pos["text_a"].apply(pp_common_sense)

    df_res = df_pos.set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


def pp_data_test(df, sentiment=True, common_sense=True):
    df = df.copy()
    cols = list(df.columns)

    def pp_text_a(row):
        text_a = ""
        text_a = text_a + " " + row[cols.index("InputSentence1")]  # sentence1
        text_a = text_a + " " + row[cols.index("InputSentence2")]  # sentence2
        text_a = text_a + " " + row[cols.index("InputSentence3")]  # sentence3
        text_a = text_a + " " + row[cols.index("InputSentence4")]  # sentence4
        return text_a

    def pp_sentiment(row):
        sentiment = analyser.polarity_scores(row)
        np_vec = np.array([sentiment["compound"], sentiment["neg"], sentiment["neu"], sentiment["pos"]],
                          dtype=np.float32)
        return np_vec

    def pp_common_sense(row):
        return np.ones(shape=(4,), dtype=np.float32)

    # Positive samples
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df.index
    df_pos["label"] = 1  # "positive"
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)

    df_pos["text_b_pos"] = df["RandomFifthSentenceQuiz2"]
    df_pos["text_b_neg"] = df["RandomFifthSentenceQuiz1"]

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["InputSentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["InputSentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["InputSentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["InputSentence4"].apply(pp_sentiment)
        df_pos["vs_sent5_pos"] = df_pos["text_b_pos"].apply(pp_sentiment)
        df_pos["vs_sent5_neg"] = df_pos["text_b_neg"].apply(pp_sentiment)

    # Common Sense
    if common_sense:
        df_pos["cs_dist_pos"] = df_pos["text_a"].apply(pp_common_sense)
        df_pos["cs_dist_neg"] = -df_pos["text_a"].apply(pp_common_sense)

    df_res = df_pos

    return df_res


if __name__ == "__main__":
    analyser = SentimentIntensityAnalyzer()

    # Load data
    data_train_val = pd.read_csv(os.path.join("data",
                                              "cloze_test_val__spring2016 - cloze_test_ALL_val.csv"))

    data_train = pd.read_csv(os.path.join("data", "train_stories.csv"))

    data_val = pd.read_csv(os.path.join("data", "test_for_report-stories_labels.csv"))
    data_test = pd.read_csv(os.path.join("data", "test-stories.csv"))

    # Training set: train + validation (fine-tuning)
    df_train = pp_data_train(data_train, sentiment=True, common_sense=True)
    df_train_val = pp_data_val(data_train_val, sentiment=True, common_sense=True)

    df_train_val = pd.concat([df_train_val, df_train_val, df_train_val], axis=0)\
        .sample(frac=1.0, random_state=0).reset_index(drop=True)
    df_train = pd.concat([df_train, df_train_val], axis=0)

    # Validation set
    df_val = pp_data_val(data_val, sentiment=True, common_sense=True)

    # Test set (no label)
    df_test = pp_data_test(data_test, sentiment=True, common_sense=True)

    df_train.to_csv(os.path.join("data_pp", "sct_v2.train.tsv"), sep="\t", header=True, index=False)
    df_val.to_csv(os.path.join("data_pp", "sct_v2.validation.tsv"), sep="\t", header=True, index=False)
    df_test.to_csv(os.path.join("data_pp", "sct_v2.test.tsv"), sep="\t", header=True, index=False)

    if not os.path.exists("data_pp_test"):
        os.makedirs("data_pp_test")

    df_train_test = pd.concat([df_train.iloc[0:data_train_val.shape[0]],
                               df_train.iloc[-data_train_val.shape[0]:]], axis=0)

    df_train_test = df_train_test.sample(frac=1.0, random_state=0).reset_index(drop=True)

    df_train_test.to_csv(os.path.join("data_pp_test", "sct_v2.train.tsv"), sep="\t", header=True, index=False)
    df_val.to_csv(os.path.join("data_pp_test", "sct_v2.validation.tsv"), sep="\t", header=True, index=False)
    df_test.to_csv(os.path.join("data_pp_test", "sct_v2.test.tsv"), sep="\t", header=True, index=False)

    # max_seq_len
    idx_a = list(df_train.columns).index("text_a")
    idx_b_pos = list(df_train.columns).index("text_b_pos")
    idx_b_neg = list(df_train.columns).index("text_b_neg")

    print("train_len", df_train.apply(lambda row: len(row[idx_a]), axis=1).max() +
          df_train.apply(lambda row: max(len(row[idx_b_pos]), len(row[idx_b_neg])), axis=1).max())
    print("val_len", df_val.apply(lambda row: len(row[idx_a]), axis=1).max() +
          df_val.apply(lambda row: max(len(row[idx_b_pos]), len(row[idx_b_neg])), axis=1).max())
    print("test_len", df_test.apply(lambda row: len(row[idx_a]), axis=1).max() +
          df_test.apply(lambda row: max(len(row[idx_b_pos]), len(row[idx_b_neg])), axis=1).max())
