import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def pp_data_train(df, method="random", sentiment=True):
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

    # Positive samples (true)
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df["storyid"]
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)
    df_pos["text_b"] = df.apply(pp_text_b_pos, axis=1)
    df_pos["label"] = "positive"

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["sentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["sentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["sentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["sentence4"].apply(pp_sentiment)
        df_pos["vs_sent5"] = df["sentence5"].apply(pp_sentiment)

    # Negative samples
    if method == "random":
        df_neg = pd.DataFrame()
        df_neg["unique_id"] = df_pos["unique_id"]
        df_neg["text_a"] = df_pos["text_a"]
        df_neg["text_b"] = df["sentence3"].sample(frac=1.0, random_state=0).reset_index(drop=True)  # Random sentence3
        df_neg["label"] = "negative"

        # Sentiment
        if sentiment:
            df_neg["vs_sent1"] = df_pos["vs_sent1"]
            df_neg["vs_sent2"] = df_pos["vs_sent2"]
            df_neg["vs_sent3"] = df_pos["vs_sent3"]
            df_neg["vs_sent4"] = df_pos["vs_sent4"]
            df_neg["vs_sent5"] = df_neg["text_b"].apply(pp_sentiment)

    # Sort by index: positive - first, negative - second
    df_res = pd.concat([df_pos, df_neg], axis=0).set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


def pp_data_val(df, sentiment=True):
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

    # Positive samples
    df_pos = pd.DataFrame()
    df_pos["unique_id"] = df["InputStoryid"]
    df_pos["text_a"] = df.apply(pp_text_a, axis=1)
    df_pos["text_b"] = df.apply(pp_text_b_pos, axis=1)
    df_pos["label"] = "positive"

    # Sentiment
    if sentiment:
        df_pos["vs_sent1"] = df["InputSentence1"].apply(pp_sentiment)
        df_pos["vs_sent2"] = df["InputSentence2"].apply(pp_sentiment)
        df_pos["vs_sent3"] = df["InputSentence3"].apply(pp_sentiment)
        df_pos["vs_sent4"] = df["InputSentence4"].apply(pp_sentiment)
        df_pos["vs_sent5"] = df_pos["text_b"].apply(pp_sentiment)

    df_neg = pd.DataFrame()
    df_neg["unique_id"] = df["InputStoryid"]
    df_neg["text_a"] = df.apply(pp_text_a, axis=1)
    df_neg["text_b"] = df.apply(pp_text_b_neg, axis=1)
    df_neg["label"] = "negative"

    # Sentiment
    if sentiment:
        df_neg["vs_sent1"] = df_pos["vs_sent1"]
        df_neg["vs_sent2"] = df_pos["vs_sent2"]
        df_neg["vs_sent3"] = df_pos["vs_sent3"]
        df_neg["vs_sent4"] = df_pos["vs_sent4"]
        df_neg["vs_sent5"] = df_neg["text_b"].apply(pp_sentiment)

    df_res = pd.concat([df_pos, df_neg], axis=0).set_index(["unique_id", "label"])\
        .sort_index(axis=0, ascending=False).reset_index(drop=False)

    return df_res


if __name__ == "__main__":
    analyser = SentimentIntensityAnalyzer()

    data_val = pd.read_csv(os.path.join("data", "cloze_test_val__spring2016 - cloze_test_ALL_val.csv"))
    data_train = pd.read_csv(os.path.join("data", "train_stories.csv"))

    # Create dataframes
    df_train = pp_data_train(data_train, sentiment=True)
    df_val = pp_data_val(data_val, sentiment=True)
    df_test = df_val[["unique_id", "label", "text_a", "text_b", "vs_sent1", "vs_sent2", "vs_sent3", "vs_sent4", "vs_sent5"]]\
        .sample(frac=0.5, random_state=0).reset_index(drop=True)

    # Save as .tsv
    df_train.to_csv(os.path.join("data_pp", "sct.train.tsv"), sep="\t", header=True, index=False)
    df_val.to_csv(os.path.join("data_pp", "sct.validation.tsv"), sep="\t", header=True, index=False)
    df_test.to_csv(os.path.join("data_pp", "sct.test.tsv"), sep="\t", header=True, index=False)

    # max_seq_len
    idx = list(df_train.columns).index("text_a")
    print("train_len", df_train.apply(lambda row: len(row[idx]), axis=1).max())
    print("val_len", df_val.apply(lambda row: len(row[idx]), axis=1).max())
    print("test_len", df_test.apply(lambda row: len(row[1]), axis=1).max())