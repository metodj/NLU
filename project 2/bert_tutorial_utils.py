# colab notebook can be found here: https://colab.research.google.com/drive/1Sl5wXmufsJi7DG9eWS75pthjUB8N_oi5
import pandas as pd

def data_to_bert_tutorial(data_path):
  our_data = pd.read_csv(data_path)
  our_data['4_sentences'] = our_data[["InputSentence1","InputSentence2","InputSentence3","InputSentence4"]].apply(lambda x: ' '.join(x), axis=1)
  processed_data = pd.DataFrame(columns=['story', 'ending', 'label'])
  for _, row in our_data.iterrows():
    sample1 = {'story': row["4_sentences"], 'ending': row["RandomFifthSentenceQuiz1"], 'label': 1 if row['AnswerRightEnding']==1 else 0}
    sample2 = {'story': row["4_sentences"], 'ending': row["RandomFifthSentenceQuiz2"], 'label': 1 if row['AnswerRightEnding']==2 else 0}
    processed_data = processed_data.append(sample1, ignore_index=True)
    processed_data = processed_data.append(sample2, ignore_index=True)

  processed_data['label'] = processed_data['label'].astype(int)
  return processed_data