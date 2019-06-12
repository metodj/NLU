## Natural Language Understanding
## Project 2

### Code and environment 

All dependencies are listed in setup.py (to set up the environment run "python setup.py install").
All data and model directories can be found here: https://drive.google.com/drive/folders/16ofhrtO6wW_tn-0II0k3fD7RA2QpD1uG?fbclid=IwAR1YJ2YMM3-wEjpmfXDrGycKVZ92RrD5SDqnCXgppkB8EL6MWsuWsMRW9is

Relevant folders are (need to be downloaded into the same directory as the python scripts):
* data
* data_pp
* data-pp_test
* bert_masked_lm_pp
* bert

### SCT absolute classification ```bert_sct.py``` (Colab) or ```bert_sct.py/ipynb``` (Colab)

#### Original data

Folder ```./data/``` contains:
* cloze_test_val__spring2016 - cloze_test_ALL_val.csv
* test-stories.csv
* test_for_report-stories_labels.csv
* train_stories.csv

#### Data Format

Data format in ```./data_pp/``` (TS IV) and ```./data_pp_test/``` (TS III):
* ```sct.train.tsv```
* ```sct.validation.tsv```
* ```sct.test.tsv```

unique_id |	label |	text_a |	text_b |	vs_sent1 |	vs_sent2 |	vs_sent3 |	vs_sent4 |	vs_sent5 | cs_dist
--- | ---- | ---- |---- |---- |---- |---- |---- |---- | ---
c9e0ad...5 |	0 |	 The football team ... scored. |	The crowd ... loudly. |	[-0.1027  0.167   0.833   0.    ] |	[0.765 0.    0.577 0.423] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] | [1. 1. 1. 1.]

Preprocessed data is obtained by running ```preprocess_data.py``` with initial data files stored in ```./data/```.

#### Experiments

* BERT AC TS III
```
python bert_sct.py 
--data_dir=./data_pp_test/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
--bert_trainable=True

--num_train_epochs=3.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

* BERT AC TS III + MLM
```
python bert_sct.py 
--data_dir=./data_pp_test/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--init_checkpoint=./bert_masked_lm_pp/model.ckpt 
--bert_dir=./bert/
--bert_trainable=True

--num_train_epochs=3.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

### SCT relative classification ```bert_sct_v2.py``` or ```bert_sct_v2.ipynb```  (Colab)

#### Data Format

Data format in ```./data_pp/``` and ```./data_pp_test/```:
* ```sct_v2.train.tsv```
* ```sct_v2.validation.tsv```
* ```sct_v2.test.tsv```

unique_id |	label |	text_a |	text_b _pos | text_b _neg |	vs_sent1 |	vs_sent2 |	vs_sent3 |	vs_sent4 |	vs_sent5_pos | vs_sent5_neg | cs_dist_pos | cs_dist_neg
--- | ---- | ---- |---- |---- |---- |---- |---- |---- | --- | --- | --- | ---
c...5 |	1 |	 The ... |	The ... | The ... |	[-0.1027  0.167   0.833   0.    ] |	[0.765 0.    0.577 0.423] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] | [1. 1. 1. 1.] | [-1. -1. -1. -1.]

Preprocessed data is obtained by running ```preprocess_data_v2.py``` with initial data files stored in ```./data/```.

#### Experiments

* BERT RC TS III
```
python bert_sct_v2.py 
--data_dir=./data_pp_test/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
--bert_trainable=True

--num_train_epochs=3.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

* BERT RC TS IV
```
python bert_sct_v2.py 
--data_dir=./data_pp/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
--bert_trainable=True

--num_train_epochs=3.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

* BERT RC TS I and II

Same as for BERT TS IV. However, beforehand ```preprocess_data_v2.py``` needs to be re-run, training set selection is done by uncommenting appropriate line.

```
# TS I
# df_train = df_train

# TS II
# df_train = pd.concat([df_train, df_train_val, df_train_val, df_train_val], axis=0)

# TS IV
df_train = df_train_val
```

```
python bert_sct_v2.py 
--data_dir=./data_pp/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
--bert_trainable=True

--num_train_epochs=1.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

#### Additional

* Sentiment

```
python index_1.py
--data_dir=./data/
--sentiment_results_dir=./data/
--results_file=results.txt
```

* Common Sense

```
python bert_sct_v2.py 
--data_dir=./data_pp_test/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 

--bert_trainable=False
--commonsense_only=True

--num_train_epochs=3.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

* Combination BERT RC TS III

```
python bert_sct_v2.py 
--data_dir=./data_pp_test/ 
--output_dir="./output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
--bert_trainable=True

--combination=True

--num_train_epochs=3.0
--batch_size=8
--max_seq_length=400
--learning_rate=0.00002
--warmup_proportion=0.1
```

#### Local directory hierarchy.

![Local directory hierarchy.](./docs/dir.jpg?raw=false "Local directory hierarchy.")
