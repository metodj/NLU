## Natural Language Understanding
## Project 2


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

BERT AC TS III + MLM
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

### SCT relative classification ```bert_sct_v2.py/ipynb``` 

#### Data Format

Data format in ```./data_pp/```:
* ```sct_v2.train.tsv```
* ```sct_v2.validation.tsv```
* ```sct_v2.test.tsv```

unique_id |	label |	text_a |	text_b _pos | text_b _neg |	vs_sent1 |	vs_sent2 |	vs_sent3 |	vs_sent4 |	vs_sent5_pos | vs_sent5_neg | cs_dist_pos | cs_dist_neg
--- | ---- | ---- |---- |---- |---- |---- |---- |---- | --- | --- | --- | ---
c...5 |	1 |	 The ... |	The ... | The ... |	[-0.1027  0.167   0.833   0.    ] |	[0.765 0.    0.577 0.423] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] | [1. 1. 1. 1.] | [-1. -1. -1. -1.]

Preprocessed data is obtained by running ```preprocess_data.py``` with initial data files stored in ```./data/```.

#### RUN Mode

Using tensorflow-hub module.
```
python bert_sct.py 
--data_dir=./data_pp/ 
--output_dir="C:\Users\roksi\output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
```

Using checkpoints, change accordingly as before.


Local directory hierarchy.

![Local directory hierarchy.](./docs/dir.PNG?raw=false "Local directory hierarchy.")
