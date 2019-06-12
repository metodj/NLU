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


### Experiments
#### Experiment A (FINAL)
* Parameters
    * BERT initialization: ```https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1```
    * Data: ```data_pp_test/``` 
  
    ``` df_train_test = pd.concat([df_train_tmp.iloc[0:data_train_val.shape[0]], df_train.iloc[-data_train_val.shape[0]:]], axis=0)``` 
    
    * Classification: Relative
    * ```num_epochs = 3.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: NO
    * Common sense: NO
    * Main: ```bert_sct_v2.py```
* Performance
    * eval_accuracy = 0.8809013
    * eval_loss = 0.46227175
    * global_step = 1403
    * loss = 0.46227175
    * precision = 1.0
    * recall = 0.8809013
* Output
    * Eval: ```11_18-27eval_results.txt```
    * Test: ```11_18-27test_results.tsv```
    
#### Experiment B (FINAL)
* Parameters
    * BERT initialization: ```https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1```
    * Data: ```data_pp/``` 
  
    ``` df_train = df_train_val``` 
    
    * Classification: Relative
    * ```num_epochs = 3.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: NO
    * Common sense: NO
    * Main: ```bert_sct_v2.py```
* Performance
    * eval_accuracy = 0.86212444
    * eval_loss = 0.51910084
    * global_step = 701
    * loss = 0.51910084
    * precision = 1.0
    * recall = 0.86212444
* Output
    * Eval: ```09_09-39eval_results.txt```
    * Test: ```09_09-39test_results.tsv```
#### Experiment C (FINAL)
* Parameters
    * BERT initialization: ```https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1```
    * Data: ```data_pp_test/```     
    * Classification: Absolute
    * ```num_epochs = 3.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: NO
    * Common sense: NO
    * Main: ```bert_sct.py```
* Performance
    * eval_accuracy = 0.8581371
    * eval_loss = 0.99690187
    * f1 = 1.0
    * global_step = 1403
    * loss = 0.99690187
* Output
    * Eval: ```09_10-20eval_results.txt```
    * Test: ```09_10-20test_results.tsv```
    
#### Experiment D (FINAL)
* Parameters
    * BERT initialization: ```./bert_masked_lm_pp```
    * Data: ```data_pp_test/```     
    * Classification: Absolute
    * ```num_epochs = 3.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: NO
    * Common sense: NO
    * Main: ```bert_sct.py```
* Performance
    * eval_accuracy = 0.8506424
    * eval_loss = 1.0681794
    * f1 = 1.0
    * global_step = 1403
    * loss = 1.0681794
* Output
    * Eval: ```09_11-18eval_results.txt```
    * Test: ```09_11-18test_results.tsv```
    
#### Experiment E (FINAL)
* Parameters
    * BERT initialization: ```https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1```
    * Data: ```data_pp/``` TS1 Full     
    * Classification: Absolute
    * ```num_epochs = 1.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: NO
    * Common sense: NO
    * Main: ```bert_sct_v2.py```
* Performance
    * eval_accuracy = 0.7988197
    * eval_loss = 0.51480865
    * global_step = 11721
    * loss = 0.51480865
    * precision = 1.0
    * recall = 0.7988197
* Output
    * Eval: ```09_22-09eval_results.txt```
    * Test: ```09_22-09test_results.tsv```
    
#### Experiment F (FINAL)
* Parameters
    * BERT initialization: ```https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1```
    * Data: ```data_pp/``` TS1 Training only     
    * Classification: Absolute
    * ```num_epochs = 1.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: NO
    * Common sense: NO
    * Main: ```bert_sct_v2.py```
* Performance
    * eval_accuracy = 0.5767167
    * eval_loss = 0.6829421
    * global_step = 0
    * loss = 0.6829421
    * precision = 1.0
    * recall = 0.5767167
* Output
    * Eval: ```10_04-38eval_results.txt```
    * Test: ```10_04-38test_results.tsv``` 
#### Experiment G (FINAL)
* Parameters
    * BERT initialization: ```https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1```
    * Data: ```data_pp_test/``` TS3     
    * Classification: Relative
    * ```num_epochs = 3.0, batch_size = 8, max_seq_length = 400, learning_rate = 0.5, warmup_proportion = 0.1```
    * Sentiment: YES
    * Common sense: YES
    * Main: ```bert_sct_v2.py```
* Performance
    * eval_accuracy = 0.87607294
    * eval_loss = 1.811894
    * global_step = 1403
    * loss = 1.811894
    * precision = 1.0
    * recall = 0.87607294
* Output
    * Eval: ```11_15-49eval_results.txt```
    * Test: ```11_15-49test_results.tsv``` 