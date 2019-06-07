## Natural Language Understanding
## Project 2



### SCT absolute classification ```bert_sct.py/ipynb``` 

#### Data Format

Data format in ```./data_pp/```:
* ```sct.train.tsv```
* ```sct.validation.tsv```
* ```sct.test.tsv```

unique_id |	label |	text_a |	text_b |	vs_sent1 |	vs_sent2 |	vs_sent3 |	vs_sent4 |	vs_sent5 | cs_dist
--- | ---- | ---- |---- |---- |---- |---- |---- |---- | ---
c9e0ad...5 |	0 |	 The football team ... scored. |	The crowd ... loudly. |	[-0.1027  0.167   0.833   0.    ] |	[0.765 0.    0.577 0.423] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] | [1. 1. 1. 1.]

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

Using checkpoint, downloaded from: https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip.
```
python bert_sct.py 
--data_dir=./data_pp/ 
--output_dir="C:\Users\roksi\output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--init_checkpoint=./bert/bert_model.ckpt 
--bert_dir=./bert/
```

Using checkpoint, pretrained on training set with masked language model and next sentence prediction loss.
```
python bert_sct.py 
--data_dir=./data_pp/ 
--output_dir="C:\Users\roksi\output" 
--do_train=True 
--do_eval=True 
--do_predict=True 
--init_checkpoint=./bert_masked_lm_pp/model.ckpt 
--bert_dir=./bert/ 
--num_train_epochs=0.01
```

### SCT relative classification ```bert_sct.py/ipynb``` 

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


### Experiments

#### Experiment A

* Parameters
    * BERT initialization: https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
    * Data: data_pp_test/
    * Classification: Absolute
    * Sentiment: NO
    * Common sense: NO
    * Main: bert_sct.py

* Performance
    * eval_accuracy = 0.8522484
    * eval_loss = 0.79642415
    * f1 = 1.0
    * global_step = 1403
    * loss = 0.79642415
    
* Output
    * Eval: 07_22-11eval_results.txt
    * Test: 07_22-11test_results.tsv

#### Experiment B

* Parameters
    * BERT initialization: ./bert_masked_lm_pp/model.ckpt
    * Data: data_pp_test/
    * Classification: Absolute
    * Sentiment: NO
    * Common sense: NO
    * Main: bert_sct.py
    
* Performance

* Output
    
    

