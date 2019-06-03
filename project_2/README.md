## Natural Language Understanding
## Project 2

### Data Format

Data format in ```./data_pp/```:
* ```sct.train.tsv```
* ```sct.validation.tsv```
* ```sct.test.tsv```

unique_id |	label |	text_a |	text_b |	vs_sent1 |	vs_sent2 |	vs_sent3 |	vs_sent4 |	vs_sent5 | cs_dist
--- | ---- | ---- |---- |---- |---- |---- |---- |---- | ---
c9e0ad...5 |	0 |	 The football team ... scored. |	The crowd ... loudly. |	[-0.1027  0.167   0.833   0.    ] |	[0.765 0.    0.577 0.423] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] |	[0. 0. 1. 0.] | [1. 1. 1. 1.]

Preprocessed data is obtained by running ```preprocess_data.py``` with initial data files stored in ```./data/```.


### Run in training, evaluation and prediction mode

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

Local directory hierarchy.

![Local directory hierarchy.](./docs/dir.PNG?raw=false "Local directory hierarchy.")
