import numpy as np
import pandas as pd
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv




# DICT TO NUMPY ARRAY

def dict_to_nparray (d):
    a = np.array([ val for val in d.values()])
    return a

def dict_to_list(d):
    return [val for val in d.values()]

# REMOVE EVENTUAL FINAL DOT IN SENTENCES

def remove_eventual_final_dot (sentence):
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    return sentence


class Story:
    def __init__(self,line_story):
        self.id = line_story[0]
        self.title =remove_eventual_final_dot(line_story[1])
        self.sentence1 = remove_eventual_final_dot(line_story[2])
        self.sentence2 = remove_eventual_final_dot(line_story[3])
        self.sentence3 = remove_eventual_final_dot(line_story[4])
        self.sentence4 = remove_eventual_final_dot(line_story[5])
        self.sentence5 = remove_eventual_final_dot(line_story[6])
    def to_print(self):
        return self.title +'\n'+ self.sentence1 +'\n'+ self.sentence2 +'\n'+ self.sentence3 +'\n'+ self.sentence4 +'\n'+ self.sentence5

'''

WRONG

# READ STORIES : rememeber to drop first element of the list in case it is a legenda
def read_stories(file):

    stories = []

    with open(file, "r") as f:
        for line in f.readlines():
            line_story = line.strip().split(",")
            stories.append(Story(line_story))
    return stories
'''


def read_stories(file):

    print("executing read stories")

    stories = []

    with open(file, "r") as f:
        csv_reader = csv.reader(f,delimiter=',')
        for line in csv_reader:
            stories.append(Story(line))

    # pop first element: legenda        REMEBEMBER TO REMOVE IT FOR VALIDATION
    stories.pop(0)

    return stories



def read_stories_tf (file):
    stories = []
    list_stories = read_stories(file)

    for story in list_stories:
        stories.append([story.sentence1, story.sentence2, story.sentence3, story.sentence4, story.sentence5])

    return stories




def read_and_embed_stories(file):   # IN THE EMBEDDING WE HAVE [NEG, NEU, POS]

    print('executing read_and_embed_stories')

    #create analyzer
    analyzer = SentimentIntensityAnalyzer()  #the analyzer is what does the embedding


    embedded_stories = []
    stories = read_stories(file)

    for story in stories:

        s1_embedded = dict_to_list(analyzer.polarity_scores(story.sentence1))
        s1_embedded.pop(3)
        s2_embedded = dict_to_list(analyzer.polarity_scores(story.sentence2))
        s2_embedded.pop(3)
        s3_embedded = dict_to_list(analyzer.polarity_scores(story.sentence3))
        s3_embedded.pop(3)
        s4_embedded = dict_to_list(analyzer.polarity_scores(story.sentence4))
        s4_embedded.pop(3)
        s5_embedded = dict_to_list(analyzer.polarity_scores(story.sentence5))
        s5_embedded.pop(3)

        embedded_story = [s1_embedded, s2_embedded, s3_embedded, s4_embedded, s5_embedded]


        embedded_stories.append(embedded_story)



    return embedded_stories




def read_embed_createtensor_from_file_stories(file):

    print("executing read_embed_createtensor_from_file_stories")

    stories_tens = tf.convert_to_tensor(read_and_embed_stories(file), dtype=tf.float32, name='stories_tens')

    return stories_tens




def embedding_io_map (emb_story):
    input_emb = emb_story[:-1]
    output_emb = emb_story[-1]
    return input_emb, output_emb


def cosine_similarity(predictions, labels):

    pred_norm = tf.math.l2_normalize(predictions, axis=1, name='normalize_predictions')
    lab_norm =  tf.math.l2_normalize(labels, axis=1, name='normalize_labels')

    cos_sim = tf.reduce_sum(tf.multiply(pred_norm, lab_norm, name='cos_sim_multiplication'), axis=1)


    return cos_sim, tf.shape(cos_sim)

'''
#main

train_file = "data/train_stories.csv"

stories = read_and_embed_stories(train_file)

print("embedded stories: ")
for _ in range(5):
    print(stories[_])


stories_tens = read_embed_createtensor_from_file_stories(train_file)

print(stories_tens)

dataset = tf.data.Dataset.from_tensor_slices(read_embed_createtensor_from_file_stories(train_file)).map(embedding_io_map)

print(dataset.output_shapes, dataset.output_types)


iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

iter_initializer = iter.make_initializer(dataset)

elem = iter.get_next()

# session

print("dataset elements:")
with tf.Session() as sess:
    sess.run(iter_initializer)
    for _ in range(10):
        print(sess.run(elem), '\n')


'''



