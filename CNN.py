from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_helpers
import numpy as np
import itertools
from nltk.tokenize import TweetTokenizer
import tensorflow as tf
import os
import re
import time

tf.flags.DEFINE_integer("distance_dim", 5, "Dimension of position vector")
tf.flags.DEFINE_integer("embedding_size", 50, "Dimension of word embedding")
tf.flags.DEFINE_integer("n1", 200, "Hidden layer1")
tf.flags.DEFINE_integer("n2", 100, "Hidden layer2")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_float("lr", 0.01, "Learning rate")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 10, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
tokenizer = TweetTokenizer()

invalid_word = "UNK"
def get_legit_word(str, flag):
    if flag == 0:
        for word in reversed(str):
            if word in [".", "!"]:
                return invalid_word
            if data_helpers.is_word(word):
                return word
        return invalid_word

    if flag == 1:
        for word in str:
            if word in [".", "!"]:
                return invalid_word
            if data_helpers.is_word(word):
                return word
        return invalid_word


def get_left_word(message, start):
    i = start - 1
    is_space = 0
    str = ""
    while i > -1:
        if message[i].isspace() and is_space == 1 and not re.search(r'\A[\w]+\Z', str):
            is_space = 0
        else:
            if message[i].isspace() and is_space == 1 and str.strip():
                break
        if message[i].isspace():
            is_space = 1
        str += message[i]
        i -= 1
    str = str[::-1]
    return tokenizer.tokenize(str)

def get_right_word(message, start):
    i = start
    is_space = 0
    str = ""
    while i < len(message):
        if message[i].isspace() and is_space == 1 and not re.search(r'\A[\w]+\Z', str):
            is_space = 0
        else:
            if message[i].isspace() and is_space == 1 and str.strip():
                break
        if message[i].isspace():
            is_space = 1
        str += message[i]
        i += 1
    return tokenizer.tokenize(str)

def w2v(word):
    if word != "UNK":
        word = word.lower()
    index = data_helpers.word2id(word)
    if index == -1:
        raise ValueError("{} doesn't exist in the vocablury.".format(word))
    else:
        return word_vector[0][index]

def lexical_level_features(df):
        for index, row in df.iterrows():
            try:
                if index >= count:
                    break
                print("======================================")
                print(index)
                message = row['Message']
                if not message:
                    continue
                if row['drug-offset-start'] < row['sideEffect-offset-start']:
                    start = (row['drug-offset-start'], row['drug-offset-end'])
                else:
                    start = (row['sideEffect-offset-start'], row['sideEffect-offset-end'])

                if row['drug-offset-end'] > row['sideEffect-offset-end']:
                    end = (row['drug-offset-start'], row['drug-offset-end'])
                else:
                    end = (row['sideEffect-offset-start'], row['sideEffect-offset-end'])
                print(message)
                start1, start2 = start[0], end[0]
                end1, end2 = start[1], end[1]
                entity1, entity2 = message[start1:end1], message[start2:end2]
                l1 = [get_legit_word([word], 1) for word in tokenizer.tokenize(entity1)]
                l2 = [get_legit_word([word], 1) for word in tokenizer.tokenize(entity2)]

                # TODO add PCA for phrases
                temp = np.zeros(FLAGS.embedding_size)
                valid_words = 0
                print(entity1)
                print(l1)
                for word in l1:
                    if word != "UNK" and data_helpers.is_word(word):
                        valid_words += 1
                        temp = np.add(temp, w2v(word))
                l1 = temp / float(valid_words)
                temp = np.zeros(FLAGS.embedding_size)
                valid_words = 0
                print(entity2)
                print(l2)
                for word in l2:
                    if word != "UNK" and data_helpers.is_word(word):
                        valid_words += 1
                        temp = np.add(temp, w2v(word))
                l2 = temp / float(valid_words)
                lword1 = w2v(get_legit_word(get_left_word(message, start1), 0))
                lword2 = w2v(get_legit_word(get_left_word(message, start2), 0))
                rword1 = w2v(get_legit_word(get_right_word(message, end1), 1))
                rword2 = w2v(get_legit_word(get_right_word(message, end2), 1))
                l3 = np.divide(np.add(lword1, rword1), 2.0)
                l4 = np.divide(np.add(lword2, rword2), 2.0)
                print(get_legit_word(get_left_word(message, start1), 0), get_legit_word(get_left_word(message, start2), 0))
                print(get_legit_word(get_right_word(message, end1), 1), get_legit_word(get_right_word(message, end2), 1))
                print("======================================")
                lf = np.vstack((l1, l2, l3, l4))
                relation = row['relType']
                if relation == "valid":
                    y = [0, 1]
                else:
                    y = [1, 0]
                yield np.asarray((lf, y))
            except Exception as e: print(e)

def get_position_vectors(distance):
    pass

def build_sentence_model():
    pass

def get_batches():
    time_start = time.time()
    lexical_features = lexical_level_features(df)
    batch_iterator = data_helpers.batch_iter(lexical_features, FLAGS.batch_size, FLAGS.num_epochs)
    return batch_iterator

df = data_helpers.read_data()
count = 500
word_vector = list(data_helpers.get_word_vector())
