from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_helpers
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
import tensorflow as tf
import pandas as pd
import re
import itertools
import math
import traceback
import gensim, logging

tf.flags.DEFINE_integer("distance_dim", 5, "Dimension of position vector")
tf.flags.DEFINE_integer("embedding_size", 50, "Dimension of word embedding")
tf.flags.DEFINE_integer("n1", 200, "Hidden layer1")
tf.flags.DEFINE_integer("n2", 100, "Hidden layer2")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_float("lr", 0.0001, "Learning rate")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("window_size", 3, "n-gram")
tf.flags.DEFINE_integer("sequence_length", 204, "max tokens b/w entities")
tf.flags.DEFINE_integer("K", 4, "K-fold cross validation")
tf.flags.DEFINE_float("early_threshold", 0.5, "Threshold to stop the training")
FLAGS = tf.flags.FLAGS
tokenizer = TweetTokenizer()

invalid_word = "UNK"

'''By default returns UNK if input given is empty'''

model = gensim.models.Word2Vec.load("~/Desktop/Relation_Extraction/model")


def word2vec(word):
    return model[word]


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


def get_sentences(text):
    indices = []
    for start, end in PunktSentenceTokenizer().span_tokenize(text):
        indices.append((start, end))
    return indices


def get_tokens(words):
    valid_words = []
    for word in words:
        if data_helpers.is_word(word) and word in model.vocab:
            valid_words.append(word)
    return valid_words


def get_left_word(message, start):
    i = start - 1
    is_space = 0
    str = ""
    while i > -1:
        if message[i].isspace() and is_space == 1 and str.strip():
            break
        if message[i].isspace() and is_space == 1 and not data_helpers.is_word(str):
            is_space = 0
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
        if message[i].isspace() and is_space == 1 and str.strip():
            break
        if message[i].isspace() and is_space == 1 and not data_helpers.is_word(str):
            is_space = 0
        if message[i].isspace():
            is_space = 1
        str += message[i]
        i += 1
    return tokenizer.tokenize(str)


# def w2v(word):
#     if word != "UNK":
#         word = word.lower()
#     index = data_helpers.word2id(word)
#     if index == -1:
#         raise ValueError("{} doesn't exist in the vocablury.".format(word))
#     else:
#         return word_vector[0][index]


count = 100


def lexical_level_features(df):
    for index, row in df.iterrows():
        try:
            # if index >= count:
            #     break
            print("======================================")
            print(index)
            message = row['Message'].lower()
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
            sent = get_sentences(message)
            start1, start2 = start[0], end[0]
            end1, end2 = start[1], end[1]
            beg = -1
            for l, r in sent:
                if (start1 >= l and start1 <= r) or (end1 >= l and end1 <= r) or (start2 >= l and start2 <= r) or (
                                end2 >= l and end2 <= r):
                    if beg == -1:
                        beg = l
                    fin = r

            print(message[beg:fin])
            entity1, entity2 = message[start1:end1], message[start2:end2]
            l1 = [get_legit_word([word], 1) for word in tokenizer.tokenize(entity1)]
            l2 = [get_legit_word([word], 1) for word in tokenizer.tokenize(entity2)]

            # TODO add PCA for phrases
            temp = np.zeros(FLAGS.embedding_size)
            valid_words = 0
            print(entity1)
            print(l1)
            for word in l1:
                if word != "UNK" and data_helpers.is_word(word) and word in model.vocab:
                    valid_words += 1
                    temp = np.add(temp, word2vec(word))
            if valid_words == 0:
                continue
            l1 = temp / float(valid_words)
            temp = np.zeros(FLAGS.embedding_size)
            valid_words = 0
            print(entity2)
            print(l2)
            for word in l2:
                if word != "UNK" and data_helpers.is_word(word) and word in model.vocab:
                    valid_words += 1
                    temp = np.add(temp, word2vec(word))
            if valid_words == 0:
                continue
            lword1 = lword2 = rword1 = rword2 = np.zeros(50)
            l2 = temp / float(valid_words)
            if get_legit_word(get_left_word(message, start1), 0) in model.vocab:
                lword1 = word2vec(get_legit_word(get_left_word(message, start1), 0))
            if get_legit_word(get_left_word(message, start2), 0) in model.vocab:
                lword2 = word2vec(get_legit_word(get_left_word(message, start2), 0))
            if get_legit_word(get_right_word(message, end1), 1) in model.vocab:
                rword1 = word2vec(get_legit_word(get_right_word(message, end1), 1))
            if get_legit_word(get_right_word(message, end2), 1) in model.vocab:
                rword2 = word2vec(get_legit_word(get_right_word(message, end2), 1))
            # l3 = np.divide(np.add(lword1, rword1), 2.0)
            # l4 = np.divide(np.add(lword2, rword2), 2.0)
            print(get_legit_word(get_left_word(message, start1), 0), get_legit_word(get_left_word(message, start2), 0))
            print(get_legit_word(get_right_word(message, end1), 1), get_legit_word(get_right_word(message, end2), 1))

            # tokens in between
            l_tokens = []
            r_tokens = []
            if beg != -1:
                l_tokens = get_tokens(tokenizer.tokenize(message[beg:start1]))
            if fin != -1:
                r_tokens = get_tokens(tokenizer.tokenize(message[end2:fin]))
            in_tokens = get_tokens(tokenizer.tokenize(message[end1:start2]))
            print(l_tokens, in_tokens, r_tokens)

            tot_tokens = len(l_tokens) + len(in_tokens) + len(r_tokens) + 2
            while tot_tokens < FLAGS.sequence_length:
                r_tokens.append("UNK")
                tot_tokens += 1
            # left tokens
            l_matrix = []
            l_len = len(l_tokens)
            r_len = len(r_tokens)
            m_len = len(in_tokens)
            for idx, token in enumerate(l_tokens):
                word_vec, pv1, pv2 = word2vec(token), pos_vec[pivot + (idx - l_len)], pos_vec[
                    pivot + (idx - l_len - 1 - m_len)]
                l_matrix.append([word_vec, pv1, pv2])

            # middle tokens
            in_matrix = []
            for idx, token in enumerate(in_tokens):
                word_vec, pv1, pv2 = word2vec(token), pos_vec[idx + 1], pos_vec[idx - m_len + pivot]
                in_matrix.append([word_vec, pv1, pv2])

            # right tokens
            r_matrix = []
            for idx, token in enumerate(r_tokens):
                if token == "UNK":
                    word_vec, pv1, pv2 = extra_emb, pos_vec[idx + m_len + 2], pos_vec[idx + 1]
                    r_matrix.append([word_vec, pv1, pv2])
                else:
                    word_vec, pv1, pv2 = word2vec(token), pos_vec[idx + m_len + 2], pos_vec[idx + 1]
                    r_matrix.append([word_vec, pv1, pv2])

            tri_gram = []
            llen = len(l_matrix)
            mlen = len(in_matrix)
            rlen = len(r_matrix)
            dist = llen + 1
            if llen > 0:
                if llen > 1:
                    tri_gram.append(
                        np.hstack((beg_emb, l_matrix[0][0], l_matrix[1][0], l_matrix[0][1], l_matrix[0][2])))
                    for i in range(1, len(l_matrix) - 1):
                        tri_gram.append(
                            np.hstack((l_matrix[i - 1][0], l_matrix[i][0], l_matrix[i + 1][0], l_matrix[i][1],
                                       l_matrix[i][2])))
                    tri_gram.append(np.hstack((l_matrix[llen - 2][0], l_matrix[llen - 1][0], l1, l_matrix[llen - 1][1],
                                               l_matrix[llen - 2][2])))
                else:
                    tri_gram.append(
                        np.hstack((beg_emb, l_matrix[0][0], l1, l_matrix[0][1], l_matrix[0][2])))
                if mlen > 0:
                    tri_gram.append(
                        np.hstack((l_matrix[llen - 1][0], l1, in_matrix[0][0], pos_vec[0], pos_vec[pivot - dist])))
                else:
                    tri_gram.append(np.hstack((l_matrix[llen - 1][0], l1, l2, pos_vec[0], pos_vec[pivot - dist])))
            else:
                if mlen > 0:
                    tri_gram.append(np.hstack((beg_emb, l1, in_matrix[0][0], pos_vec[0], pos_vec[pivot - dist])))
                else:
                    tri_gram.append(np.hstack((beg_emb, l1, l2, pos_vec[0], pos_vec[pivot - dist])))

            if mlen > 0:
                if mlen > 1:
                    tri_gram.append(np.hstack((l1, in_matrix[0][0], in_matrix[1][0], in_matrix[0][1], in_matrix[0][2])))
                    for i in range(1, len(in_matrix) - 1):
                        tri_gram.append(np.hstack((in_matrix[i - 1][0], in_matrix[i][0], in_matrix[i + 1][0],
                                                   in_matrix[i][1], in_matrix[i][2])))
                    tri_gram.append(np.hstack((in_matrix[mlen - 2][0], in_matrix[mlen - 1][0], l2,
                                               in_matrix[mlen - 1][1], in_matrix[mlen - 2][2])))
                else:
                    tri_gram.append(np.hstack((l1, in_matrix[0][0], l2, in_matrix[0][1], in_matrix[0][2])))
                if rlen > 0:
                    tri_gram.append(np.hstack((in_matrix[mlen - 1][0], l2, r_matrix[0][0], pos_vec[dist], pos_vec[0])))
                else:
                    tri_gram.append(np.hstack((in_matrix[mlen - 1][0], l2, end_emb, pos_vec[dist], pos_vec[0])))
            else:
                if rlen > 0:
                    tri_gram.append(np.hstack((l1, l2, r_matrix[0][0], pos_vec[dist], pos_vec[0])))
                else:
                    tri_gram.append(np.hstack((l1, l2, end_emb, pos_vec[dist], pos_vec[0])))
            if rlen > 0:
                if rlen > 1:
                    tri_gram.append(np.hstack((l2, r_matrix[0][0], r_matrix[1][0], r_matrix[0][1], r_matrix[0][2])))
                    for i in range(1, len(r_matrix) - 1):
                        tri_gram.append(np.hstack(
                            (r_matrix[i - 1][0], r_matrix[i][0], r_matrix[i + 1][0], r_matrix[i][1], r_matrix[i][2])))
                    tri_gram.append(np.hstack((r_matrix[rlen - 2][0], r_matrix[rlen - 1][0], end_emb,
                                               r_matrix[rlen - 1][1], r_matrix[rlen - 2][2])))

                else:
                    tri_gram.append(np.hstack((l2, r_matrix[0][0], end_emb, r_matrix[0][1], r_matrix[0][2])))
            # tri_gram.append(np.hstack((l1, in_matrix[0][0], in_matrix[1][0], in_matrix[0][1], in_matrix[0][2])))
            #
            # for idx in range(1, mlen - 1):
            #     tri_gram.append(
            #         np.hstack((in_matrix[idx - 1][0], in_matrix[idx][0], in_matrix[idx + 1][0], in_matrix[idx][1], in_matrix[idx][2])))
            # tri_gram.append(
            #     np.hstack((in_matrix[mlen - 2][0], in_matrix[mlen - 1][0], l2, in_matrix[mlen - 1][1], in_matrix[mlen - 1][2])))
            # tri_gram.append(np.hstack((in_matrix[mlen - 1][0], l2, end_emb, pos_vec_entities[2], pos_vec_entities[3])))
            print("======================================")
            # lf = np.vstack((l1, l2, l3, l4))
            relation = row['relType']
            print(np.asarray(tri_gram).shape)
            if relation == "valid":
                y = [0.0, 1.0]
            else:
                y = [1.0, 0.0]
            yield np.asarray((np.asarray(tri_gram), np.asarray(y)))
        except Exception as e:
            traceback.print_exc()


def get_batches():
    print("Loading train data...")
    lexical_features = lexical_level_features(df)
    batch_iterator = data_helpers.batch_iter(lexical_features, FLAGS.batch_size, FLAGS.num_epochs)
    return batch_iterator


def get_batches_test():
    print("Loading test data...")
    df = data_helpers.read_data("/home/sahil/ML-bucket/test.csv")
    lexical_features = lexical_level_features(df)
    batch_iterator = data_helpers.batch_iter(lexical_features, FLAGS.batch_size, 1, shuffle=False)
    return batch_iterator

def get_validation_data():
    df = data_helpers.read_data("/home/sahil/ML-bucket/data/validation.csv")
    lexical_features = lexical_level_features(df)
    X_val = list()
    Y_val = list()
    for iter in lexical_features:
        X_val.append(iter[0])
        Y_val.append(iter[1])
    return np.asarray(X_val), np.asarray(Y_val)

df = data_helpers.read_data()

np.random.seed(42)
pivot = 2 * FLAGS.sequence_length + 1
pos_vec = np.random.uniform(-1, 1, (pivot + 1, FLAGS.distance_dim))
# pos_vec_entities = np.random.uniform(-1, 1, (4, FLAGS.distance_dim))

# beginning and end of sentence embeddings
beg_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)
end_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)
extra_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)

# sequence_length = 0
# ain = ""
'''Find the max length b/w entities'''
# for index, row in df.iterrows():
#     message = row['Message']
#     if not message:
#         continue
#     if row['drug-offset-start'] < row['sideEffect-offset-start']:
#         start = (row['drug-offset-start'], row['drug-offset-end'])
#     else:
#         start = (row['sideEffect-offset-start'], row['sideEffect-offset-end'])
#
#     if row['drug-offset-end'] > row['sideEffect-offset-end']:
#         end = (row['drug-offset-start'], row['drug-offset-end'])
#     else:
#         end = (row['sideEffect-offset-start'], row['sideEffect-offset-end'])
#
#     start1, start2 = start[0], end[0]
#     end1, end2 = start[1], end[1]
#     str = ""
#     sent = get_sentences(message)
#     beg = -1
#     for l, r in sent:
#         if (start1 >= l and start1 <= r) or (end1 >= l and end1 <= r) or (start2 >= l and start2 <= r) or (
#                         end2 >= l and end2 <= r):
#             if beg == -1:
#                 beg = l
#             fin = r
#             str += message[l:r]
#     if beg != -1:
#         l_tokens = get_tokens(tokenizer.tokenize(message[beg:start1]))
#     if fin != -1:
#         r_tokens = get_tokens(tokenizer.tokenize(message[end2:fin]))
#     in_tokens = get_tokens(tokenizer.tokenize(message[end1:start2]))
#     tot_len = len(l_tokens) + len(in_tokens) + len(r_tokens)
#     entity1 = message[start1:end1]
#     entity2 = message[start2:end2]
#     if tot_len > sequence_length:
#         ain = (tot_len, entity1, entity2, message[beg:fin])
#     sequence_length = max(sequence_length, tot_len)
#
# print(sequence_length)
# print(ain)

def hack():
    df = pd.read_csv("/home/sahil/Downloads/test.csv")
    for index, row in df.iterrows():
        arr = [[float(row['x1']), float(row['x2']), float(row['x3'])]]
        y = float(row['y'])
        if y == 0.0:
            y = [1.0, 0.0]
        else:
            y = [0.1, 1.0]
        yield np.asarray((np.asarray(arr), np.asarray(y)))

def fun():
    r = hack()
    s = data_helpers.batch_iter(r, 64, 1)
    return s
