from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_helpers
import numpy as np
from nltk.tokenize import TweetTokenizer
import tensorflow as tf
import re

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
tf.flags.DEFINE_integer("sequence_length", 31, "max tokens b/w entities")
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


def get_tokens(words):
    valid_words = []
    for word in words:
        if data_helpers.is_word(word):
            valid_words.append(word)
    while len(valid_words) < FLAGS.sequence_length:
        valid_words.append(invalid_word)
    return valid_words


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
            # if index >= count:
            #     break
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
            if valid_words == 0:
                continue
            l1 = temp / float(valid_words)
            temp = np.zeros(FLAGS.embedding_size)
            valid_words = 0
            print(entity2)
            print(l2)
            for word in l2:
                if word != "UNK" and data_helpers.is_word(word):
                    valid_words += 1
                    temp = np.add(temp, w2v(word))
            if valid_words == 0:
                continue
            l2 = temp / float(valid_words)
            lword1 = w2v(get_legit_word(get_left_word(message, start1), 0))
            lword2 = w2v(get_legit_word(get_left_word(message, start2), 0))
            rword1 = w2v(get_legit_word(get_right_word(message, end1), 1))
            rword2 = w2v(get_legit_word(get_right_word(message, end2), 1))
            l3 = np.divide(np.add(lword1, rword1), 2.0)
            l4 = np.divide(np.add(lword2, rword2), 2.0)
            print(get_legit_word(get_left_word(message, start1), 0), get_legit_word(get_left_word(message, start2), 0))
            print(get_legit_word(get_right_word(message, end1), 1), get_legit_word(get_right_word(message, end2), 1))

            # tokens in between
            in_tokens = get_tokens(tokenizer.tokenize(message[end1:start2]))
            matrix = []
            for idx, token in enumerate(in_tokens):
                word_vec, pv1, pv2 = w2v(token), pos_vec[idx], pos_vec[pivot - idx]
                matrix.append([word_vec, pv1, pv2])
            tri_gram = []
            mlen = len(matrix)
            tri_gram.append(np.hstack((beg_emb, l1, matrix[0][0], pos_vec_entities[0], pos_vec_entities[1])))
            tri_gram.append(np.hstack((l1, matrix[0][0], matrix[1][0], matrix[0][1], matrix[0][2])))
            for idx in range(1, mlen - 1):
                tri_gram.append(np.hstack((matrix[idx - 1][0], matrix[idx][0], matrix[idx + 1][0], matrix[idx][1], matrix[idx][2])))
            tri_gram.append(np.hstack((matrix[mlen - 2][0], matrix[mlen - 1][0], l2, matrix[mlen - 1][1], matrix[mlen - 1][2])))
            tri_gram.append(np.hstack((matrix[mlen - 1][0], l2, end_emb, pos_vec_entities[2], pos_vec_entities[3])))
            print("======================================")
            lf = np.vstack((l1, l2, l3, l4))
            relation = row['relType']
            if relation == "valid":
                y = [0.0, 1.0]
            else:
                y = [1.0, 0.0]
            yield np.asarray((np.asarray(tri_gram), np.asarray(y)))
        except Exception as e:
            print(e)


def get_batches():
    lexical_features = lexical_level_features(df)
    batch_iterator = data_helpers.batch_iter(lexical_features, FLAGS.batch_size, FLAGS.num_epochs)
    return batch_iterator


df = data_helpers.read_data()
word_vector = list(data_helpers.get_word_vector())

np.random.seed(42)
pivot = 2 * FLAGS.sequence_length + 1
pos_vec = np.random.uniform(-1, 1, (pivot + 1, FLAGS.distance_dim))
pos_vec_entities = np.random.uniform(-1, 1, (4, FLAGS.distance_dim))

# beginning and end of sentence embeddings
beg_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)
end_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)
count = 1000
'''Find the max length b/w entities'''
# for index, row in df.iterrows():
#     message = row['Message']
#     if not message:
#         continueVocabularyProcessor
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
#     l = get_tokens(tokenizer.tokenize(message[end1: start2]))
#     sequence_length = max(sequence_length, len(l))

# flag = 0
# i = 0
# for batch in enumerate(get_batches()):
#     X_train, y_train = zip(*batch)
#     for i in X_train:
#         for j in i:
#             for k in j:
#                 if np.isnan(k):
#                     flag = 1
#                 if flag == 1:
#                     break
#             if flag == 1:
#                 break
#         if flag == 1:
#             break
#     print(i, flag)
#     i += 1
#     if flag == 1:
#         break
