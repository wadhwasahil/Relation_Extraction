from pyspark import SparkContext, SparkConf
import json
from pyspark.streaming import StreamingContext
import tensorflow as tf
from nltk.tokenize import TweetTokenizer
import data_helpers
from nltk.tokenize.punkt import PunktSentenceTokenizer
import gensim
import numpy as np


sc = SparkContext(appName="HBaseInputFormat")
sc.addPyFile("/home/sahil/Desktop/Relation_Extraction/data_helpers.py")
ssc = StreamingContext(sc, 1)
# Eval Parameters

tf.flags.DEFINE_string("checkpoint_dir", "data/1485336002/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
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
FLAGS._parse_flags()

tokenizer = TweetTokenizer()
invalid_word = "UNK"
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
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

np.random.seed(42)
pivot = 2 * FLAGS.sequence_length + 1
pos_vec = np.random.uniform(-1, 1, (pivot + 1, FLAGS.distance_dim))
# pos_vec_entities = np.random.uniform(-1, 1, (4, FLAGS.distance_dim))

# beginning and end of sentence embeddings
beg_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)
end_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)
extra_emb = np.random.uniform(-1, 1, FLAGS.embedding_size)


def generate_vector(message, start1, end1, start2, end2):
    sent = get_sentences(message)
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
        return None
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
        return None
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
    print(get_legit_word(get_left_word(message, start1), 0),
          get_legit_word(get_left_word(message, start2), 0))
    print(get_legit_word(get_right_word(message, end1), 1),
          get_legit_word(get_right_word(message, end2), 1))

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
        word_vec, pv1, pv2 = word2vec(token), pos_vec[idx + 1], pos_vec[idx - m_len]
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
            tri_gram.append(
                np.hstack((beg_emb, l1, in_matrix[0][0], pos_vec[0], pos_vec[pivot - dist])))
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
    print(np.asarray(tri_gram).shape)
    return np.asarray(tri_gram)


def get_value(row):
    message = row[0]
    rowkey = row[1]
    start1 = row[2]
    end1 = row[3]
    start2 = row[4]
    end2 = row[5]

    if start2 < start1:  # swap if entity2 comes first
        start1, start2 = start2, start1
        end1, end2 = end2, end1

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("X_train").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            target = []
            predicted = []



host = "localhost"
table = "posts"
conf = {"hbase.zookeeper.quorum": "localhost", "hbase.mapreduce.inputtable": "posts"}
keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"


def get_valid_items(items):
    message = drug_json = sideEffect_json = rowkey = ""
    for item in items:
        json_text = json.loads(item)
        rowkey = json_text["row"]
        if json_text["qualifier"] == "message":
            message = json_text["value"]
        if json_text["qualifier"] == "drug":
            drug_json = json_text["value"]
        if json_text["qualifier"] == "sideEffect":
            sideEffect_json = json_text["value"]
    drug_json_array = json.loads(drug_json)
    sideEffect_json_array = json.loads(sideEffect_json)
    if message is None or drug_json is None or sideEffect_json is None or drug_json == "null" or sideEffect_json == "null":
        return ([(rowkey, message, None, None, None, None)])
    if not len(drug_json_array) or not len(sideEffect_json_array):
        return ([(rowkey, message, None, None, None, None)])
    arr = []
    # print(drug_json, sideEffect_json)
    for drug_json in drug_json_array:
        drug_offset_start = drug_json["startNode"]["offset"]
        drug_offset_end = drug_json["endNode"]["offset"]
        for sideEffect_json in sideEffect_json_array:
            sideEffect_offset_start = sideEffect_json["startNode"]["offset"]
            row = rowkey + "-" + str(drug_offset_start) + "-" + str(sideEffect_offset_start)
            sideEffect_offset_end = sideEffect_json["endNode"]["offset"]
            arr.append(
                (row, message, drug_offset_start, drug_offset_end, sideEffect_offset_start, sideEffect_offset_end))
    return arr


def filter_rows(row):
    if row[0] is None or row[1] is None or row[2] is None or row[3] is None or row[4] is None or row[5] is None:
        return False
    return True


def save_record(rdd):
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"
    conf = {"hbase.zookeeper.quorum": "localhost",
            "hbase.mapred.outputtable": "drugSegments_test",
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    row_rdd = rdd.map(lambda x: x.split("\n"))
    row_rdd.foreach(get_valid_items)
    datamap = row_rdd.map(
        lambda x: (str(json.loads(x)["row"]), [str(json.loads(x)["row"]), "ml_results", "cats_json", "lolva"]))
    print(datamap)
    datamap.saveAsNewAPIHadoopDataset(conf=conf, keyConverter=keyConv, valueConverter=valueConv)


def get_input(row):
    print("here1**********************************////////////////////////")

    try:
        print(FLAGS.__flags, "====================================================")
        FLAGS_temp = FLAGS
    except:
        pass
    rowkey = row[0]
    message = row[1]
    start1 = row[2]
    end1 = row[3]
    start2 = row[4]
    end2 = row[5]
    if start2 < start1:  # swap if entity2 comes first
        start1, start2 = start2, start1
        end1, end2 = end2, end1
    print(start1, end1, start2, end2)
    # input_vec = generate_vector(message, start1, end1, start2, end2)
    # print(input_vec)
    # return (rowkey, input_vec)



def display(rdd):
    print(rdd)


hbase_rdd = sc.newAPIHadoopRDD(
    "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
    "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
    "org.apache.hadoop.hbase.client.Result",
    keyConverter=keyConv,
    valueConverter=valueConv,
    conf=conf)

hbase_rdd = hbase_rdd.map(lambda x: x[1]).map(
    lambda x: x.split("\n"))  # message_rdd = hbase_rdd.map(lambda x:x[0]) will give only row-key
data_rdd = hbase_rdd.flatMap(lambda x: get_valid_items(x))
data_rdd = data_rdd.filter(lambda x: filter_rows(x))
data_rdd.foreach(get_input)
# data_rdd.foreach(print)
# get_input(("125-234", "taxol causes pain", 0, 5, 13, 17))
# save_record(hbase_rdd)
# messages = hbase_rdd.take(1)
# for message in messages:
#     text = message.split("\n")
#     for row in text:
#         print(json.loads(row))

def get_value(row):
    print("**********************************************")
    graph = tf.Graph()
    rowkey = row[0]
    checkpoint_file = "/home/sahil/Desktop/Relation_Extraction/data/1485336002/checkpoints/model-300"
    print("Loading model................................")
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("X_train").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            batch_predictions = sess.run(predictions, {input_x: [row[1]], dropout_keep_prob: 1.0})
            print(batch_predictions)
            return (rowkey, batch_predictions)