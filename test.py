import codecs
import tensorflow as tf
import CNN
import glob
import pandas as pd
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import sklearn as sk
from sklearn.metrics import confusion_matrix

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "data/1485336002/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=CNN.FLAGS.allow_soft_placement,
      log_device_placement=CNN.FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("X_train").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        target = []
        predicted = []
        batches = CNN.get_batches_test()
        for batch in batches:
            X_test , y_test = zip(*batch)
            scores, batch_predictions = sess.run([scores, predictions], {input_x: X_test,dropout_keep_prob: 1.0})
            for y in y_test:
                if y[0] == 1:
                    target = np.concatenate([target, [0]])
                else:
                    target = np.concatenate([target, [1]])
            predicted = np.concatenate([predicted, batch_predictions])
        print(float(sum(target == predicted))/ float(predicted.shape[0]))