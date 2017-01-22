import CNN
from text_cnn import TextCNN
import data_helpers
import os
import numpy as np
import time
import tensorflow as tf
import datetime

with tf.Graph().as_default():
    start_time = time.time()
    session_conf = tf.ConfigProto(allow_soft_placement=CNN.FLAGS.allow_soft_placement,
        log_device_placement=CNN.FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(filter_sizes=list(map(int, CNN.FLAGS.filter_sizes.split(","))),
            num_filters=CNN.FLAGS.num_filters, vec_shape=(CNN.FLAGS.sequence_length + 2, CNN.FLAGS.embedding_size * CNN.FLAGS.window_size + 2 * CNN.FLAGS.distance_dim),
            l2_reg_lambda=CNN.FLAGS.l2_reg_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        # grad_summaries = []
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
        #         sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)
        # grad_summaries_merged = tf.merge_summary(grad_summaries)
        #
        # # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "data", timestamp))
        # print("Writing to {}\n".format(out_dir))
        #
        # # Summaries for loss and accuracy
        # loss_summary = tf.scalar_summary("loss", cnn.loss)
        # acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
        #
        # # Train Summaries
        # train_summary_op = tf.constant(1)
        # # train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        # train_summary_dir = os.path.join(out_dir, "summaries", "train")
        # train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
        #
        # # Dev summaries
        # dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        # saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_text_train, y_batch):
            feed_dict = {
                cnn.input_x: x_text_train,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: CNN.FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy, scores = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy, cnn.scores],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # train_summary_writer.add_summary(summaries, step)


        def dev_step(x_text_dev, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_text_dev,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # if writer:
            #     writer.add_summary(summaries, step)

        print("Loading batches...")
        batch_iter = CNN.get_batches()
        batch_iter_test = CNN.get_batches_test()
        for batch in batch_iter:
            X_train , y_train = zip(*batch)
            X_test, y_test = zip(*(next(batch_iter_test)))
            train_step(np.asarray(X_train), np.asarray(y_train))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % CNN.FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(np.asarray(X_test), np.asarray(y_test))
                print("")
            # if current_step % CNN.FLAGS.checkpoint_every == 0:
                # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # print("Saved model checkpoint to {}\n".format(path))
        print("Finished in time %0.3f" % (time.time() - start_time))