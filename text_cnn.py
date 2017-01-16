import tensorflow as tf

class TextCNN(object):
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters, emb_vec, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, emb_vec.shape], name="X_train")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="Y_train")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        self.input_x = tf.exp(self.input_x, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
