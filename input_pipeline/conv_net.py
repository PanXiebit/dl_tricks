import tensorflow as tf


class convNet(object):
    def __init__(self, image_size, image_depth, num_classes, learning_rate, keep_dropout):
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.lr = learning_rate
        self.keep_dropout = keep_dropout

        # add placeholder
        self.input_x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_depth], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")

        # inference and accuracy
        self.logits = self.forward(input=self.input_x)  # [None, 10]
        self.prediction = tf.argmax(self.logits, axis=1) # [None]
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.input_y)), tf.float32))

        # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits))

        # single train operation
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.train_step = tf.Variable(0, trainable=False, name='train_step')
        self.train_step = tf.assign(self.train_step, tf.add(self.train_step, tf.constant(1)))

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


    def _make_layer(self,inputs, filters, kernel_size, strides, padding="same", stddev=0.01):
        output = tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size, strides=strides, padding=padding,
                                  activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev),name="conv")
        output = tf.layers.batch_normalization(output, training=self.is_training, name="bn")
        output = tf.nn.leaky_relu(output)
        return output

    def forward(self, input):
        with tf.variable_scope("layer1"):
            output = self._make_layer(input, 32, (3,3),(1,1), stddev=0.0001) # [None, 32, 32, 32]

        with tf.variable_scope("layer2"):
            output = tf.layers.max_pooling2d(output, pool_size=(3,3), strides=(2,2), padding="valid") # (32-3)/2+1=14, [None,15,15,32]

        with tf.variable_scope("layer3"):
            output = self._make_layer(output, 32, (3,3),(1,1)) # [None,15,15,32]

        with tf.variable_scope("layer4"):
            output = tf.layers.average_pooling2d(output, pool_size=(3,3), strides=(2,2), padding="valid") #(15-3)/2+1=7, [None,7,7,32]

        with tf.variable_scope("layer5"):
            output = self._make_layer(output, 64, (3, 3), (1, 1))  # [None,6,6,64]

        with tf.variable_scope("layer6"):
            output = tf.layers.average_pooling2d(output, pool_size=(3, 3), strides=(2, 2), padding="valid") #(7-3)/2+1=2 [None, 3,3,64]

        with tf.variable_scope("fully-layer"):
            # output = tf.reshape(output, [-1,])
            output = tf.layers.flatten(output)  # [None, 576]
            output = tf.layers.dense(output, units=self.num_classes)
            logits = tf.nn.softmax(output)
        return logits



