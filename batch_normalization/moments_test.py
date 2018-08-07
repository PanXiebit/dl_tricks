import tensorflow as tf

a = tf.random_normal(shape=[3,4,5], dtype=tf.float32)

mean, var = tf.nn.moments(a, [0,1])

with tf.Session() as sess:
    print(sess.run(tf.shape(a)[:-1]))  # tensor
    print(a.get_shape()[:-1])
    print(mean.shape, var.shape)