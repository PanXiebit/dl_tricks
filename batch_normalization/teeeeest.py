import tensorflow as tf

a = tf.random_normal([5,32,32,10])
b = tf.get_variable(name='b', shape=a.get_shape(), initializer=tf.zeros_initializer)

pred = tf.constant(True,dtype=tf.bool)
c = tf.cond(pred, lambda:(3,4), lambda:(4,5))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tf.shape(a))
    print(sess.run(c))
