import tensorflow as tf


t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                   [[9, 3, 3], [4, 4, 4]],
                   [[5, 5, 5], [6, 6, 6]]])

with tf.Session() as sess:
    b = tf.slice(t, [1, 0, 0], [2, 1, 2])
    print(sess.run(b))
    # begin=[1,0,0] 表示起始位置,第一个1表示t中第一个维度中的index=1,也就是[[3,3,3],[4,4,4]]
    #                          第2个0表示第二个维度中的index=0,也就是 [3,3,3]
    #                          第3个0表示第三个维度中的index=0,也就是第一个3是起始位置
    # size=[2,1,2] 分别表示从