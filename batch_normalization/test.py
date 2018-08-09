import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from batch_norm import Model

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
tf.reset_default_graph()
def test():
    correct = 0
    preds = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./temp/bn-save.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./temp/"))  # restore global variables
        graph = tf.get_default_graph()
        accuracy = graph.get_tensor_by_name("accuracy:0")
        predict = graph.get_tensor_by_name("prediction:0")
        x = graph.get_tensor_by_name("input_x:0")
        y = graph.get_tensor_by_name("input_y:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        for i in range(100):
            corr, pred = sess.run([accuracy, predict],
                                  feed_dict = {x:[mnist.test.images[i]],
                                               y:[mnist.test.labels[i]],
                                               is_training:False})
            correct += corr
            preds.append(pred)
    print("test accuracy is {}".format(correct/100))
    print("prediction is {}".format(preds))

if __name__ == "__main__":
    test()
    print(tf.get_default_graph()) # 0x7fd7fdd11588