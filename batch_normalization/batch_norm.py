import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.moving_averages import assign_moving_average

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

class Model:
    def __init__(self):
        # add placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
        self.y_ = tf.placeholder(tf.float32, shape = [None, 10],name='input_y')
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")

        # parameters
        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

        # Small epsilon value for the BN transform
        epsilon = 1e-3

        # inference
        with tf.name_scope("layer1"):
            w1_BN = tf.Variable(w1_initial)   # [784, 100]
            z1_BN = tf.matmul(self.x, w1_BN)  # [None, 100]
            BN_1 = self.batch_norm_wrap1(z1_BN, self.is_training)
            l1_BN = tf.nn.sigmoid(BN_1) 
        with tf.name_scope('layer2'):
            w2_BN = tf.Variable(w2_initial)
            z2_BN = tf.matmul(l1_BN,w2_BN)
            BN_2 = self.batch_norm_wrap1(z2_BN, self.is_training)
            l2_BN = tf.nn.sigmoid(BN_2)
        with tf.name_scope("layer3-softmax"):
            w3_BN = tf.Variable(w3_initial)
            b3_BN = tf.Variable(tf.zeros([10]))
            y_BN = tf.nn.softmax(tf.matmul(l2_BN, w3_BN)+b3_BN)

        # loss
        self.loss = -tf.reduce_sum(self.y_*tf.log(y_BN))
        tf.summary.scalar("loss", self.loss)
        # optimizer
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)   
        # accuracy
        correct_prediction_BN = tf.equal(tf.argmax(y_BN, 1), tf.argmax(self.y_, 1))
        self.accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy_BN)
        
    @staticmethod
    def batch_norm_wrap1(inputs, is_training, decay = 0.999, epsilon=1e-3):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training is not None:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
            # 上下文控制依赖，在 with 控制下，接下来的操作用到了上文中的操作，
            # 只有在执行了 train_mean, train_var 才执行接下来的 bn
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)

    @staticmethod
    def batch_norm_wrap2(x, is_training, eps=1e-05, decay=0.9, name=None):  # x.shape=[None, 100]
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = x.get_shape()[-1]    # tuple (100)
            moving_mean = tf.get_variable('mean', params_shape,
                                          initializer=tf.zeros_initializer,
                                          trainable=False) # [100,]
            moving_variance = tf.get_variable('variance', params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)

            def mean_var_with_update():
                #mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
                mean, variance = tf.nn.moments(x, x.get_shape()[-1], name='moments')

                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    # Return a tensor with the same shape and contents as input.
                    return tf.identity(mean), tf.identity(variance)

            # true_fn: The callable to be performed if pred is true.
            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_variance))
            if is_training:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x

tf.reset_default_graph()
model = Model()
saver = tf.train.Saver()
### ============================= train and valid =========================== ###
def train():
    acc = []
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('/tmp/summary/mnist' + '/train', sess.graph)
        #test_writer = tf.summary.FileWriter('/tmp/summary/mnist' + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(60)
            _, train_acc = sess.run([model.train_step, model.accuracy_BN],
                feed_dict={model.x:batch[0], model.y_:batch[1], model.is_training:True})
            if i%100 == 0:
                res = sess.run([model.accuracy_BN, model.loss, merged],
                              feed_dict={model.x:mnist.test.images,model.y_:mnist.test.labels,model.is_training:None})
                acc.append(res[0])
                if i%200 == 0:
                    print("{} steps, train accuracy is {}, val_acc is {}".format(i, train_acc, res[0]))
        saver.save(sess=sess, save_path='./temp-bn-save')
        writer.close()

### ==================================== test ============================================== ###         
def test():
    tf.reset_default_graph()
    model = Model()
    correct = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './temp-bn-save')
        for i in range(100):
            corr = sess.run([model.accuracy_BN],
                                 feed_dict = {model.x:[mnist.test.images[i]], 
                                             model.y_:[mnist.test.labels[i]],
                                              model.is_training:None})
            correct += corr
    print("test accuracy is {}".format(correct/100))

if __name__ == "__main__":
    train()
    print("======test=====")
    #test()