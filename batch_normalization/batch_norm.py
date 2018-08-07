import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.contrib.layers import batch_norm

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

class Model:
    def __init__(self):
        # add placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
        self.y = tf.placeholder(tf.float32, shape = [None, 10],name='input_y')
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

        # parameters
        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

        # inference
        with tf.name_scope("layer1"):
            w1 = tf.Variable(w1_initial)   # [784, 100]
            z1 = tf.matmul(self.x, w1)  # [None, 100]
            BN = self.batch_norm_mine(z1, self.is_training)
            # BN_1 = batch_norm(z1_BN, center=True, scale=True, is_training=self.is_training)
            l1 = tf.nn.sigmoid(BN)
        with tf.name_scope('layer2'):
            w2 = tf.Variable(w2_initial)
            z2 = tf.matmul(l1,w2)
            BN = self.batch_norm_mine(z2, self.is_training)
            # BN_2 = batch_norm(z2_BN, center=True, scale=True, is_training=self.is_training)
            l2 = tf.nn.sigmoid(BN)
        with tf.name_scope("layer3-softmax"):
            w3 = tf.Variable(w3_initial)
            b3 = tf.Variable(tf.zeros([10]))
            y = tf.nn.softmax(tf.matmul(l2, w3)+b3)

        # loss
        self.loss = -tf.reduce_sum(self.y*tf.log(y))
        tf.summary.scalar("loss", self.loss)
        # optimizer
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)   
        # accuracy
        self.predict = tf.argmax(y, axis=1)
        correct_prediction = tf.equal(self.predict, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

    def batch_norm_mine(self,
                       inputs,
                       is_training=True,
                       epsilon=1e-8,
                       decay = 0.9):
        """

        :param inputs: [batch, height, width, depth]`
        :param epsilon:
        :param decay:
        :param is_training:
        :return:
        """
        with tf.variable_scope("batch-normalization"):
            pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name="pop-mean") #[depth]
            pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name="pop-var")

            def mean_and_var_update():
                axes = list(range(len(inputs.get_shape()) - 1))
                # used with convolutional filters with shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`
                # 每一个kernel所对应的batch个图，求它们所有像素的mean和variance
                batch_mean, batch_var = tf.nn.moments(inputs, axes, name="moments")  # [depth]

                # 用滑动平均值来统计整体的均值和方差,在训练阶段并用不上,在测试阶段才会用,这里是保证在训练阶段计算了滑动平均值
                # moving_average_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                # 也可用 assign_moving_average(pop_mean, batch_mean, decay)
                # moving_average_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                # 也可用 assign_moving_average(pop_var, batch_var, decay)

                # control_dependencies 的作用是,但我们在训练阶段引用 batch_mean,batch_batch_var 时,
                # tf.identity 是一个op操作,因此会先执行control_dependencies中的参数操作.
                # A list of `Operation` or `Tensor` objects which must be executed or computed
                # before running the operations defined in the context.
                with tf.control_dependencies([assign_moving_average(pop_mean, batch_mean, decay),
                                              assign_moving_average(pop_var, batch_var, decay)]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, variance = tf.cond(is_training, mean_and_var_update, lambda:(pop_mean, pop_var))

            if is_training is not None:
                beta = tf.Variable(initial_value=tf.zeros(inputs.get_shape()[-1]), name="shift")
                gamma = tf.Variable(initial_value=tf.ones(inputs.get_shape()[-1]), name="scale")
                return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, mean, variance, None, None, epsilon)


model = Model()
saver = tf.train.Saver()
### ============================= train and valid =========================== ###
def train():
    acc = []
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graph/train',graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(60)
            _, train_acc = sess.run([model.train_step, model.accuracy],
                feed_dict={model.x:batch[0], model.y:batch[1], model.is_training:True})
            if i%100 == 0:
                res = sess.run([model.accuracy, model.loss, merged],
                              feed_dict={model.x:mnist.test.images,model.y:mnist.test.labels,
                                         model.is_training:None})
                acc.append(res[0])
                if i%200 == 0:
                    print("{} steps, train_acc is {}, val_acc is {}".format(i, train_acc, res[0]))
        saver.save(sess=sess, save_path='./temp/bn-save')
    writer.close()

### ==================================== test ============================================== ###         
def test():
    tf.reset_default_graph()
    model = Model()
    correct = 0
    preds = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph("./temp/bn-save.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./temp/"))
        for i in range(100):
            corr, pred = sess.run([model.accuracy, model.predict],
                                 feed_dict = {model.x:[mnist.test.images[i]],
                                              model.y:[mnist.test.labels[i]],
                                              model.is_training:None})
            correct += corr
            preds.append(pred)
    print("test accuracy is {}".format(correct/100))
    print("prediction is {}".format(preds))

if __name__ == "__main__":
    train()
    print("======test=====")
    test()