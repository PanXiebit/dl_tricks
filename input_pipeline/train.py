from conv_net import convNet
from cifar_pipeline import read_cifar10
import os
import tensorflow as tf
import numpy as np


# Hyperparameters
BATCH_SIZE = 64
LR = 0.01
IMAGE_SIZE = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10
KEEP_DROPOUT = 0.5
NUM_EPOCHS = 5

def getDataFiles():
    data_file = []
    for i in range(1,6):
        file_name = os.path.join("dataset", "data_batch_{}".format(i))
        data_file.append(file_name)
    return data_file

model = convNet(image_size=IMAGE_SIZE, image_depth=IMAGE_DEPTH, num_classes=NUM_CLASSES,
                learning_rate=LR, keep_dropout=KEEP_DROPOUT)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(NUM_EPOCHS):
        data_file = getDataFiles()
        image_batch, labels_batch = read_cifar10(data_file, batch_size=64)
        image_batch, labels_batch = np.array(image_batch), np.array(labels_batch)
        _, loss, acc, step = sess.run([model.train_op,model.loss,model.accuracy,model.global_step],
                                feed_dict={model.input_x:image_batch,
                                           model.input_y:labels_batch})

        if step % 100 == 0:
            print("{} steps, loss is {}, acc is {}".format(step, loss, acc))



