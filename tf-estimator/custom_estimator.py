"""
tf.estimator framework provides a high-level API for training machine learning models,
defining train(), evaluate() and predict() operations, handling checkpointing, loading,
initializing, serving, building the graph and the session out of the box. There is a
small family of pre-made estimators, like the ones we used earlier, but it’s most likely
that you will need to build your own.
"""
import os
import tensorflow as tf
import tensorflow_estimator as estimator
from tensorflow.contrib.layers import embed_sequence
from tensorflow.contrib.estimator import binary_classification_head
# from baseline_estimator import train_and_evaluate
from tensorboard import summary as summary_lib

from nlp_estimator import vocab_size, embedding_size, model_dir, train_input_fn, eval_input_fn

# loss function
head = binary_classification_head() # this head uses `sigmoid_cross_entropy_with_logits` loss.

def cnn_model_fn(features, labels, mode, params):
    # mapping the features into our embedding layer
    print(features)
    input_layer = embed_sequence(
        ids=features["x"],
        vocab_size=vocab_size,
        embed_dim=embedding_size,
        initializer=params["embedding_initializer"]
    )  # [batch, sentence_len, embed_size]
    print(input_layer.shape)

    training = mode == estimator.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                    rate=0.2,
                                    training=training)
    conv = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu
    )   # [batch, sentence_len, filters]
    print(conv.shape)

    pool = tf.reduce_max(input_tensor=conv,axis=1)  # [batch, filters]

    hidden = tf.layers.dense(inputs=pool, units=250)

    dropout_hidden = tf.layers.dropout(inputs=hidden,
                                       rate=0.2,
                                       training=training)

    logits = tf.layers.dense(inputs=dropout_hidden, units=1)

    # This will be None when predicting
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    optimizer = tf.train.AdamOptimizer()

    def _train_op_fn(loss):
        return optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return head.create_estimator_spec(
        features= features,
        labels=labels,
        mode=mode,
        logits=logits,
        train_op_fn=_train_op_fn
    )

params = {"embedding_initializer" : tf.random_uniform_initializer(-0.8, 0.8)}
cnn_classifier = estimator.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=os.path.join(model_dir, "cnn"),
    params=params
)

if __name__ == "__main__":
    # train_and_evaluate(cnn_classifier)
    cnn_classifier.train(
        input_fn=train_input_fn,
        steps=2500
    )
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    tf.reset_default_graph()
    pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test.astype(bool),
                              num_thresholds=21)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(classifier.model_dir, 'eval'), sess.graph)
        writer.add_summary(sess.run(pr), global_step=0)
        writer.close()