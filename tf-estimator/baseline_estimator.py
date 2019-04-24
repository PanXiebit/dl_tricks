import os
import numpy as np
import tensorflow as tf
from nlp_estimator import vocab_size, model_dir, train_input_fn, eval_input_fn, y_test, word_inverted_index
from tensorflow_estimator import estimator
from tensorboard import summary as summary_lib
import matplotlib.pyplot as plt


column = tf.feature_column.categorical_column_with_identity("x", vocab_size)
print(column)
linear_classifier = estimator.LinearClassifier(
    feature_columns=[column],
    model_dir= os.path.join(model_dir, "bow_sparse")
)

embedding_size = 50
word_embedding_column = tf.feature_column.embedding_column(
    column, dimension=embedding_size)
print(word_embedding_column)


dnn_classifier = estimator.DNNClassifier(
    hidden_units=[100],
    feature_columns=[word_embedding_column],
    model_dir=os.path.join(model_dir, 'bow_embeddings'))

# all_classifiers = {}

def train_and_evaluate(classifier):
    # Save a reference to the classifier to run predictions later
    # all_classifiers[classifier.model_dir] = classifier
    classifier.train(
        input_fn=train_input_fn,
        steps=2500
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    predictions = np.array([p['logistic'][0] for p in classifier.predict(input_fn=eval_input_fn)])
    # Reset the graph to be able to reuse name scopes
    tf.reset_default_graph()
    pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test.astype(bool),
                              num_thresholds=21)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(classifier.model_dir, 'eval'), sess.graph)
        writer.add_summary(sess.run(pr), global_step=0)
        writer.close()




if __name__ == "__main__":
    # linear classifier
    # train_and_evaluate(linear_classifier)
    # weights = linear_classifier.get_variable_value('linear/linear_model/x/weights').flatten()
    # print(weights.shape)
    # sorted_indexes = np.argsort(weights)
    # extremes = np.concatenate((sorted_indexes[-8:], sorted_indexes[:8]))
    # extreme_weights = sorted([(weights[i], word_inverted_index[i]) for i in extremes])
    #
    # y_pos = np.arange(len(extreme_weights))
    #
    # plt.bar(y_pos, [pair[0] for pair in extreme_weights], align='center', alpha=0.5)
    # plt.xticks(y_pos, [pair[1] for pair in extreme_weights], rotation=45, ha='right')
    # plt.ylabel('Weight')
    # plt.title('Most significant tokens')
    # plt.show()

    # dnn classifier
    train_and_evaluate(dnn_classifier)