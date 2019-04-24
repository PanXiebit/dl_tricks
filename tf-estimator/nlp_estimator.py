import os
import string
import tempfile
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence


tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

vocab_size = 5000
sentence_size = 200
embedding_size = 50
# model_dir = tempfile.mkdtemp()
model_dir = "./model/"

# we assign the first indices in the vocabulary to special tokens that we use
# for padding, as start token, and for indicating unknown words
pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2

print("Loading data...")

(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(
    num_words=vocab_size, start_char=start_id, oov_char=oov_id,
    index_from=index_offset)

print(len(y_train), "train examples")
print(len(y_test), "test examples")
# print(x_train_variable[:5])

print("Pad sequences (samples x time)")

x_train = sequence.pad_sequences(x_train_variable,
                                 maxlen=sentence_size,
                                 truncating="post", # remove values from sequences larger than `maxlen`, either at the beginning or at the end of the sequences.
                                 padding="post", # padding: String, 'pre' or 'post'
                                 value=pad_id)

x_test = sequence.pad_sequences(x_test_variable,
                                maxlen=sentence_size,
                                truncating="post", # remove values from sequences larger than `maxlen`, either at the beginning or at the end of the sequences.
                                padding="post", # padding: String, 'pre' or 'post'
                                value=pad_id)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

word_index = imdb.get_word_index()
# print(len(word_index))
word_inverted_index = {v + index_offset: k for k, v in word_index.items()}

# The first indexes in the map are reserved to represent things other than tokens
word_inverted_index[pad_id] = '<PAD>'
word_inverted_index[start_id] = '<START>'
word_inverted_index[oov_id] = '<OOV>'
# print(len(word_inverted_index))

# for i, (word, index) in enumerate(word_index.items()):
#     if i>10:
#         break
#     print(word, index)
#
# for i in range(0, 10):
#     print(i, word_inverted_index[i])

def index_to_text(indexes):
    return ' '.join([word_inverted_index[i] for i in indexes])

# print(index_to_text(x_train_variable[0]))

### From arrays to tensors
x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])

def parser(x, length, y):
    features = {"x":x, "len":length}
    return features, y

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train_variable))
    dataset = dataset.batch(60)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

