import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, LSTM, Dense, Activation, Input
from keras.callbacks import TensorBoard, ModelCheckpoint

is_train = True
is_save = True

char_dim = 100  #embedding size for single word
lstm_dim = 100  #num of hidden units in LSTM
dropout = 0.5
batch_size = 32
optimizer = "adam"
lr = 0.001 #learning rate
epoch = 10

#path for data
train_file = os.path.join("data", "example_medicine.train")
test_file = os.path.join("data", "example_medicine.test")
dev_file = os.path.join("data", "example_medicine.dev")
#path for data_medicine
# train_file = os.path.join("data", "example_medicine.train")
# test_file = os.path.join("data", "example_medicine.test")
# dev_file = os.path.join("data", "example_medicine.dev")
emb_file = os.path.join("data", "wiki_100.utf8")    #path for pre_trained embedding

