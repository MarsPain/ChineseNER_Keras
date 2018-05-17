import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model, Model,optimizers
from keras.layers import Embedding, LSTM, Dense, Activation, Input, Bidirectional, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_contrib.layers.crf import CRF

is_train = True
is_save = True

emb_dim = 100  #embedding size for single word
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

#构建模型
input = Input(shape=(None,))
#word_index为用tokenizer处理后的word_index，embedding_matrix为词嵌入矩阵
word_emb = Embedding(len(word_index)+1, emb_dim, weights=[embedding_matrix], dropout=0.5)(input)
bilstm = Bidirectional(LSTM(100, return_sequences=True), dropout=0.5)(word_emb)
#tag_index为tag与索引的映射，TimeDistributed为包装器，将一个层应用到输入的每一个时间步上，
# 最后输出维度为shape(None,None,len(tag_index)),每个维度的输出输入到crf层，用crf层
dense = TimeDistributed(Dense(len(tag_index)))(bilstm)
model = Model(inputs=inputs, outputs=dense)
# crf_layer = CRF(len(tag_index), sparse_target = True) #若后接CRF
# crf = crf_layer(dense)
# model = Model(inputs=inputs, outputs=crf)
model.summary()

# 编译模型
optmr = optimizers.Adam(lr=lr, beta_1=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#若使用crf作为最后一层，则修改模型编译的配置：
# model.compile(loss='crf.loss_function',
#               optimizer='adam',
#               metrics=['crf.accuracy'])