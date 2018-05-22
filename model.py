import numpy as np
import os
import sys
import random
from keras.models import Sequential, load_model, Model,optimizers
from keras.layers import Embedding, LSTM, Dense, Activation, Input, Bidirectional, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_contrib.layers import CRF

is_train = True
is_save = True
emb_dim = 100  #embedding size for single word
lstm_dim = 100  #num of hidden units in LSTM
dropout = 0.5
optimizer = "adam"
lr = 0.001 #learning rate
epoch = 10

def create_model(embedding_matrix, tag_index):
    #构建模型
    input = Input(shape=(None,))
    #word_index为用tokenizer处理后的word_index，embedding_matrix为词嵌入矩阵
    word_emb = Embedding(len(embedding_matrix), emb_dim, weights=[embedding_matrix])(input)
    bilstm = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2))(word_emb)
    #tag_index为tag与索引的映射，TimeDistributed为包装器，将一个层应用到输入的每一个时间步上
    # (每一个时间步上一个word，所以要应用到每一个时间步上，才能对每一个word进行标注预测)，
    # 最后输出维度为shape(None,None,len(tag_index)),每个节点的输出可以直接经过激活层进行判断，
    # 也可以输入到crf层进行进一步的处理
    # print("bilstm:", bilstm)
    dense = TimeDistributed(Dense(len(tag_index), activation='softmax'))(bilstm)
    # print("dense:", dense)
    # model = Model(inputs=input, outputs=dense)
    crf_layer = CRF(len(tag_index), sparse_target = True) #若后接CRF
    crf = crf_layer(dense)
    model = Model(inputs=input, outputs=crf)
    model.summary()

    # 编译模型
    optmr = optimizers.Adam(lr=lr, beta_1=0.5)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optmr,
    #               metrics=['accuracy'])
    #若使用crf作为最后一层，则修改模型编译的配置：
    model.compile(loss=crf_layer.loss_function, #注意这里的参数配置，crf_layer为对CRF()进行初始化的命名
                  optimizer=optmr,
                  metrics=[crf_layer.accuracy])

    return model