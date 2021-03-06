import numpy as np
import os
import sys
import random
from keras.models import Sequential, load_model, Model,optimizers
from keras.layers import Embedding, LSTM, Dense, Activation, Input, Bidirectional, \
    TimeDistributed, Dropout, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_contrib.layers import CRF
from keras.initializers import Zeros
# from keras_crf_test import CRF

is_train = True
is_save = True
emb_dim = 100  #embedding size for single word
seg_dim = 0 #embedding size for word's seg_feature(词向量维度)
lstm_dim = 100  #num of hidden units in LSTM
dropout = 0.7
optimizer = "adam"
lr = 0.001 #learning rate

class Model_Class:
    def create_model(self, embedding_matrix, tag_index):
        #构建BiLSTM+CRF模型
        char_input = Input(shape=(None,))
        #word_index为用tokenizer处理后的word_index，embedding_matrix为词嵌入矩阵
        word_emb = Embedding(len(embedding_matrix), emb_dim, weights=[embedding_matrix], mask_zero=True)(char_input)
        #若需要词特征，则进行字词向量的拼接
        if seg_dim:
            seg_input = Input(shape=(None, ))
            seg_emb = Embedding(4, seg_dim)(seg_input)
            word_emb = concatenate([word_emb, seg_emb], axis=-1)
        bilstm = Bidirectional(LSTM(100, return_sequences=True, dropout=dropout))(word_emb)
        #tag_index为tag与索引的映射，TimeDistributed为包装器，将一个层应用到输入的每一个时间步上
        # (每一个时间步上一个word，所以要应用到每一个时间步上，才能对每一个word进行标注预测)，
        # 最后输出维度为shape(None,None,len(tag_index)),每个节点的输出可以直接经过激活层进行判断，
        # 也可以输入到crf层进行进一步的处理
        # print("bilstm:", bilstm)
        dense = TimeDistributed(Dense(len(tag_index)))(bilstm)
        # print("dense:", dense)
        # model = Model(inputs=input, outputs=dense)
        crf_layer = CRF(len(tag_index), sparse_target = True) #keras_contrib包的CRF层 ,sparse_target是什么参数？
        # crf_layer = CRF(len(tag_index)) #keras_crf层
        crf = crf_layer(dense)
        if seg_dim:
            model = Model(inputs=[char_input, seg_input], outputs=crf)
        else:
            model = Model(inputs=char_input, outputs=crf)
        model.summary()
        # 编译模型
        optmr = optimizers.Adam(lr=lr, beta_1=0.5)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=optmr,
        #               metrics=['accuracy'])
        #若使用crf作为最后一层，则修改模型编译的配置：
        model.compile(
                      loss=crf_layer.loss_function, #注意这里的参数配置，crf_layer为对CRF()进行初始化的命名
                      # loss=crf_layer.loss,    #keras_crf层
                      optimizer=optmr,
                      metrics=[crf_layer.accuracy])

        #单独的BiLSTM模型
        # input = Input(shape=(None,))
        # word_emb = Embedding(len(embedding_matrix), emb_dim, weights=[embedding_matrix], mask_zero=True)(input)
        # bilstm = Bidirectional(LSTM(100, return_sequences=True, dropout=dropout))(word_emb)
        # dense = TimeDistributed(Dense(len(tag_index)))(bilstm)
        # model = Model(inputs=input, outputs=dense)
        # optmr = optimizers.Adam(lr=lr, beta_1=0.5)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=optmr,
        #               metrics=['accuracy'])

        # 单独的CRF模型
        # input = Input(shape=(None,))
        # word_emb = Embedding(len(embedding_matrix), emb_dim, weights=[embedding_matrix], mask_zero=True)(input)
        # crf_layer = CRF(len(tag_index), sparse_target = True) #keras_contrib包的CRF层 ,sparse_target是什么参数？
        # crf = crf_layer(word_emb)
        # model = Model(inputs=input, outputs=crf)
        # model.summary()
        # optmr = optimizers.Adam(lr=lr, beta_1=0.5)
        # model.compile(
        #               loss=crf_layer.loss_function, #注意这里的参数配置，crf_layer为对CRF()进行初始化的命名
        #               optimizer=optmr,
        #               metrics=[crf_layer.accuracy])

        #序列式模型
        # model = Sequential()
        # model.add(Embedding(len(embedding_matrix), emb_dim, weights=[embedding_matrix], input_shape=(None,)))
        # model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=dropout)))
        # model.add(TimeDistributed(Dense(len(tag_index))))
        # crf_layer = CRF(len(tag_index), sparse_target = True)
        # model.add(crf_layer)
        # model.summary()
        # optmr = optimizers.Adam(lr=lr, beta_1=0.5)
        # model.compile(
        #       loss=crf_layer.loss_function, #注意这里的参数配置，crf_layer为对CRF()进行初始化的命名
        #       # loss=crf_layer.loss,    #keras_crf层
        #       optimizer=optmr,
        #       metrics=[crf_layer.accuracy])

        return model