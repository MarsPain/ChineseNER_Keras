import keras
from keras.models import Sequential
from keras.layers import  Dense, Activation
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

#keras的序列（Sequential）模型
#用add()一个个将layer添加到模型中
# model = Sequential()
# # Dense表示全连接层，32表示units数量，即隐层神经元数量和输出的维度
# # activation表示该输出层的激活函数，input_dim对输入数据的shape进行指定，即每个样本的维度
# # 顺便复习下神经网络的层与层之间的数据传输：
# # 输入层维度为100，则输入层每个样本的shape=(1,100)
# # Dense层神经元为32，即输入层乘以一个参数矩阵W得到(1,32)的结果作为Dense层的输出
# # 那么参数矩阵维度为(100,32)
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(1, activation='softmax'))
# #编译模型
# model.compile(optimizer="rmsprop",
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# #进行训练
# data = np.random.random((1000,100)) #1000个样本，每个样本100个维度
# labels = np.random.randint(2, size=(1000,1))
# model.fit(data, labels, epochs=10, batch_size=32)

#如何处理多分类
#理解softmax和用于二分类的sigmoid的区别以及多分类中one-hot的意义
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer="rmsprop",
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# data = np.random.random((1000,100)) #1000个样本，每个样本100个维度
# labels = np.random.randint(10, size=(1000,1))
# one_hot_labels = keras.utils.to_categorical(labels, num_classes=10) #多分类要将label类型转换为onehot向量
# model.fit(data, one_hot_labels, epochs=10, batch_size=32)


#keras的函数式（Functional）模型
inputs = Input(shape=(784,))    #返回一个tensor
#分别定义layer，每个layer接受一个tensor，输出一个tensor
X = Dense(64, activation='relu')(inputs)
X = Dense(64, activation='relu')(X)
predictions = Dense(10, activation='softmax')(X)
#创建包含Input layer和三个Dense layer的Model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
data = np.random.random((1000,784)) #1000个样本，每个样本784个维度
labels = np.random.randint(10, size=(1000,1))
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10) #多分类要将label类型转换为onehot向量
model.fit(data, one_hot_labels, epochs=40)

