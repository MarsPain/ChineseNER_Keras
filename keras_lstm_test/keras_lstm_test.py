import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

data_dir = "data"
text_data_dir = data_dir + '/20_newsgroup/'
max_sequence_length = 1000
max_nb_words = 20000
embedding_dim = 100
validation_split = 0.2
batch_size = 32

#将预训练的词向量文件中的词向量转换为容易查询的字典
embedding_index = {}
with open(os.path.join(data_dir, 'glove.6B.100d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]    #word
         #相应word的词向量vector，asarray与array都是将结构数据转化为array，区别在于asarray不会占用新内存
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector
print("已匹配 %s 词向量" % len(embedding_index))

print("对文本数据进行处理")
texts = []  #text样本列表
labels_index = {}   #字典
labels = [] #标签label的列表
for name in sorted(os.listdir(text_data_dir)):
    print(name)
