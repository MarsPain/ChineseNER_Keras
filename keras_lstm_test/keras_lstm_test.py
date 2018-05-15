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
# print(embedding_index['hi'])
print("已匹配 %s 词向量" % len(embedding_index))

print("对文本数据进行处理")
texts = []  #text样本列表
labels_index = {}   #text原类名和label ID的字典
labels = [] #标签label的列表
for name in sorted(os.listdir(text_data_dir)):
    #遍历处理每种类别的text文件夹
    path = os.path.join(text_data_dir, name)
    if os.path.isdir(path):
        #为每种类别赋值label ID
        label_id = len(labels_index)    #一个小trick，根据
        labels_index[name] = label_id
        for file_name in sorted(os.listdir(path)):
            #遍历处理每种类别下的所有text文件
            if file_name.isdigit():
                with open(os.path.join(path, file_name), encoding='latin-1') as file_path:
                    # print(sys.version_info)
                    # 将每个text文件及其对应的label添加到相应数组中
                    texts.append(file_path.read())
                    labels.append(label_id)
# print(labels_index['alt.atheism'])
# print(labels_index['comp.sys.ibm.pc.hardware'])
print('找到 %s 个texts' % len(texts))

#利用keras的Tokenizer对所有texts进行处理，主要将texts中的word映射到相应的index
tokenizer = Tokenizer(nb_words=max_nb_words)
tokenizer.fit_on_texts(texts)   #以texts作为训练的文本列表
sequence = tokenizer.texts_to_sequences(texts)  #将文本列表转换为序列列表，每个序列对应一段文本
word_index = tokenizer.word_index   #将字符串(word)映射为它们作为索引的排名（如ChineseNER中用出现次数作为排名）
print(word_index['hi'])
print(sequence[2][:20])
print("找到 %s 个单词" % len(word_index))