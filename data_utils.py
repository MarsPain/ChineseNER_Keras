import os
import re
import codecs
from keras.preprocessing.text import Tokenizer

#加载数据并用嵌套列表存储每个sentence以及sentence中的每个word以及相应的标注
def load_sentences(path):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = re.sub('\d', '0', line.rstrip()) #将所有数字转为0
        if not line:
            if len(sentence) > 0:
                if 'START' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = '$' + line[1:]
                word = line.split()
            else:
                word = line.split()
            sentence.append(word)
    if len(sentence) > 0:
        if 'START' not in sentence[0][0]:
            sentences.append(sentence)

    return sentences

def prepare_data(sentences):
    train_data = []
    texts = []
    for s in sentences:
        string = [w[0] for w in s]
        string = " ".join(string)   #由于是处理中文，所以拼接的时候加上空格，否则tokenizer会将其识别为一个整体
        texts.append(string)

    #利用keras的tokenizer对texts进行处理
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)   #texts作为处理对象
    sequence = tokenizer.texts_to_sequences(texts)  #将字符串转换为由索引表示的序列数据
    word_index = tokenizer.word_index   #word到索引的映射列表

    train_data.append(sequence)
    train_data.append(word_index)
    return train_data