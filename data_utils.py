import os
import re
import codecs
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import jieba

emb_dim = 100
max_sequence_length = 50

#加载数据并用嵌套列表存储每个sentence以及sentence中的每个word以及相应的标注
def load_sentences(path):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip()    #如果标签类型用数字表示，则不能用下面的分割方法，会导致所有数字为0
        # line = re.sub('\d', '0', line.rstrip()) #将所有数字转为0
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

def prepare_data(sentences, seg_dim, tag_index):
    data = []
    texts = []
    seg_sequence = []
    for s in sentences:
        # tag = [w[-1] for w in s]
        # print("tag:", tag)
        string = [w[0] for w in s]
        string = " ".join(string)   #由于是处理中文，所以拼接的时候加上空格，否则tokenizer会将其识别为一个整体
        texts.append(string)
        #如果需要词特征和词向量
        if seg_dim:
            segs_features = get_seg_features("".join(string))
            seg_sequence.append(segs_features)
    if len(seg_sequence) > 0:
        seg_sequence = np.asarray(seg_sequence)
        print("seg_sequence:", type(seg_sequence), seg_sequence.shape)


    #利用keras的tokenizer对texts进行处理
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)   #texts作为处理对象
    word_sequence = tokenizer.texts_to_sequences(texts)  #将文本转换为由索引表示的序列数据
    #是否需要对word_sequence进行填充处理？
    word_sequence = pad_sequences(word_sequence, maxlen=max_sequence_length)
    print("word_sequence:", type(word_sequence), word_sequence.shape)
    word_index = tokenizer.word_index   #word到索引的映射列表

    #得到序列化的tags
    # tags_sequence = np.asarray([[tag_index[w[-1]] for w in s] for s in sentences])
    # tags_sequence = np.asarray([tag_index[w[-1]] for w in s for s in sentences])
    # print("tags_sequence:", tags_sequence.shape, type(tags_sequence))
    all_label = []
    for s in sentences:
        for i in range(max_sequence_length):
            if i < len(s):
                all_label.append(tag_index[s[i][-1]])
            else:
                all_label.append(0)
    # print("all_label:", all_label)

    print("tag_index:", tag_index)
    imp_weight_dict = {}
    set = ("B-1", "I-1", "E-1", "O-1", "S-1")
    for key, value in tag_index.items():
        if  key in set:
            imp_weight_dict[value] = 2
    print(imp_weight_dict)

    #将多类别label转换为one-hot向量
    all_label = to_categorical(all_label)   #测试下to_categorical的参数即使是嵌套数组，依然能得到整个列表的one-hot表示？
    # print("all_label:", all_label)
    labels = []
    for i in range(len(sentences)):
        labels.append(all_label[i*max_sequence_length:(i+1)*max_sequence_length])
    labels = np.asarray(labels)
    print("labels:", labels.shape)

    for i in range(len(labels)):
        if np.argmax(labels[i]) in imp_weight_dict:
            labels[i] = labels[i] * imp_weight_dict[np.argmax(labels[i])]

    data.append(word_sequence)
    data.append(word_index)
    data.append(labels)
    if len(seg_sequence) > 0:
        data.append(seg_sequence)
    return data

def get_tag_index(sentences):
    tags = [[char[-1] for char in s] for s in sentences]
    # print("tags:", tags)
    dict_tags = {}
    for items in tags:
        for tag in items:
            dict_tags[tag] = dict_tags[tag]+1 if tag in dict_tags else 1
    # print(dict_tags)

    #传统方法获取tag_to_id的映射和tag的序列表示
    tag_to_id, id_to_tag = create_mapping(dict_tags)
    tag_index = tag_to_id
    # print("tag_index:",tag_index)
    return tag_index, id_to_tag

    #传统方式得到序列化的tags
    # # tags_sequence = np.asarray([[tag_index[w[-1]] for w in s] for s in sentences])
    # # tags_sequence = np.asarray([tag_index[w[-1]] for w in s for s in sentences])
    # # print("tags_sequence:", tags_sequence.shape, type(tags_sequence))
    # all_label = []
    # for s in sentences:
    #     for i in range(max_sequence_length):
    #         if i < len(s):
    #             all_label.append(tag_index[s[i][-1]])
    #         else:
    #             all_label.append(0)
    # # print("all_label:", all_label)
    # #将多类别label转换为one-hot向量
    # all_label = to_categorical(all_label)   #测试下to_categorical的参数即使是嵌套数组，依然能得到整个列表的one-hot表示？
    # # print("all_label:", all_label)
    # labels = []
    # for i in range(len(sentences)):
    #     labels.append(all_label[i*max_sequence_length:(i+1)*max_sequence_length])
    # labels = np.asarray(labels)
    # print("labels:", labels.shape)

    #使用tokenizer获取tag_to_id的映射和tag的序列表示，但是存在tokenizer将“-”识别为空格所以行不通
    # tags = []
    # for s in sentences:
    #     string = [w[-1] for w in s]
    #     string = " ".join(string)
    #     tags.append(string)
    # print(tags)
    # tags_sequence = text_to_word_sequence(tags, split=" ")
    # tag_index = tokenizer.word_index
    # print("tag_to_id",tag_index)
    # print(tags_sequence,len(tags_sequence))

    #使用keras的preprocessing.text包中的text_to_word_sequence和one-hot获取tag的序列表示，
    # 好像无法对嵌套列表进行处理？
    # tag_to_id, id_to_tag = create_mapping(dict_tags)
    # tag_index = tag_to_id
    # # print(tag_index)
    # tags_sequence = [[tag_index[w[-1]] for w in s] for s in sentences]
    # tags_sequence = text_to_word_sequence(np.asarray(tags_sequence), split=" ")

#根据字典dico创建双向映射
def create_mapping(dict):
    # print("dict:", dict)
    sorted_items = sorted(dict.items(), key=lambda x: (-x[1], x[0]))    #按照词频排序
    # print("sorted_items:", sorted_items)
    # for i, v in enumerate(sorted_items):
    #     print(i, v)
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}  #id（根据词频排序从0开始）到word
    item_to_id = {v: k for k, v in id_to_item.items()}  #反转映射
    return item_to_id, id_to_item

#将预训练的词向量存储为易查询的字典
def create_emb_index(emb_file):
    embedding_index = {}
    with open(emb_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]    #word
             #相应word的词向量vector，asarray与array都是将结构数据转化为array，区别在于asarray不会占用新内存
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    # print(embedding_index['的'])
    print("已匹配 %s 词向量" % len(embedding_index))
    return embedding_index

def create_emb_matrix(word_index, embedding_index):
    embedding_matrix = np.zeros((len(word_index)+1, emb_dim))
    for word, i in word_index.items():
        #此处为什么不能用embedding_index[word]获取词向量？因为用get(word)替代[i],遇到key不存在不会报异常，而是返回None
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:    #若该词存在于embedding_index中，则初始化，否则保持为0向量
            embedding_matrix[i] = embedding_vector
    # print(embedding_matrix[76])
    # print(embedding_matrix.shape)
    print("embedding_matrix构建完成")
    return embedding_matrix

#利用jieba分词对sentence进行分词作为词特征
def get_seg_features(string):
    seg_feature = []
    # print(string)
    for word in jieba.cut(string):
        # print(word)
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    while len(seg_feature) < 50:
        seg_feature.extend([0])
    if len(seg_feature) > 50:
        seg_feature = seg_feature[:50]
    # print(len(seg_feature))
    return seg_feature