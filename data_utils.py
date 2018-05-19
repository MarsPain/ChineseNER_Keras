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
    data = []
    texts = []
    for s in sentences:
        string = [w[0] for w in s]
        string = " ".join(string)   #由于是处理中文，所以拼接的时候加上空格，否则tokenizer会将其识别为一个整体
        texts.append(string)

    #利用keras的tokenizer对texts进行处理
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)   #texts作为处理对象
    word_sequence = tokenizer.texts_to_sequences(texts)  #将文本转换为由索引表示的序列数据
    word_index = tokenizer.word_index   #word到索引的映射列表

    tags = [[char[-1] for char in s] for s in sentences]
    dict_tags = {}
    for items in tags:
        for tag in items:
            dict_tags[tag] = dict_tags[tag]+1 if tag in dict_tags else 1

    #传统方式获取tag_to_id的映射和tag的序列表示
    tag_to_id, id_to_tag = create_mapping(dict_tags)
    tag_index = tag_to_id
    print(tag_index)
    tags_sequence = [[tag_to_id[w[-1]] for w in s] for s in sentences]  #得到序列化的tags
    # print(tags_sequence,len(tags_sequence))

    #使用tokenizer获取tag_to_id的映射和tag的序列表示，但是tokenizer将“-”识别为空格
    # tags = []
    # for s in sentences:
    #     string = [w[-1] for w in s]
    #     string = " ".join(string)
    #     tags.append(string)
    # # print(tags)
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(tags)
    # tags_sequence = tokenizer.texts_to_sequences(tags)
    # tag_index = tokenizer.word_index
    # print("tag_to_id",tag_index)
    # print(tags_sequence,len(tags_sequence))

    data.append(word_sequence)
    data.append(word_index)
    data.append(tags_sequence)
    data.append(tag_index)
    return data

def create_mapping(dict):
    #根据字典dico创建双向映射
    sorted_items = sorted(dict.items(), key=lambda x: (-x[1], x[0]))    #按照词频排序
    # print(sorted_items)
    # for i, v in enumerate(sorted_items):
    #     print(i, v)
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}  #id（根据词频排序从0开始）到word
    item_to_id = {v: k for k, v in id_to_item.items()}  #反转映射
    return item_to_id, id_to_item