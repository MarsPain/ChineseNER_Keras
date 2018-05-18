import os
import re
import codecs

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