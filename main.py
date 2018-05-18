import os
import re
from data_utils import load_sentences, prepare_data

#path for data
# train_file = os.path.join("data", "example.train")
# test_file = os.path.join("data", "example.test")
# dev_file = os.path.join("data", "example.dev")
#path for data_medicine
train_file = os.path.join("data", "example_medicine.train")
test_file = os.path.join("data", "example_medicine.test")
dev_file = os.path.join("data", "example_medicine.dev")
emb_file = os.path.join("data", "wiki_100.utf8")    #path for pre_trained embedding

#load data and get sentences
train_sentences = load_sentences(train_file)
dev_sentences = load_sentences(dev_file)
test_sentences = load_sentences(test_file)
# print(train_sentences[5], '\n', dev_sentences[5], '\n', dev_sentences[5])

#prepare data，对sentences进行处理得到sentence的序列化表示，以及word到ID的映射序列
train_data = prepare_data(train_sentences)
print("train_data:", "\n", train_data[0][5], "\n", train_data[1], "\n", train_data[2])