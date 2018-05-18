import os
import re
from data_utils import load_sentences

#path for data
# train_file = os.path.join("data", "example.train")
# test_file = os.path.join("data", "example.test")
# dev_file = os.path.join("data", "example.dev")
#path for data_medicine
train_file = os.path.join("data", "example_medicine.train")
test_file = os.path.join("data", "example_medicine.test")
dev_file = os.path.join("data", "example_medicine.dev")
emb_file = os.path.join("data", "wiki_100.utf8")    #path for pre_trained embedding

#load data
train_sentences = load_sentences(train_file)
dev_sentences = load_sentences(dev_file)
test_sentences = load_sentences(test_file)
# print(train_sentences[5], '\n', dev_sentences[5], '\n', dev_sentences[5])